#!/usr/bin/env python3
"""
Improved heuristic relabeling v2 for v3 labels.

Key improvements:
1. Detect SECTION_HEADER first (all caps, generic words, top 5%)
2. Only look for RECIPE_TITLE after section headers
3. Recipe title must come before first INGREDIENT_LINE
4. Stricter matching criteria
"""

import json
import sys
from pathlib import Path

def detect_section_header(words, bboxes, height):
    """
    Detect section headers - typically ALL CAPS, generic terms, at top of page.

    Examples: "BREAD AND BREAD MAKING", "COCOA AND CHOCOLATE", "FRUIT DRINKS"
    """
    SECTION_INDICATORS = [
        'AND', 'MAKING', 'THE', 'FOR', 'OF', 'WITH', 'TO', 'IN', 'ON'
    ]

    for start in range(min(50, len(words))):  # Only check first 50 words
        for length in [2, 3, 4, 5, 6]:
            if start + length > len(words):
                break

            end = start + length
            sequence = words[start:end]

            # Must have bbox
            if len(bboxes) < end:
                continue

            sequence_bboxes = bboxes[start:end]

            # Check Y position - must be in top 5% of page
            avg_y = sum(bbox[1] for bbox in sequence_bboxes) / length
            y_percent = avg_y / height * 100

            if y_percent > 5:
                continue

            # Check if ALL CAPS
            if not all(w.isupper() or not w.isalpha() for w in sequence):
                continue

            # Check if contains section indicator words
            if not any(indicator in sequence for indicator in SECTION_INDICATORS):
                continue

            # This looks like a section header!
            return start, end, ' '.join(sequence)

    return None, None, None


def detect_recipe_title(words, bboxes, labels, width, height, section_header_end=None):
    """
    Detect recipe titles using improved heuristics.

    Key rules:
    - Must come AFTER section header (if present)
    - Must come BEFORE first ingredient
    - 2-5 words long
    - Contains food/recipe terms OR "How to" pattern
    - Position: 5-20% of page (not top 5% - that's section headers)
    - Not all caps (that's section headers)
    """

    # Food/recipe related terms
    FOOD_TERMS = {
        'bread', 'tea', 'coffee', 'cocoa', 'chocolate', 'punch', 'soup', 'cake',
        'pie', 'cookies', 'rolls', 'muffins', 'pudding', 'sauce', 'salad', 'meat',
        'fish', 'chicken', 'beef', 'pork', 'egg', 'omelet', 'pancake', 'waffle',
        'toast', 'biscuit', 'scone', 'tart', 'cream', 'custard', 'jelly', 'jam',
        'syrup', 'broth', 'stew', 'roast', 'fried', 'baked', 'boiled', 'steamed'
    }

    n = len(words)
    if n == 0:
        return labels, None

    # Find first INGREDIENT_LINE token (if any)
    first_ingredient_idx = None
    for i, label in enumerate(labels):
        if label == 3:  # INGREDIENT_LINE
            first_ingredient_idx = i
            break

    # Search window: after section header, before ingredients
    search_start = section_header_end if section_header_end else 0
    search_end = first_ingredient_idx if first_ingredient_idx else min(n, 150)

    if search_start >= search_end:
        return labels, None

    candidates = []

    for start in range(search_start, search_end):
        for length in [2, 3, 4, 5]:
            if start + length > search_end:
                break

            end = start + length
            sequence = words[start:end]
            sequence_bboxes = bboxes[start:end]

            # Skip if any word is missing bbox
            if len(sequence_bboxes) < length:
                continue

            # Check Y position - must be 5-30% of page
            avg_y = sum(bbox[1] for bbox in sequence_bboxes) / length
            y_percent = avg_y / height * 100

            if y_percent < 5 or y_percent > 30:
                continue

            # Join sequence
            text = ' '.join(sequence)
            text_lower = text.lower()

            # Skip if ALL CAPS (that's likely a section header we missed)
            if all(w.isupper() or not w.isalpha() for w in sequence):
                continue

            # Score this candidate
            score = 0
            reasons = []

            # Pattern 1: "How to" format (very strong signal)
            if text_lower.startswith('how to'):
                score += 15
                reasons.append("'How to' pattern")

            # Pattern 2: Contains food terms
            words_in_seq = [w.lower().rstrip('.,;:') for w in sequence]
            food_matches = [w for w in words_in_seq if w in FOOD_TERMS]
            if food_matches:
                score += 5 * len(food_matches)
                reasons.append(f"food terms: {food_matches}")

            # Pattern 3: Ends with period (common for titles)
            if text.endswith('.'):
                score += 3
                reasons.append("ends with period")

            # Pattern 4: Good Y position (5-15% is ideal)
            if 5 <= y_percent <= 15:
                score += 5
                reasons.append(f"ideal position (y={y_percent:.1f}%)")
            elif 15 < y_percent <= 20:
                score += 2
                reasons.append(f"good position (y={y_percent:.1f}%)")

            # Pattern 5: Proper length (2-4 words is most common)
            if 2 <= length <= 4:
                score += 3
                reasons.append(f"{length} words")

            # Pattern 6: First letter capitalized
            if sequence[0][0].isupper():
                score += 1
                reasons.append("capitalized")

            # Pattern 7: Comes after section header
            if section_header_end and start >= section_header_end:
                score += 5
                reasons.append("after section header")

            # Pattern 8: Comes before ingredients
            if first_ingredient_idx and end < first_ingredient_idx:
                score += 3
                reasons.append("before ingredients")

            # Require minimum score
            if score >= 12:  # Threshold for considering as title
                candidates.append({
                    'start': start,
                    'end': end,
                    'text': text,
                    'score': score,
                    'reasons': reasons,
                    'y_percent': y_percent
                })

    # If we found candidates, pick the best one
    if candidates:
        # Sort by score (desc) then by position (asc)
        candidates.sort(key=lambda x: (-x['score'], x['start']))
        best = candidates[0]

        # Apply RECIPE_TITLE label
        for i in range(best['start'], best['end']):
            labels[i] = 2  # RECIPE_TITLE = 2

        return labels, best

    return labels, None


def relabel_jsonl(input_path, output_path):
    """Re-label JSONL file with improved v2 heuristics."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")
    print()

    titles_found = 0
    section_headers_found = 0
    total_pages = 0

    with open(input_path) as infile, open(output_path, 'w') as outfile:
        for line in infile:
            page = json.loads(line)
            total_pages += 1

            page_num = page['page_num']
            words = page['words']
            bboxes = page['bboxes']
            labels = page.get('labels', [9] * len(words))  # Default to 'O'
            width = page['width']
            height = page['height']

            print(f"Page {page_num}:")

            # Step 1: Detect section header
            sh_start, sh_end, sh_text = detect_section_header(words, bboxes, height)

            if sh_start is not None:
                # Label section header
                for i in range(sh_start, sh_end):
                    labels[i] = 1  # SECTION_HEADER = 1
                section_headers_found += 1
                print(f"  📋 Section header: '{sh_text}'")

            # Step 2: Detect recipe title (after section header)
            new_labels, title_info = detect_recipe_title(
                words, bboxes, labels, width, height,
                section_header_end=sh_end
            )

            if title_info:
                titles_found += 1
                print(f"  ✓ Recipe title: '{title_info['text']}' (score={title_info['score']}, y={title_info['y_percent']:.1f}%)")
                print(f"    Reasons: {', '.join(title_info['reasons'])}")

            # Update page
            page['labels'] = new_labels

            # Write to output
            outfile.write(json.dumps(page) + '\n')

    print()
    print("=" * 80)
    print(f"SUMMARY:")
    print(f"  Total pages: {total_pages}")
    print(f"  Section headers found: {section_headers_found} ({section_headers_found/total_pages*100:.1f}%)")
    print(f"  Recipe titles found: {titles_found} ({titles_found/total_pages*100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python improve_v3_heuristics_v2.py <input_jsonl> [output_jsonl]")
        print()
        print("Example:")
        print("  python improve_v3_heuristics_v2.py \\")
        print("    data/processed/v3_headers_titles/boston_v3_suggested.jsonl \\")
        print("    data/processed/v3_headers_titles/boston_v3_improved_v2.jsonl")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.jsonl', '_improved_v2.jsonl')

    relabel_jsonl(input_file, output_file)
