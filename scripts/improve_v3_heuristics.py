#!/usr/bin/env python3
"""
Improved heuristic relabeling for v3 labels with focus on RECIPE_TITLE detection.

Based on manual label analysis of 7 demo pages.
"""

import json
import sys
from pathlib import Path

def detect_recipe_title(words, bboxes, labels, width, height):
    """
    Detect recipe titles using improved heuristics.

    Patterns from manual analysis:
    - 2-4 words long
    - Top 20% of page (with some exceptions up to 50%)
    - Ends with period (~85% of time)
    - Contains food-related terms or "How to" pattern
    - Comes before ingredients
    """

    # Food/recipe related terms
    FOOD_TERMS = {
        'bread', 'tea', 'coffee', 'cocoa', 'chocolate', 'punch', 'soup', 'cake',
        'pie', 'cookies', 'rolls', 'muffins', 'pudding', 'sauce', 'salad', 'meat',
        'fish', 'chicken', 'beef', 'pork', 'egg', 'omelet', 'pancake', 'waffle',
        'toast', 'biscuit', 'scone', 'tart', 'cream', 'custard', 'jelly', 'jam',
        'syrup', 'broth', 'stew', 'roast', 'fried', 'baked', 'boiled', 'steamed'
    }

    # Words to exclude (page headers, etc.)
    EXCLUDE_TERMS = {
        'boston', 'cooking-school', 'cook', 'book', 'chapter', 'contents',
        'index', 'page', 'continued'
    }

    # Section header patterns (typically ALL CAPS, multiple words)
    SECTION_PATTERNS = [
        'AND', 'MAKING', 'THE', 'FOR', 'OF', 'WITH'  # Common words in section headers
    ]

    n = len(words)
    if n == 0:
        return labels

    # Find sequences of 2-4 consecutive words that could be titles
    candidates = []

    for start in range(n):
        for length in [2, 3, 4, 5]:  # Allow up to 5 words
            if start + length > n:
                break

            end = start + length
            sequence = words[start:end]
            sequence_bboxes = bboxes[start:end]

            # Skip if any word is missing bbox
            if len(sequence_bboxes) < length:
                continue

            # Check Y position (top 30% of page, with some leniency)
            avg_y = sum(bbox[1] for bbox in sequence_bboxes) / length
            y_percent = avg_y / height * 100

            if y_percent > 30:  # Skip if too far down the page
                continue

            # Join sequence
            text = ' '.join(sequence)
            text_lower = text.lower()

            # Skip if contains excluded terms
            if any(term in text_lower for term in EXCLUDE_TERMS):
                continue

            # Check if this looks like a section header (all caps with generic words)
            is_likely_section_header = False
            if all(w.isupper() or not w.isalpha() for w in sequence):
                # All caps - check if it contains section pattern words
                if any(pattern in sequence for pattern in SECTION_PATTERNS):
                    is_likely_section_header = True

            # Score this candidate
            score = 0
            reasons = []

            # Penalize section headers
            if is_likely_section_header:
                score -= 10
                reasons.append("likely section header (all caps)")

            # Pattern 1: "How to" format
            if text_lower.startswith('how to'):
                score += 10
                reasons.append("starts with 'How to'")

            # Pattern 2: Contains food terms
            words_in_seq = [w.lower().rstrip('.,') for w in sequence]
            food_matches = [w for w in words_in_seq if w in FOOD_TERMS]
            if food_matches:
                score += 5 * len(food_matches)
                reasons.append(f"contains food terms: {food_matches}")

            # Pattern 3: Ends with period
            if text.endswith('.'):
                score += 3
                reasons.append("ends with period")

            # Pattern 4: In top 10% of page
            if y_percent < 10:
                score += 5
                reasons.append(f"in top 10% (y={y_percent:.1f}%)")
            elif y_percent < 20:
                score += 2
                reasons.append(f"in top 20% (y={y_percent:.1f}%)")

            # Pattern 5: Proper length (2-4 words preferred)
            if 2 <= length <= 4:
                score += 2
                reasons.append(f"good length ({length} words)")

            # Pattern 6: First letter capitalized
            if sequence[0][0].isupper():
                score += 1
                reasons.append("capitalized")

            # Pattern 7: Comes before ingredients
            # Check if there are INGREDIENT_LINE labels after this sequence
            has_ingredients_after = any(
                labels[i] == 3  # INGREDIENT_LINE = 3
                for i in range(end, min(end + 50, n))
            )
            if has_ingredients_after:
                score += 3
                reasons.append("has ingredients after")

            # Pattern 8: Not a page number
            if sequence[0].isdigit() and len(sequence[0]) <= 3:
                score -= 5
                reasons.append("starts with page number")

            # Require minimum score
            if score >= 10:  # Threshold for considering as title
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
        # Filter out section headers (score < 0)
        candidates = [c for c in candidates if c['score'] > 0]

        if not candidates:
            return labels

        # Strategy: prefer candidates that:
        # 1. Are NOT in the very top (likely section headers are at y < 5%)
        # 2. Have high scores
        # 3. Come before ingredients

        # Separate very-top candidates from others
        very_top = [c for c in candidates if c['y_percent'] < 5]
        others = [c for c in candidates if c['y_percent'] >= 5]

        # If we have candidates NOT at the very top, prefer those
        if others:
            # Sort by score (desc) then by position (asc)
            others.sort(key=lambda x: (-x['score'], x['start']))
            best = others[0]
        else:
            # All candidates at very top, pick best scoring
            very_top.sort(key=lambda x: (-x['score'], x['start']))
            best = very_top[0]

        # Apply RECIPE_TITLE label
        for i in range(best['start'], best['end']):
            labels[i] = 2  # RECIPE_TITLE = 2

        print(f"  ✓ Found title: '{best['text']}' (score={best['score']}, y={best['y_percent']:.1f}%)")
        print(f"    Reasons: {', '.join(best['reasons'])}")

    return labels


def relabel_jsonl(input_path, output_path):
    """Re-label JSONL file with improved heuristics."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")
    print()

    titles_found = 0
    total_pages = 0

    with open(input_path) as infile, open(output_path, 'w') as outfile:
        for line in infile:
            page = json.loads(line)
            total_pages += 1

            page_num = page['page_num']
            words = page['words']
            bboxes = page['bboxes']
            width = page['width']
            height = page['height']

            # Get labels, or create default if missing
            if 'labels' not in page:
                labels = [9] * len(words)  # Default to 'O' label (9)
            else:
                labels = page['labels']

            print(f"Page {page_num}:")

            # Apply improved heuristics
            new_labels = detect_recipe_title(words, bboxes, labels.copy(), width, height)

            # Check if we found a title
            if 2 in new_labels and 2 not in labels:
                titles_found += 1

            # Update page
            page['labels'] = new_labels

            # Write to output
            outfile.write(json.dumps(page) + '\n')

    print()
    print("=" * 80)
    print(f"SUMMARY:")
    print(f"  Total pages: {total_pages}")
    print(f"  New titles found: {titles_found}")
    print(f"  Success rate: {titles_found/total_pages*100:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python improve_v3_heuristics.py <input_jsonl> [output_jsonl]")
        print()
        print("Example:")
        print("  python improve_v3_heuristics.py \\")
        print("    data/processed/v3_headers_titles/boston_v3_suggested.jsonl \\")
        print("    data/processed/v3_headers_titles/boston_v3_improved.jsonl")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.jsonl', '_improved.jsonl')

    relabel_jsonl(input_file, output_file)
