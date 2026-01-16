#!/usr/bin/env python3
"""Analyze manually labeled recipe titles to identify patterns for heuristics."""

import json
from pathlib import Path

# Load manual annotations
with open("data/label_studio/demo-manual-label-v3.json") as f:
    annotations = json.load(f)

print("=" * 80)
print("RECIPE TITLE PATTERN ANALYSIS")
print("=" * 80)
print()

print("Manual Labels Summary:")
print("-" * 80)
for page in annotations["pages"]:
    title = page["recipe_title"]
    words = page["selected_words"]
    print(f"Page {page['page_num']:3d}: {title:30s} ({len(words)} words)")

print()
print("=" * 80)
print("PATTERNS IDENTIFIED")
print("=" * 80)
print()

# Pattern 1: Title length
print("1. TITLE LENGTH PATTERNS:")
word_counts = [len(p["selected_words"]) for p in annotations["pages"]]
print(f"   - Min words: {min(word_counts)}")
print(f"   - Max words: {max(word_counts)}")
print(f"   - Average: {sum(word_counts)/len(word_counts):.1f} words")
print(f"   - Most common: {max(set(word_counts), key=word_counts.count)} words")
print()

# Pattern 2: Punctuation
print("2. PUNCTUATION PATTERNS:")
titles_with_period = sum(1 for p in annotations["pages"] if p["recipe_title"].endswith("."))
print(f"   - Titles ending with '.': {titles_with_period}/{len(annotations['pages'])} ({titles_with_period/len(annotations['pages'])*100:.0f}%)")
print()

# Pattern 3: Common words/patterns
print("3. COMMON TITLE WORDS:")
all_words = []
for p in annotations["pages"]:
    all_words.extend(p["selected_words"])
word_freq = {}
for word in all_words:
    word_lower = word.lower().rstrip(".")
    word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
common_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
for word, count in common_words:
    print(f"   - '{word}': {count} times")
print()

# Pattern 4: Title structure
print("4. TITLE STRUCTURE PATTERNS:")
for p in annotations["pages"]:
    title = p["recipe_title"]
    words = p["selected_words"]

    # Check for common patterns
    has_recipe_type = any(word.lower() in ["tea", "cocoa", "punch", "bread", "rolls"]
                          for word in words)
    has_descriptor = len(words) > 1
    has_how_to = title.lower().startswith("how to")

    pattern_desc = []
    if has_how_to:
        pattern_desc.append("Instruction format 'How to...'")
    elif has_descriptor and has_recipe_type:
        pattern_desc.append("Descriptor + Recipe Type")
    elif has_recipe_type:
        pattern_desc.append("Recipe Type only")

    print(f"   Page {p['page_num']}: {title:30s} -> {', '.join(pattern_desc)}")
print()

# Pattern 5: Capitalization
print("5. CAPITALIZATION PATTERNS:")
proper_case_count = sum(1 for p in annotations["pages"]
                        if p["selected_words"][0][0].isupper())
print(f"   - Titles starting with capital: {proper_case_count}/{len(annotations['pages'])}")
all_caps = sum(1 for p in annotations["pages"]
               if all(word.isupper() or not word.isalpha() for word in p["selected_words"]))
print(f"   - All caps titles: {all_caps}/{len(annotations['pages'])}")
print()

print("=" * 80)
print("HEURISTIC RECOMMENDATIONS")
print("=" * 80)
print()
print("Based on the manual labels, recipe titles can be identified by:")
print()
print("1. **Position**: Likely appear early on the page (need to verify with bbox data)")
print()
print("2. **Length**: Typically 2-4 words")
print()
print("3. **Punctuation**: ~85% end with a period")
print()
print("4. **Structure patterns**:")
print("   - 'How to [verb] [item]' format (e.g., 'How to make Tea.')")
print("   - '[Descriptor] [Recipe Type]' format (e.g., 'Graham Bread.', 'Claret Punch.')")
print("   - Common recipe types: Bread, Tea, Cocoa, Punch, Rolls")
print()
print("5. **Context clues**:")
print("   - Appears before ingredient lists (marked as INGREDIENT_LINE)")
print("   - Appears after page headers (like 'BOSTON COOKING-SCHOOL COOK BOOK')")
print("   - Not part of running instructions")
print()
print("6. **Font/Position** (requires bbox analysis):")
print("   - Likely larger font size (taller bbox)")
print("   - Positioned near top of page content area")
print("   - Centered or left-aligned")
print()
print("=" * 80)
print("NEXT STEPS")
print("=" * 80)
print()
print("1. Load the full dataset and examine bbox coordinates for these pages")
print("2. Measure actual Y-position and bbox height of these titles")
print("3. Compare with surrounding text (ingredients, instructions)")
print("4. Update heuristic script with these patterns:")
print("   - Y-position threshold (e.g., top 30% of page)")
print("   - Bbox height threshold (e.g., 20% larger than body text)")
print("   - Length filter (2-6 words)")
print("   - Pattern matching ('How to', common food terms)")
print("   - Position relative to ingredients (must come before)")
print()
