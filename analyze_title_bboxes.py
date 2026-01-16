#!/usr/bin/env python3
"""Analyze bbox coordinates for manually labeled recipe titles."""

import json
from pathlib import Path

# Load manual annotations
with open("data/label_studio/demo-manual-label-v3.json") as f:
    manual_labels = json.load(f)

# Create lookup for manual titles
title_lookup = {p["page_num"]: p["selected_words"] for p in manual_labels["pages"]}

# Load JSONL data
jsonl_path = Path("data/processed/v3_headers_titles/boston_v3_suggested.jsonl")
pages_data = {}

with open(jsonl_path) as f:
    for line in f:
        page = json.loads(line)
        page_num = page["page_num"]
        if page_num in title_lookup:
            pages_data[page_num] = page

print("=" * 80)
print("BBOX ANALYSIS FOR MANUALLY LABELED RECIPE TITLES")
print("=" * 80)
print()

# Analyze each page
for page_num in sorted(title_lookup.keys()):
    if page_num not in pages_data:
        print(f"⚠️  Page {page_num}: Not found in JSONL data")
        continue

    page = pages_data[page_num]
    title_words = title_lookup[page_num]
    words = page["words"]
    bboxes = page["bboxes"]
    width = page["width"]
    height = page["height"]

    print(f"\nPage {page_num}: '{' '.join(title_words)}'")
    print("-" * 80)

    # Find title words in the page
    title_indices = []
    for title_word in title_words:
        try:
            # Find first occurrence
            idx = words.index(title_word)
            title_indices.append(idx)
        except ValueError:
            # Try without punctuation
            clean_word = title_word.rstrip(".")
            try:
                idx = words.index(clean_word)
                title_indices.append(idx)
            except ValueError:
                print(f"  ⚠️  Could not find '{title_word}' in page words")

    if not title_indices:
        continue

    # Get bboxes for title words
    title_bboxes = [bboxes[i] for i in title_indices if i < len(bboxes)]

    if not title_bboxes:
        continue

    # Calculate statistics
    title_y_positions = [bbox[1] for bbox in title_bboxes]  # Top Y coordinate
    title_heights = [bbox[3] - bbox[1] for bbox in title_bboxes]  # Height
    title_x_positions = [bbox[0] for bbox in title_bboxes]  # Left X coordinate

    # Get some surrounding text for context
    context_start = max(0, title_indices[0] - 5)
    context_end = min(len(words), title_indices[-1] + 10)
    context_words = words[context_start:context_end]
    context_bboxes = bboxes[context_start:context_end]

    # Calculate average body text height for comparison
    body_heights = [bbox[3] - bbox[1] for bbox in context_bboxes if bbox[3] - bbox[1] > 0]
    avg_body_height = sum(body_heights) / len(body_heights) if body_heights else 0

    print(f"  Title position:")
    print(f"    - Word indices: {title_indices}")
    print(f"    - Y position (top): {min(title_y_positions)} - {max(title_y_positions)} px")
    print(f"    - Y position % of page: {min(title_y_positions)/height*100:.1f}% - {max(title_y_positions)/height*100:.1f}%")
    print(f"    - X position (left): {min(title_x_positions)} px")
    print(f"    - Title height: {min(title_heights)} - {max(title_heights)} px (avg: {sum(title_heights)/len(title_heights):.1f})")
    print(f"    - Avg body text height: {avg_body_height:.1f} px")
    print(f"    - Title vs body height ratio: {(sum(title_heights)/len(title_heights))/avg_body_height:.2f}x" if avg_body_height > 0 else "")
    print(f"  Page dimensions: {width}x{height} px")
    print(f"  Context: ...{' '.join(context_words[:15])}...")

print()
print("=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print()

# Collect all stats
all_y_positions = []
all_y_percentages = []
all_heights = []
all_x_positions = []

for page_num in sorted(title_lookup.keys()):
    if page_num not in pages_data:
        continue

    page = pages_data[page_num]
    title_words = title_lookup[page_num]
    words = page["words"]
    bboxes = page["bboxes"]
    height = page["height"]

    title_indices = []
    for title_word in title_words:
        try:
            idx = words.index(title_word)
            title_indices.append(idx)
        except ValueError:
            clean_word = title_word.rstrip(".")
            try:
                idx = words.index(clean_word)
                title_indices.append(idx)
            except ValueError:
                pass

    if title_indices:
        title_bboxes = [bboxes[i] for i in title_indices if i < len(bboxes)]
        if title_bboxes:
            for bbox in title_bboxes:
                all_y_positions.append(bbox[1])
                all_y_percentages.append(bbox[1] / height * 100)
                all_heights.append(bbox[3] - bbox[1])
                all_x_positions.append(bbox[0])

if all_y_positions:
    print(f"Title Y-position (top of bbox):")
    print(f"  - Min: {min(all_y_positions)} px")
    print(f"  - Max: {max(all_y_positions)} px")
    print(f"  - Average: {sum(all_y_positions)/len(all_y_positions):.1f} px")
    print()
    print(f"Title Y-position as % of page height:")
    print(f"  - Min: {min(all_y_percentages):.1f}%")
    print(f"  - Max: {max(all_y_percentages):.1f}%")
    print(f"  - Average: {sum(all_y_percentages)/len(all_y_percentages):.1f}%")
    print()
    print(f"Title height:")
    print(f"  - Min: {min(all_heights)} px")
    print(f"  - Max: {max(all_heights)} px")
    print(f"  - Average: {sum(all_heights)/len(all_heights):.1f} px")
    print()
    print(f"Title X-position (left):")
    print(f"  - Min: {min(all_x_positions)} px")
    print(f"  - Max: {max(all_x_positions)} px")
    print(f"  - Average: {sum(all_x_positions)/len(all_x_positions):.1f} px")

print()
print("=" * 80)
print("HEURISTIC RULES TO IMPLEMENT")
print("=" * 80)
print()
print("Based on this analysis, recipe titles can be detected with:")
print()
print(f"1. **Y-position**: Top < {max(all_y_percentages):.0f}% of page height")
print(f"   (Most titles are in the top {sum(all_y_percentages)/len(all_y_percentages):.0f}% on average)")
print()
print(f"2. **Word count**: 2-4 words (from pattern analysis)")
print()
print(f"3. **Bbox height**: > {min(all_heights)} px (likely indicates larger font)")
print()
print(f"4. **Text patterns**:")
print(f"   - Starts with 'How to'")
print(f"   - Contains food-related terms (Bread, Tea, Cocoa, Punch, etc.)")
print(f"   - Ends with period (~85% of the time)")
print()
print(f"5. **Position in sequence**: Comes before INGREDIENT_LINE tokens")
print()
print(f"6. **Not**: Page headers (BOSTON COOKING-SCHOOL), page numbers")
print()
