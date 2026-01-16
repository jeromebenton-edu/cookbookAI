#!/usr/bin/env python3
"""
Select pages for manual labeling.

Strategy:
1. Pages with section headers (from v2 heuristics)
2. Pages with potential recipe titles nearby
3. Diverse coverage across the cookbook
4. Pages that v2 heuristics struggled with
"""

import json
from pathlib import Path
from collections import defaultdict

# Load v2 improved dataset
input_path = Path("data/processed/v3_headers_titles/boston_v3_improved_v2.jsonl")

pages_with_sections = []
pages_with_titles = []
pages_with_both = []
all_pages = []

with open(input_path) as f:
    for line in f:
        page = json.loads(line)
        page_num = page['page_num']
        labels = page['labels']
        words = page['words']

        has_section = 1 in labels  # SECTION_HEADER
        has_title = 2 in labels     # RECIPE_TITLE

        page_info = {
            'page_num': page_num,
            'has_section': has_section,
            'has_title': has_title,
            'num_words': len(words),
            'num_ingredients': labels.count(3),
            'num_instructions': labels.count(4)
        }

        all_pages.append(page_info)

        if has_section:
            pages_with_sections.append(page_info)
        if has_title:
            pages_with_titles.append(page_info)
        if has_section and has_title:
            pages_with_both.append(page_info)

print("=" * 80)
print("PAGE SELECTION FOR MANUAL LABELING")
print("=" * 80)
print()

print(f"Total pages: {len(all_pages)}")
print(f"Pages with section headers: {len(pages_with_sections)}")
print(f"Pages with recipe titles: {len(pages_with_titles)}")
print(f"Pages with BOTH: {len(pages_with_both)}")
print()

# Recommendation 1: Pages with section headers (likely to have titles too)
print("=" * 80)
print("BATCH 1: Pages with Section Headers (20 pages)")
print("=" * 80)
print("These pages have clear section headers, making it easier to find recipe titles")
print()

# Take every 6th page with section header for diversity
batch1 = [p for i, p in enumerate(pages_with_sections) if i % 6 == 0][:20]
batch1_pages = [p['page_num'] for p in batch1]
print(f"Pages: {batch1_pages}")
print()

# Recommendation 2: Pages with ingredients/instructions but no detected title
print("=" * 80)
print("BATCH 2: Recipe Pages WITHOUT Detected Titles (20 pages)")
print("=" * 80)
print("These pages likely have recipes but v2 heuristics missed the title")
print()

recipe_pages_no_title = [
    p for p in all_pages
    if (p['num_ingredients'] > 5 or p['num_instructions'] > 10)
    and not p['has_title']
    and p['page_num'] >= 38  # Actual recipes start around page 38
]

# Take every 10th for diversity
batch2 = [p for i, p in enumerate(recipe_pages_no_title) if i % 10 == 0][:20]
batch2_pages = [p['page_num'] for p in batch2]
print(f"Pages: {batch2_pages}")
print()

# Recommendation 3: Early pages (intro/TOC) and late pages (index)
print("=" * 80)
print("BATCH 3: Diverse Pages Across Cookbook (10 pages)")
print("=" * 80)
print("Early pages (1-30), middle (300-350), late (550+)")
print()

batch3_pages = [5, 10, 15, 20, 25, 300, 320, 340, 550, 580]
print(f"Pages: {batch3_pages}")
print()

# Combined recommendation
print("=" * 80)
print("RECOMMENDED LABELING PLAN")
print("=" * 80)
print()
print("Start with Batch 1 (20 pages with section headers)")
print("  - Easier to label because structure is clear")
print("  - Good for training section header + title detection")
print()
print("Then do Batch 2 (20 pages with recipes but no title)")
print("  - These are the ones the model needs most")
print("  - Will improve recall significantly")
print()
print("Optional: Batch 3 (10 diverse pages)")
print("  - For better overall coverage")
print()

# Save to files
batch1_file = Path("data/label_studio/batch1_pages.txt")
batch2_file = Path("data/label_studio/batch2_pages.txt")
batch3_file = Path("data/label_studio/batch3_pages.txt")

batch1_file.parent.mkdir(parents=True, exist_ok=True)

with open(batch1_file, 'w') as f:
    f.write(','.join(map(str, batch1_pages)))

with open(batch2_file, 'w') as f:
    f.write(','.join(map(str, batch2_pages)))

with open(batch3_file, 'w') as f:
    f.write(','.join(map(str, batch3_pages)))

print(f"✓ Batch 1 pages saved to: {batch1_file}")
print(f"✓ Batch 2 pages saved to: {batch2_file}")
print(f"✓ Batch 3 pages saved to: {batch3_file}")
print()

# Create combined Label Studio tasks file for Batch 1
print("Creating Label Studio tasks for Batch 1...")

# We'd need to load the actual page data to create full tasks
# For now, just output the page numbers

print()
print("=" * 80)
print("NEXT STEPS")
print("=" * 80)
print()
print("1. Review the page numbers above")
print("2. Load Label Studio with Batch 1 (20 pages)")
print("3. Label RECIPE_TITLE, SECTION_HEADER, PAGE_HEADER")
print("4. Export annotations")
print("5. Train model on clean labels")
print("6. Evaluate improvement")
print("7. If good, continue with Batch 2")
print()
print("Estimated time:")
print("  - 2-3 minutes per page = 40-60 minutes for Batch 1")
print("  - 40-60 minutes for Batch 2")
print("  - Total: ~2 hours for 40 high-quality pages")
print()
