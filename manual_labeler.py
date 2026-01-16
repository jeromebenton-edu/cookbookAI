#!/usr/bin/env python3
"""
Fast text-based manual labeling tool.

Shows you word lists for each page, you type the recipe title words.
Much faster than drawing bounding boxes!
"""

import json
import sys
from pathlib import Path

def load_pages(pages_to_label):
    """Load pages from v3_suggested dataset."""
    pages_data = {}

    with open("data/processed/v3_headers_titles/boston_v3_suggested.jsonl") as f:
        for line in f:
            page = json.loads(line)
            page_num = page['page_num']

            if page_num in pages_to_label:
                pages_data[page_num] = page

    return pages_data


def display_page(page, page_num):
    """Display page words for labeling."""
    words = page['words']
    labels = page.get('labels', ['O'] * len(words))

    print("\n" + "=" * 80)
    print(f"PAGE {page_num}")
    print("=" * 80)

    # Show first 200 words with indices
    print("\nWords on this page (first 200):")
    print("-" * 80)
    for i in range(min(200, len(words))):
        word = words[i]
        # Highlight if already labeled
        marker = ""
        if labels[i] in [0, 'PAGE_HEADER']:
            marker = " [PAGE_HEADER]"
        elif labels[i] in [1, 'SECTION_HEADER']:
            marker = " [SECTION_HEADER]"
        elif labels[i] in [2, 'RECIPE_TITLE']:
            marker = " [RECIPE_TITLE]"

        print(f"{i:3d}: {word}{marker}")

    if len(words) > 200:
        print(f"... ({len(words) - 200} more words)")

    print("-" * 80)


def label_page(page, page_num):
    """Interactively label a page."""
    words = page['words']
    labels = page.get('labels', ['O'] * len(words))

    display_page(page, page_num)

    print("\nInstructions:")
    print("  - Type word indices or text (e.g., '5 6 7 8' or 'How to make Tea')")
    print("  - Press Enter with no input to skip a label type")
    print("  - Type 'skip' to skip entire page")
    print("  - Type 'quit' to save and exit")
    print()

    # Step 1: Page header (optional)
    while True:
        user_input = input(f"Page {page_num} PAGE_HEADER (or Enter to skip): ").strip()

        if user_input.lower() == 'quit':
            return None

        if user_input.lower() == 'skip':
            print(f"⏭️  Skipping entire page {page_num}")
            return labels

        if not user_input:
            break  # Skip page header

        indices = parse_input(user_input, words)
        if indices:
            for i in indices:
                labels[i] = 'PAGE_HEADER'
            print(f"✓ PAGE_HEADER: '{' '.join([words[i] for i in indices])}'")
            break
        else:
            print(f"⚠️  Could not find '{user_input}'. Try again or press Enter to skip")

    # Step 2: Section header (optional)
    while True:
        user_input = input(f"Page {page_num} SECTION_HEADER (or Enter to skip): ").strip()

        if user_input.lower() == 'quit':
            return None

        if user_input.lower() == 'skip':
            print(f"⏭️  Skipping entire page {page_num}")
            return labels

        if not user_input:
            break  # Skip section header

        indices = parse_input(user_input, words)
        if indices:
            for i in indices:
                labels[i] = 'SECTION_HEADER'
            print(f"✓ SECTION_HEADER: '{' '.join([words[i] for i in indices])}'")
            break
        else:
            print(f"⚠️  Could not find '{user_input}'. Try again or press Enter to skip")

    # Step 3: Recipe title (required - this is the main one)
    while True:
        user_input = input(f"Page {page_num} RECIPE_TITLE (or 'skip'/'quit'): ").strip()

        if user_input.lower() == 'quit':
            return None

        if user_input.lower() == 'skip':
            print(f"⏭️  Skipping page {page_num}")
            return labels

        if not user_input:
            print("⚠️  RECIPE_TITLE is required. Type 'skip' to skip this page")
            continue

        indices = parse_input(user_input, words)
        if indices:
            for i in indices:
                labels[i] = 'RECIPE_TITLE'
            print(f"✓ RECIPE_TITLE: '{' '.join([words[i] for i in indices])}'")
            break
        else:
            print(f"⚠️  Could not find '{user_input}'. Try word indices (e.g., '5 6 7 8')")
            # Show suggestions (search all words, show first 20 matches)
            search_words = user_input.split()
            print("Suggestions:")
            matches = []
            for i, word in enumerate(words):
                if any(search_word.lower() in word.lower() for search_word in search_words):
                    matches.append((i, word))
                    if len(matches) >= 20:
                        break
            for i, word in matches:
                print(f"  {i}: {word}")

    return labels


def parse_input(user_input, words):
    """Parse user input as either indices or text."""
    # Check if user typed indices
    if all(part.isdigit() for part in user_input.split()):
        indices = [int(idx) for idx in user_input.split()]
        if all(0 <= idx < len(words) for idx in indices):
            return indices
        return None

    # Otherwise, try to find the text
    search_words = user_input.split()
    return find_word_sequence(words, search_words)


def find_word_sequence(words, search_words):
    """Find a sequence of words in the word list."""
    # Try exact match first
    for i in range(len(words) - len(search_words) + 1):
        match = True
        for j, search_word in enumerate(search_words):
            # Normalize for comparison
            page_word = words[i + j].lower().rstrip('.,;:')
            search_word_norm = search_word.lower().rstrip('.,;:')

            if page_word != search_word_norm:
                match = False
                break

        if match:
            return list(range(i, i + len(search_words)))

    return None


def save_labeled_data(pages_data, output_path):
    """Save labeled pages back to JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all pages from source
    all_pages = []
    with open("data/processed/v3_headers_titles/boston_v3_suggested.jsonl") as f:
        for line in f:
            page = json.loads(line)

            # If this page was labeled, use new labels
            if page['page_num'] in pages_data:
                page['labels'] = pages_data[page['page_num']]['labels']

            all_pages.append(page)

    # Write all pages
    with open(output_path, 'w') as f:
        for page in all_pages:
            f.write(json.dumps(page) + '\n')

    print(f"\n✓ Saved {len(all_pages)} pages to {output_path}")
    print(f"  ({len(pages_data)} pages manually labeled)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python manual_labeler.py <batch_file>")
        print()
        print("Example:")
        print("  python manual_labeler.py data/label_studio/batch1_pages.txt")
        print()
        print("Or specify pages directly:")
        print("  python manual_labeler.py 69,76,78,88,90,92,94")
        sys.exit(1)

    # Parse pages to label
    arg = sys.argv[1]

    if Path(arg).exists():
        # Load from file
        with open(arg) as f:
            pages_str = f.read().strip()
            pages_to_label = [int(p.strip()) for p in pages_str.split(',')]
    else:
        # Parse comma-separated list
        pages_to_label = [int(p.strip()) for p in arg.split(',')]

    print("=" * 80)
    print("MANUAL LABELING TOOL")
    print("=" * 80)
    print(f"\nPages to label: {pages_to_label}")
    print(f"Total: {len(pages_to_label)} pages")
    print()

    # Load pages
    print("Loading pages...")
    pages_data = load_pages(pages_to_label)

    if len(pages_data) < len(pages_to_label):
        missing = set(pages_to_label) - set(pages_data.keys())
        print(f"⚠️  Warning: Could not find pages: {missing}")

    print(f"✓ Loaded {len(pages_data)} pages")
    print()
    input("Press Enter to start labeling...")

    # Label each page
    labeled_count = 0

    for page_num in sorted(pages_to_label):
        if page_num not in pages_data:
            continue

        page = pages_data[page_num]

        labels = label_page(page, page_num)

        if labels is None:
            # User quit
            print("\n⏹️  Quitting early")
            break

        # Update page with new labels
        pages_data[page_num]['labels'] = labels
        labeled_count += 1

        print(f"\n✓ Page {page_num} labeled ({labeled_count}/{len(pages_to_label)})")

    # Save results
    if labeled_count > 0:
        output_path = "data/processed/v3_headers_titles/boston_v3_manual_labels.jsonl"

        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)

        save_labeled_data(pages_data, output_path)

        print("\n✓ Labeling complete!")
        print(f"  Labeled: {labeled_count} pages")
        print(f"  Output: {output_path}")
        print()
        print("Next steps:")
        print("  1. Rebuild dataset with manual labels:")
        print(f"     python tools/build_v3_dataset.py \\")
        print(f"       --input {output_path} \\")
        print(f"       --output data/datasets/boston_layoutlmv3_v3_manual")
        print()
        print("  2. Train model:")
        print(f"     python tools/train_v3_model.py \\")
        print(f"       --dataset data/datasets/boston_layoutlmv3_v3_manual/dataset_dict \\")
        print(f"       --output models/layoutlmv3_v3_manual \\")
        print(f"       --epochs 20")
    else:
        print("\n⚠️  No pages were labeled")


if __name__ == "__main__":
    main()
