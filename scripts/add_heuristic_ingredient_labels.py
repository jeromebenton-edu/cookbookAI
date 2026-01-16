#!/usr/bin/env python3
"""Add heuristic INGREDIENT_LINE labels to pages without dense supervision.

This script enhances existing OCR/weak-labeled pages by adding heuristic
INGREDIENT_LINE labels using common cookbook ingredient patterns:
1. Lines starting with quantity patterns (numbers, fractions, decimals)
2. Lines containing quantity + unit patterns
3. Lines appearing inside detected ingredient blocks

This allows us to expand ingredient supervision in the recipe-only dataset.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import label schema
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.datasets.labels import label2id, LABELS

# Common measurement units
UNITS = {
    'cup', 'cups', 'c',
    'tablespoon', 'tablespoons', 'tbsp', 'tbs', 'T',
    'teaspoon', 'teaspoons', 'tsp', 'ts', 't',
    'ounce', 'ounces', 'oz',
    'pound', 'pounds', 'lb', 'lbs',
    'pint', 'pints', 'pt',
    'quart', 'quarts', 'qt',
    'gallon', 'gallons', 'gal',
    'gram', 'grams', 'g',
    'kilogram', 'kilograms', 'kg',
    'milliliter', 'milliliters', 'ml',
    'liter', 'liters', 'l',
    'pinch', 'dash', 'handful',
    'can', 'cans', 'package', 'packages', 'pkg',
    'slice', 'slices', 'piece', 'pieces',
}

# Common instruction verbs (to detect when ingredient block ends)
INSTRUCTION_VERBS = {
    'add', 'bake', 'beat', 'blend', 'boil', 'bring', 'brush',
    'chill', 'chop', 'combine', 'cook', 'cool', 'cover', 'cream', 'cut',
    'dice', 'dip', 'divide', 'drain', 'drizzle',
    'fold', 'form', 'fry',
    'garnish', 'grate', 'grease',
    'heat',
    'knead',
    'let',
    'make', 'melt', 'mince', 'mix',
    'place', 'pour', 'prepare', 'press', 'put',
    'reduce', 'remove', 'rinse', 'roll',
    'sauté', 'season', 'separate', 'serve', 'set', 'simmer',
    'slice', 'spread', 'sprinkle', 'stir', 'strain',
    'take', 'toast', 'toss', 'transfer', 'turn',
    'use',
    'wash', 'whip', 'whisk',
}

# Quantity patterns
QUANTITY_START_PATTERN = re.compile(
    r'^\s*(\d+(/\d+)?|\d+\.\d+)',  # Starts with number, fraction, or decimal
    re.IGNORECASE
)

QUANTITY_UNIT_PATTERN = re.compile(
    r'(\d+(/\d+)?|\d+\.\d+)\s*(' + '|'.join(UNITS) + r')\b',
    re.IGNORECASE
)


def is_quantity_line(text: str) -> bool:
    """Check if a line starts with a quantity pattern."""
    if not text or len(text) < 2:
        return False
    return bool(QUANTITY_START_PATTERN.match(text))


def has_quantity_unit(text: str) -> bool:
    """Check if a line contains quantity + unit pattern."""
    if not text or len(text) < 3:
        return False
    return bool(QUANTITY_UNIT_PATTERN.search(text))


def is_likely_instruction(text: str) -> bool:
    """Check if a text line is likely an instruction (signals end of ingredient block)."""
    if not text or len(text) < 3:
        return False

    # Normalize: lowercase, remove leading punctuation/numbers
    normalized = text.lower().strip().lstrip('.-0123456789) ')

    # Check if starts with common instruction verb
    for verb in INSTRUCTION_VERBS:
        if normalized.startswith(verb + ' ') or normalized == verb:
            return True

    return False


def is_likely_title(text: str, bbox: List[float], page_height: float) -> bool:
    """Check if a line is likely a title (appears in top portion of page)."""
    if not text or not bbox or page_height <= 0:
        return False

    # Check if line is in top 15% of page
    y_position = bbox[1]  # y-coordinate
    relative_position = y_position / page_height

    # Title heuristic: short text in top portion
    is_short = len(text.strip()) < 50
    is_top = relative_position < 0.15

    return is_short and is_top


def reconstruct_lines(words: List[str], bboxes: List[List[float]]) -> List[tuple[int, int, str, List[float]]]:
    """Reconstruct text lines from words and bboxes.

    Returns:
        List of (start_idx, end_idx, line_text, bbox) tuples
    """
    if not words or not bboxes or len(words) != len(bboxes):
        return []

    lines = []
    current_line_start = 0
    current_line_words = [words[0]]
    current_y = bboxes[0][1]  # y-coordinate of first bbox
    current_bbox = list(bboxes[0])

    # Group words into lines based on y-coordinate proximity
    y_threshold = 20  # pixels - tune based on DPI

    for i in range(1, len(words)):
        word = words[i]
        bbox = bboxes[i]

        # Check if this word is on a new line
        if abs(bbox[1] - current_y) > y_threshold:
            # Save current line
            line_text = ' '.join(current_line_words)
            lines.append((current_line_start, i, line_text, current_bbox))

            # Start new line
            current_line_start = i
            current_line_words = [word]
            current_y = bbox[1]
            current_bbox = list(bbox)
        else:
            current_line_words.append(word)
            # Expand bbox to include this word
            current_bbox[2] = max(current_bbox[2], bbox[2])  # x2
            current_bbox[3] = max(current_bbox[3], bbox[3])  # y2

    # Add final line
    if current_line_words:
        line_text = ' '.join(current_line_words)
        lines.append((current_line_start, len(words), line_text, current_bbox))

    return lines


def detect_ingredient_block(
    lines: List[tuple[int, int, str, List[float]]],
    page_height: float,
) -> Optional[tuple[int, int]]:
    """Detect the start and end of an ingredient block.

    Returns:
        (start_line_idx, end_line_idx) or None if no block detected
    """
    if not lines:
        return None

    # Find first ingredient-like line (skip titles in top 15%)
    start_idx = None
    for i, (_, _, line_text, bbox) in enumerate(lines):
        # Skip title area
        if is_likely_title(line_text, bbox, page_height):
            continue

        # Check for ingredient patterns
        if is_quantity_line(line_text) or has_quantity_unit(line_text):
            start_idx = i
            break

    if start_idx is None:
        return None

    # Find end of ingredient block
    # Ends when we hit instruction verbs or large gap
    end_idx = len(lines)
    blank_line_count = 0

    for i in range(start_idx + 1, len(lines)):
        _, _, line_text, _ = lines[i]

        # Check for instruction verb (end of ingredients)
        if is_likely_instruction(line_text):
            end_idx = i
            break

        # Check for blank gap (2+ consecutive non-ingredient lines)
        if not line_text.strip():
            blank_line_count += 1
            if blank_line_count >= 2:
                end_idx = i - blank_line_count
                break
        else:
            blank_line_count = 0

    return (start_idx, end_idx)


def add_heuristic_ingredient_labels(
    page: Dict[str, Any],
    min_line_length: int = 3,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Add heuristic INGREDIENT_LINE labels to a page.

    Args:
        page: Page data with words, bboxes, labels
        min_line_length: Minimum line length (in chars) to consider for labeling
        dry_run: If True, only compute stats without modifying labels

    Returns:
        Modified page with added ingredient labels
    """
    words = page.get("words", [])
    bboxes = page.get("bboxes", [])
    labels = page.get("labels", ["O"] * len(words))  # Default to STRING labels
    page_height = page.get("height", 1000)  # Default height

    if not words or len(words) != len(bboxes) or len(words) != len(labels):
        return page

    # Convert to mutable list
    labels = list(labels)

    # Detect label format (string or int) - always use strings for consistency
    label_format = "string" if labels and isinstance(labels[0], str) else "int"

    # Convert integer labels to strings for consistency
    if label_format == "int":
        labels = [LABELS[lbl] if isinstance(lbl, int) and lbl < len(LABELS) else "O" for lbl in labels]
        label_format = "string"

    ingredient_label = "INGREDIENT_LINE"
    o_label = "O"

    # Reconstruct lines from words
    lines = reconstruct_lines(words, bboxes)

    # Detect ingredient block
    ingredient_block = detect_ingredient_block(lines, page_height)

    # Track stats
    labeled_lines = 0
    labeled_tokens = 0

    if ingredient_block is not None:
        start_line_idx, end_line_idx = ingredient_block

        # Label all lines in the ingredient block
        for i in range(start_line_idx, end_line_idx):
            start_idx, end_idx, line_text, bbox = lines[i]

            # Skip short lines
            if len(line_text) < min_line_length:
                continue

            # Skip lines that are clearly instructions
            if is_likely_instruction(line_text):
                continue

            # Check if line looks like an ingredient
            should_label = False

            # Heuristic 1: Starts with quantity
            if is_quantity_line(line_text):
                should_label = True

            # Heuristic 2: Contains quantity + unit
            if has_quantity_unit(line_text):
                should_label = True

            # Heuristic 3: Inside ingredient block (between first ingredient and first instruction)
            # Already inside block, so tentatively label if it's not obviously wrong
            if not is_likely_instruction(line_text) and not is_likely_title(line_text, bbox, page_height):
                should_label = True

            # Apply label if appropriate
            if should_label:
                line_labeled_tokens = 0
                for idx in range(start_idx, end_idx):
                    # Only replace O tokens (preserve existing curated labels)
                    if labels[idx] == o_label:
                        if not dry_run:
                            labels[idx] = ingredient_label
                        line_labeled_tokens += 1
                        labeled_tokens += 1

                if line_labeled_tokens > 0:
                    labeled_lines += 1

    # Update page with new labels (unless dry_run)
    if not dry_run:
        page["labels"] = labels

    # Add metadata about heuristic labeling
    if "heuristic_labeling" not in page:
        page["heuristic_labeling"] = {}

    page["heuristic_labeling"]["ingredient_labels_added"] = labeled_tokens
    page["heuristic_labeling"]["ingredient_lines_labeled"] = labeled_lines

    return page


def process_jsonl(
    in_jsonl: Path,
    out_jsonl: Path,
    min_line_length: int = 3,
    dry_run: bool = False,
    max_pages: Optional[int] = None,
) -> Dict[str, Any]:
    """Process JSONL file and add heuristic ingredient labels.

    Args:
        in_jsonl: Input JSONL file
        out_jsonl: Output JSONL file
        min_line_length: Minimum line length for labeling
        dry_run: If True, only compute stats without writing output
        max_pages: Limit number of pages to process (for testing)

    Returns:
        Stats dict with counts and top pages
    """
    stats = Counter(
        total_pages=0,
        pages_labeled=0,
        total_ingredient_tokens_added=0,
        total_ingredient_lines_labeled=0,
    )

    pages_by_tokens = []  # (page_num, tokens_added)

    if not dry_run:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    fout = None if dry_run else out_jsonl.open("w")

    try:
        with in_jsonl.open() as fin:
            for line in fin:
                if not line.strip():
                    continue

                page = json.loads(line)
                stats["total_pages"] += 1

                # Add heuristic labels
                page = add_heuristic_ingredient_labels(
                    page,
                    min_line_length=min_line_length,
                    dry_run=dry_run,
                )

                # Track stats
                tokens_added = page.get("heuristic_labeling", {}).get("ingredient_labels_added", 0)
                lines_labeled = page.get("heuristic_labeling", {}).get("ingredient_lines_labeled", 0)

                if tokens_added > 0:
                    stats["pages_labeled"] += 1
                    stats["total_ingredient_tokens_added"] += tokens_added
                    stats["total_ingredient_lines_labeled"] += lines_labeled
                    page_num = page.get("page_num", -1)
                    pages_by_tokens.append((page_num, tokens_added, lines_labeled))

                if not dry_run:
                    fout.write(json.dumps(page) + "\n")

                # Stop if max_pages reached
                if max_pages and stats["total_pages"] >= max_pages:
                    break
    finally:
        if fout:
            fout.close()

    # Sort pages by tokens added (descending)
    pages_by_tokens.sort(key=lambda x: x[1], reverse=True)

    return {
        "stats": dict(stats),
        "top_pages": pages_by_tokens[:10],
        "distribution": _compute_distribution(pages_by_tokens),
    }


def _compute_distribution(pages_by_tokens: List[tuple]) -> Dict[str, int]:
    """Compute distribution of ingredient tokens added per page."""
    if not pages_by_tokens:
        return {}

    tokens_per_page = [tokens for _, tokens, _ in pages_by_tokens]

    return {
        "min": min(tokens_per_page),
        "max": max(tokens_per_page),
        "mean": sum(tokens_per_page) / len(tokens_per_page),
        "pages_modified": len(tokens_per_page),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Add heuristic INGREDIENT_LINE labels to pages"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file with OCR/labels"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file with added ingredient labels"
    )
    parser.add_argument(
        "--min-line-length",
        type=int,
        default=3,
        help="Minimum line length (chars) to consider for labeling"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only compute stats without writing output"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limit number of pages to process (for testing)"
    )

    args = parser.parse_args()

    print(f"Processing {args.input}")
    if args.dry_run:
        print("DRY RUN MODE: No output will be written")
    if args.max_pages:
        print(f"Processing only first {args.max_pages} pages")

    result = process_jsonl(
        args.input,
        args.output,
        min_line_length=args.min_line_length,
        dry_run=args.dry_run,
        max_pages=args.max_pages,
    )

    stats = result["stats"]
    top_pages = result["top_pages"]
    distribution = result["distribution"]

    print(f"\nHeuristic ingredient labeling {'simulation' if args.dry_run else 'complete'}!")
    print(f"  Total pages: {stats['total_pages']}")
    print(f"  Pages modified: {stats['pages_labeled']}")
    print(f"  Total ingredient tokens added: {stats['total_ingredient_tokens_added']}")
    print(f"  Total ingredient lines labeled: {stats['total_ingredient_lines_labeled']}")

    if distribution:
        print(f"\nDistribution of ingredient tokens added per page:")
        print(f"  Min:  {distribution['min']}")
        print(f"  Mean: {distribution['mean']:.1f}")
        print(f"  Max:  {distribution['max']}")

    if top_pages:
        print(f"\nTop 10 pages by ingredient tokens added:")
        for page_num, tokens, lines in top_pages:
            print(f"  Page {page_num:4d}: {tokens:4d} tokens, {lines:3d} lines")

    if not args.dry_run:
        print(f"\n  Output: {args.output}")


if __name__ == "__main__":
    main()
