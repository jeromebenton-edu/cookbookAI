#!/usr/bin/env python3
"""Add heuristic INSTRUCTION_STEP labels to pages without dense supervision.

This script enhances existing OCR/weak-labeled pages by adding heuristic
INSTRUCTION_STEP labels using simple rules:
1. After ingredients block ends, label subsequent text lines as INSTRUCTION_STEP
2. Label lines that start with verbs as INSTRUCTION_STEP

This allows us to expand the recipe-only dataset without manual labeling.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any

# Import label schema
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.datasets.labels import label2id, LABELS

# Common instruction verbs (imperative form)
INSTRUCTION_VERBS = {
    'add', 'bake', 'beat', 'blend', 'boil', 'bring', 'brush', 'butter',
    'chill', 'chop', 'combine', 'cook', 'cool', 'cover', 'cream', 'cut',
    'dice', 'dip', 'divide', 'drain', 'drizzle', 'fold', 'form',
    'fry', 'garnish', 'grate', 'grease', 'heat', 'knead', 'let', 'make',
    'melt', 'mince', 'mix', 'pare', 'place', 'pour', 'prepare', 'press',
    'prick', 'put', 'reduce', 'remove', 'repeat', 'rinse', 'roll', 'rub',
    'sauté', 'scald', 'season', 'separate', 'serve', 'set', 'sift', 'simmer',
    'slice', 'spoon', 'spread', 'sprinkle', 'stand', 'steam', 'stir', 'strain',
    'stuff', 'take', 'toast', 'toss', 'transfer', 'turn', 'use', 'wash',
    'whip', 'whisk', 'wipe', 'work', 'wrap',
}


def is_likely_instruction(text: str) -> bool:
    """Check if a text line is likely an instruction based on verb heuristics."""
    if not text or len(text) < 3:
        return False

    # Normalize: lowercase, remove leading punctuation
    normalized = text.lower().strip().lstrip('.-0123456789) ')

    # Check if starts with common instruction verb
    for verb in INSTRUCTION_VERBS:
        if normalized.startswith(verb + ' ') or normalized == verb:
            return True

    return False


def find_ingredient_block_end(labels: List) -> int:
    """Find the index where the ingredient block ends.

    Returns:
        Index of last INGREDIENT_LINE token + 1, or -1 if no ingredients found
    """
    ingredient_id = label2id["INGREDIENT_LINE"]
    last_ingredient_idx = -1

    for i, label in enumerate(labels):
        # Handle both string and integer labels
        if label == ingredient_id or label == "INGREDIENT_LINE":
            last_ingredient_idx = i

    return last_ingredient_idx + 1 if last_ingredient_idx >= 0 else -1


def reconstruct_lines(words: List[str], bboxes: List[List[float]]) -> List[tuple[int, int, str]]:
    """Reconstruct text lines from words and bboxes.

    Returns:
        List of (start_idx, end_idx, line_text) tuples
    """
    if not words or not bboxes or len(words) != len(bboxes):
        return []

    lines = []
    current_line_start = 0
    current_line_words = [words[0]]
    current_y = bboxes[0][1]  # y-coordinate of first bbox

    # Group words into lines based on y-coordinate proximity
    y_threshold = 20  # pixels - tune based on DPI

    for i in range(1, len(words)):
        word = words[i]
        bbox = bboxes[i]

        # Check if this word is on a new line
        if abs(bbox[1] - current_y) > y_threshold:
            # Save current line
            line_text = ' '.join(current_line_words)
            lines.append((current_line_start, i, line_text))

            # Start new line
            current_line_start = i
            current_line_words = [word]
            current_y = bbox[1]
        else:
            current_line_words.append(word)

    # Add final line
    if current_line_words:
        line_text = ' '.join(current_line_words)
        lines.append((current_line_start, len(words), line_text))

    return lines


def add_heuristic_instruction_labels(
    page: Dict[str, Any],
    label_after_ingredients: bool = True,
    label_verb_lines: bool = True,
    min_line_length: int = 3,
) -> Dict[str, Any]:
    """Add heuristic INSTRUCTION_STEP labels to a page.

    Args:
        page: Page data with words, bboxes, labels
        label_after_ingredients: Label text after ingredients as INSTRUCTION_STEP
        label_verb_lines: Label verb-leading lines as INSTRUCTION_STEP
        min_line_length: Minimum line length (in chars) to consider for labeling

    Returns:
        Modified page with added instruction labels
    """
    words = page.get("words", [])
    bboxes = page.get("bboxes", [])
    labels = page.get("labels", ["O"] * len(words))  # Default to STRING labels

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

    if label_format == "string":
        instruction_label = "INSTRUCTION_STEP"
        o_label = "O"
    else:
        instruction_label = label2id["INSTRUCTION_STEP"]
        o_label = label2id["O"]

    # Find where ingredients end
    ingredients_end_idx = find_ingredient_block_end(labels)

    # Reconstruct lines from words
    lines = reconstruct_lines(words, bboxes)

    # Track stats
    labeled_lines = 0
    labeled_tokens = 0

    for start_idx, end_idx, line_text in lines:
        # Skip short lines
        if len(line_text) < min_line_length:
            continue

        # Check if this line should be labeled
        should_label = False

        # Heuristic 1: Label lines after ingredients block
        if label_after_ingredients and ingredients_end_idx > 0:
            if start_idx >= ingredients_end_idx:
                should_label = True

        # Heuristic 2: Label verb-leading lines
        if label_verb_lines and is_likely_instruction(line_text):
            should_label = True

        # Apply label if appropriate
        if should_label:
            # Label tokens marked as O or TITLE (TITLE after ingredients is likely mislabeled instructions)
            title_label = "TITLE" if label_format == "string" else label2id["TITLE"]

            for i in range(start_idx, end_idx):
                # Convert O or TITLE tokens to INSTRUCTION_STEP
                if labels[i] == o_label or (start_idx >= ingredients_end_idx and labels[i] == title_label):
                    labels[i] = instruction_label
                    labeled_tokens += 1

            if labeled_tokens > 0:
                labeled_lines += 1

    # Update page with new labels
    page["labels"] = labels

    # Add metadata about heuristic labeling
    if "heuristic_labeling" not in page:
        page["heuristic_labeling"] = {}

    page["heuristic_labeling"]["instruction_labels_added"] = labeled_tokens
    page["heuristic_labeling"]["instruction_lines_labeled"] = labeled_lines

    return page


def process_jsonl(
    in_jsonl: Path,
    out_jsonl: Path,
    label_after_ingredients: bool = True,
    label_verb_lines: bool = True,
    min_line_length: int = 3,
    min_ingredient_tokens: int = 5,
) -> Dict[str, int]:
    """Process JSONL file and add heuristic instruction labels.

    Args:
        in_jsonl: Input JSONL file
        out_jsonl: Output JSONL file
        label_after_ingredients: Enable after-ingredients heuristic
        label_verb_lines: Enable verb-line heuristic
        min_line_length: Minimum line length for labeling
        min_ingredient_tokens: Only add labels to pages with this many ingredient tokens

    Returns:
        Stats dict with counts
    """
    stats = Counter(
        total_pages=0,
        pages_with_ingredients=0,
        pages_labeled=0,
        total_instruction_tokens_added=0,
        total_instruction_lines_labeled=0,
    )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    ingredient_id = label2id["INGREDIENT_LINE"]

    with in_jsonl.open() as fin, out_jsonl.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue

            page = json.loads(line)
            stats["total_pages"] += 1

            # Check if page has ingredients (handle both string and int labels)
            labels = page.get("labels", [])
            ingredient_count = sum(1 for l in labels if l == ingredient_id or l == "INGREDIENT_LINE")

            if ingredient_count >= min_ingredient_tokens:
                stats["pages_with_ingredients"] += 1

                # Add heuristic labels
                page = add_heuristic_instruction_labels(
                    page,
                    label_after_ingredients=label_after_ingredients,
                    label_verb_lines=label_verb_lines,
                    min_line_length=min_line_length,
                )

                # Track stats
                if page.get("heuristic_labeling", {}).get("instruction_labels_added", 0) > 0:
                    stats["pages_labeled"] += 1
                    stats["total_instruction_tokens_added"] += page["heuristic_labeling"]["instruction_labels_added"]
                    stats["total_instruction_lines_labeled"] += page["heuristic_labeling"]["instruction_lines_labeled"]

            fout.write(json.dumps(page) + "\n")

    return dict(stats)


def main():
    parser = argparse.ArgumentParser(
        description="Add heuristic INSTRUCTION_STEP labels to pages"
    )
    parser.add_argument(
        "--in_jsonl",
        type=Path,
        required=True,
        help="Input JSONL file with OCR/labels"
    )
    parser.add_argument(
        "--out_jsonl",
        type=Path,
        required=True,
        help="Output JSONL file with added instruction labels"
    )
    parser.add_argument(
        "--label-after-ingredients",
        action="store_true",
        default=True,
        help="Label text after ingredients block as INSTRUCTION_STEP"
    )
    parser.add_argument(
        "--no-label-after-ingredients",
        action="store_false",
        dest="label_after_ingredients",
        help="Disable after-ingredients heuristic"
    )
    parser.add_argument(
        "--label-verb-lines",
        action="store_true",
        default=True,
        help="Label verb-leading lines as INSTRUCTION_STEP"
    )
    parser.add_argument(
        "--no-label-verb-lines",
        action="store_false",
        dest="label_verb_lines",
        help="Disable verb-line heuristic"
    )
    parser.add_argument(
        "--min-line-length",
        type=int,
        default=3,
        help="Minimum line length (chars) to consider for labeling"
    )
    parser.add_argument(
        "--min-ingredient-tokens",
        type=int,
        default=5,
        help="Only add labels to pages with at least this many ingredient tokens"
    )

    args = parser.parse_args()

    print(f"Processing {args.in_jsonl}")
    print(f"Heuristics enabled:")
    print(f"  - Label after ingredients: {args.label_after_ingredients}")
    print(f"  - Label verb lines: {args.label_verb_lines}")

    stats = process_jsonl(
        args.in_jsonl,
        args.out_jsonl,
        label_after_ingredients=args.label_after_ingredients,
        label_verb_lines=args.label_verb_lines,
        min_line_length=args.min_line_length,
        min_ingredient_tokens=args.min_ingredient_tokens,
    )

    print(f"\nHeuristic labeling complete!")
    print(f"  Total pages: {stats['total_pages']}")
    print(f"  Pages with ingredients: {stats['pages_with_ingredients']}")
    print(f"  Pages labeled with instructions: {stats['pages_labeled']}")
    print(f"  Total instruction tokens added: {stats['total_instruction_tokens_added']}")
    print(f"  Total instruction lines labeled: {stats['total_instruction_lines_labeled']}")
    print(f"  Output: {args.out_jsonl}")


if __name__ == "__main__":
    main()
