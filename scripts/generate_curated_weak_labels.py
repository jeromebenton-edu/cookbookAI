#!/usr/bin/env python3
"""Generate token-level weak labels from curated recipe JSON files.

Converts manually curated recipe JSON (ground truth) into weak-labeled JSONL
by aligning title/ingredients/instructions with OCR tokens using fuzzy matching.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from rapidfuzz import fuzz
except ImportError:
    print("Warning: rapidfuzz not available, using simple matching")
    fuzz = None

# Import label schema
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.datasets.labels import label2id, LABELS


def normalize_text(text: str) -> str:
    """Normalize text for matching: lowercase, strip punctuation, collapse spaces."""
    text = text.lower()
    # Remove punctuation except spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize_simple(text: str) -> List[str]:
    """Simple tokenization for matching."""
    return normalize_text(text).split()


def compute_token_similarity(text1: str, text2: str) -> float:
    """Compute similarity between two text strings using token set matching."""
    if fuzz:
        # Use rapidfuzz token set ratio for better fuzzy matching
        return fuzz.token_set_ratio(text1, text2) / 100.0
    else:
        # Fallback: simple token overlap
        tokens1 = set(tokenize_simple(text1))
        tokens2 = set(tokenize_simple(text2))
        if not tokens1 or not tokens2:
            return 0.0
        overlap = len(tokens1 & tokens2)
        return overlap / max(len(tokens1), len(tokens2))


def find_token_span(
    target_text: str,
    ocr_words: List[str],
    start_idx: int = 0,
    match_threshold: float = 0.7
) -> Tuple[int, int, float]:
    """
    Find the best matching span of OCR tokens for target text.

    Returns:
        (start_idx, end_idx, similarity_score)
        Returns (-1, -1, 0.0) if no good match found.
    """
    target_tokens = tokenize_simple(target_text)
    if not target_tokens:
        return (-1, -1, 0.0)

    best_start = -1
    best_end = -1
    best_score = 0.0

    # Try different window sizes around target length
    target_len = len(target_tokens)
    min_len = max(1, target_len - 2)
    max_len = target_len + 3

    # Scan through OCR tokens
    for i in range(start_idx, len(ocr_words)):
        for window_len in range(min_len, min(max_len + 1, len(ocr_words) - i + 1)):
            span_text = ' '.join(ocr_words[i:i + window_len])
            similarity = compute_token_similarity(target_text, span_text)

            if similarity > best_score:
                best_score = similarity
                best_start = i
                best_end = i + window_len

    if best_score >= match_threshold:
        return (best_start, best_end, best_score)
    else:
        return (-1, -1, best_score)


def align_text_to_tokens(
    text: str,
    ocr_words: List[str],
    label: str,
    start_idx: int = 0,
    match_threshold: float = 0.7
) -> Tuple[List[int], int, float]:
    """
    Align a text string to OCR tokens and return token labels.

    Returns:
        (labels_list, next_start_idx, match_score)
        labels_list has -1 for unmatched tokens, label_id for matched tokens
    """
    label_id = label2id.get(label, label2id["O"])

    start, end, score = find_token_span(text, ocr_words, start_idx, match_threshold)

    if start == -1:
        # No match found
        return ([], start_idx, 0.0)

    # Create labels: mark matched tokens with label_id
    labels = []
    for i in range(start, end):
        labels.append(label_id)

    return (labels, end, score)


def load_ocr_page(ocr_jsonl: Path, page_num: int) -> Dict[str, Any] | None:
    """Load OCR data for a specific page number."""
    with ocr_jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("page_num") == page_num:
                return rec
    return None


def generate_weak_labels_for_recipe(
    recipe: Dict[str, Any],
    ocr_data: Dict[str, Any],
    match_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Generate token-level weak labels for a curated recipe by aligning with OCR tokens.

    Returns:
        JSONL entry with page_num, words, bboxes, labels, source, debug info
    """
    page_num = recipe["source"]["page"]
    title = recipe.get("title", "")
    ingredients = recipe.get("ingredients", [])
    instructions = recipe.get("instructions", [])

    ocr_words = ocr_data.get("words", [])
    ocr_bboxes = ocr_data.get("bboxes", [])

    if not ocr_words:
        print(f"Warning: No OCR words for page {page_num}")
        return None

    # Initialize labels as "O" (background)
    labels = [label2id["O"]] * len(ocr_words)

    # Track matching stats
    match_stats = {
        "title": {"matched": False, "score": 0.0},
        "ingredients": [],
        "instructions": []
    }

    current_idx = 0

    # Align title
    if title:
        title_labels, next_idx, score = align_text_to_tokens(
            title, ocr_words, "TITLE", current_idx, match_threshold
        )
        if title_labels:
            for i, tid in enumerate(range(current_idx, next_idx)):
                if tid < len(labels) and i < len(title_labels):
                    labels[tid] = title_labels[i]
            current_idx = next_idx
            match_stats["title"]["matched"] = True
            match_stats["title"]["score"] = score

    # Align ingredients
    for ing in ingredients:
        if not ing.strip():
            continue
        ing_labels, next_idx, score = align_text_to_tokens(
            ing, ocr_words, "INGREDIENT_LINE", current_idx, match_threshold
        )
        if ing_labels:
            for i, tid in enumerate(range(current_idx, next_idx)):
                if tid < len(labels) and i < len(ing_labels):
                    labels[tid] = ing_labels[i]
            current_idx = next_idx
            match_stats["ingredients"].append({"text": ing, "matched": True, "score": score})
        else:
            match_stats["ingredients"].append({"text": ing, "matched": False, "score": score})

    # Align instructions
    for inst in instructions:
        if not inst.strip():
            continue
        inst_labels, next_idx, score = align_text_to_tokens(
            inst, ocr_words, "INSTRUCTION_STEP", current_idx, match_threshold
        )
        if inst_labels:
            for i, tid in enumerate(range(current_idx, next_idx)):
                if tid < len(labels) and i < len(inst_labels):
                    labels[tid] = inst_labels[i]
            current_idx = next_idx
            match_stats["instructions"].append({"text": inst, "matched": True, "score": score})
        else:
            match_stats["instructions"].append({"text": inst, "matched": False, "score": score})

    # Compute match coverage
    ing_matched = sum(1 for ing in match_stats["ingredients"] if ing["matched"])
    inst_matched = sum(1 for inst in match_stats["instructions"] if inst["matched"])
    total_fields = 1 + len(ingredients) + len(instructions)  # title + ingredients + instructions
    matched_fields = (1 if match_stats["title"]["matched"] else 0) + ing_matched + inst_matched
    match_coverage = matched_fields / total_fields if total_fields > 0 else 0.0

    # Count label distribution
    label_counts = Counter(labels)
    label_dist = {LABELS[lid]: count for lid, count in label_counts.items()}

    return {
        "page_num": page_num,
        "book": recipe.get("book", "Boston Cooking-School Cook Book"),
        "year": recipe.get("year", 1918),
        "image_path": ocr_data.get("image_path", f"data/pages/boston/{page_num:04d}.png"),
        "width": ocr_data.get("width", 750),
        "height": ocr_data.get("height", 1200),
        "words": ocr_words,
        "bboxes": ocr_bboxes,
        "labels": labels,
        "source": "curated",
        "recipe_id": recipe.get("id", f"page_{page_num}"),
        "has_labels": True,
        "match_coverage": match_coverage,
        "match_stats": match_stats,
        "label_distribution": label_dist,
    }


def load_curated_recipes(curated_dir: Path) -> List[Dict[str, Any]]:
    """Load all curated recipe JSON files."""
    recipes = []
    for json_file in sorted(curated_dir.glob("*.json")):
        try:
            with json_file.open() as f:
                recipe = json.load(f)
                # Validate required fields
                if "source" in recipe and "page" in recipe["source"]:
                    recipes.append(recipe)
                else:
                    print(f"Skipping {json_file.name}: missing source.page")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    return recipes


def main():
    parser = argparse.ArgumentParser(
        description="Generate token-level weak labels from curated recipe JSON files"
    )
    parser.add_argument(
        "--curated-dir",
        type=Path,
        default=Path("frontend/public/recipes/boston"),
        help="Directory containing curated recipe JSON files"
    )
    parser.add_argument(
        "--ocr-jsonl",
        type=Path,
        default=Path("data/ocr/boston_pages.jsonl"),
        help="OCR JSONL file with page words and bboxes"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/labels/boston_curated_weak_labeled.jsonl"),
        help="Output JSONL file with token-level labels"
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.7,
        help="Minimum similarity threshold for matching (0-1)"
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.5,
        help="Minimum match coverage to include page (0-1)"
    )

    args = parser.parse_args()

    print(f"Loading curated recipes from {args.curated_dir}")
    recipes = load_curated_recipes(args.curated_dir)
    print(f"Found {len(recipes)} curated recipes")

    if not recipes:
        print("No curated recipes found. Exiting.")
        return

    if not args.ocr_jsonl.exists():
        print(f"Error: OCR JSONL not found at {args.ocr_jsonl}")
        print("Run OCR first: make build-dataset or scripts/render_boston_pages.py")
        return

    print(f"Loading OCR data from {args.ocr_jsonl}")

    # Generate weak labels
    labeled_pages = []
    coverage_stats = []

    for recipe in recipes:
        page_num = recipe["source"]["page"]
        recipe_id = recipe.get("id", f"page_{page_num}")

        print(f"\nProcessing {recipe_id} (page {page_num})")
        print(f"  Title: {recipe.get('title', 'N/A')}")
        print(f"  Ingredients: {len(recipe.get('ingredients', []))}")
        print(f"  Instructions: {len(recipe.get('instructions', []))}")

        # Load OCR for this page
        ocr_data = load_ocr_page(args.ocr_jsonl, page_num)
        if not ocr_data:
            print(f"  Warning: No OCR data found for page {page_num}, skipping")
            continue

        # Generate labels
        labeled_page = generate_weak_labels_for_recipe(
            recipe, ocr_data, args.match_threshold
        )

        if not labeled_page:
            continue

        coverage = labeled_page["match_coverage"]
        coverage_stats.append(coverage)

        print(f"  Match coverage: {coverage:.1%}")
        print(f"  Label distribution: {labeled_page['label_distribution']}")

        if coverage >= args.min_coverage:
            labeled_pages.append(labeled_page)
            print(f"  ✓ Added (coverage: {coverage:.1%})")
        else:
            print(f"  ✗ Skipped (coverage {coverage:.1%} < threshold {args.min_coverage:.1%})")

    # Write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for page in labeled_pages:
            f.write(json.dumps(page) + "\n")

    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"{'='*80}")
    print(f"Total curated recipes: {len(recipes)}")
    print(f"Pages with OCR data: {len(coverage_stats)}")
    print(f"Pages written: {len(labeled_pages)}")
    print(f"Pages skipped (low coverage): {len(coverage_stats) - len(labeled_pages)}")

    if coverage_stats:
        avg_coverage = sum(coverage_stats) / len(coverage_stats)
        print(f"\nCoverage statistics:")
        print(f"  Average: {avg_coverage:.1%}")
        print(f"  Min: {min(coverage_stats):.1%}")
        print(f"  Max: {max(coverage_stats):.1%}")

    # Aggregate label counts
    if labeled_pages:
        total_label_counts = Counter()
        for page in labeled_pages:
            for label, count in page["label_distribution"].items():
                total_label_counts[label] += count

        print(f"\nTotal label distribution:")
        for label in LABELS:
            count = total_label_counts.get(label, 0)
            pct = count / sum(total_label_counts.values()) * 100 if total_label_counts else 0
            print(f"  {label:20s}: {count:6d} ({pct:5.2f}%)")

    print(f"\nOutput written to: {args.out}")


if __name__ == "__main__":
    main()
