"""Build LayoutLMv3-ready datasets from weak-labeled JSONL."""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import datasets
from datasets import Features, Sequence, Value
from tqdm import tqdm

from training.datasets.bbox import normalize_bboxes
from training.datasets.labels import LABELS, id2label, label2id

logger = logging.getLogger(__name__)

FEATURES = Features(
    {
        "id": Value("string"),
        "page_num": Value("int32"),
        "image_path": Value("string"),
        "words": Sequence(Value("string")),
        "bboxes": Sequence(Sequence(Value("int32"))),
        "labels": Sequence(Value("int32")),
        "label_names": Sequence(Value("string")),
        "width": Value("int32"),
        "height": Value("int32"),
        "has_labels": Value("bool"),
        "png_id": Value("string"),
        # Label coverage statistics for filtering
        "non_o_ratio": Value("float32"),
        "non_o_token_count": Value("int32"),
        "ingredient_token_count": Value("int32"),
        "instruction_token_count": Value("int32"),
        "title_token_count": Value("int32"),
    }
)


@dataclass
class PageExample:
    page_num: int
    id: str
    image_path: str
    words: List[str]
    bboxes: List[List[int]]
    labels: List[int]
    label_names: List[str]
    width: int
    height: int
    has_labels: bool
    png_id: str
    # Label coverage statistics
    non_o_ratio: float
    non_o_token_count: int
    ingredient_token_count: int
    instruction_token_count: int
    title_token_count: int


def collapse_labels(labels: Sequence[str], collapse_note: bool) -> List[str]:
    if not collapse_note:
        return list(labels)
    collapsed = []
    for lbl in labels:
        if lbl == "NOTE":
            collapsed.append("O")
        else:
            collapsed.append(lbl)
    return collapsed


def process_record(
    rec: dict,
    collapse_note: bool,
    min_tokens: int,
    stats: dict,
) -> PageExample | None:
    page_num = int(rec.get("page_num", -1))
    words = rec.get("words") or []
    bboxes = rec.get("bboxes") or []
    labels = rec.get("labels")
    provided_labels = labels is not None and len(labels) > 0
    width = int(rec.get("width") or 0)
    height = int(rec.get("height") or 0)
    png_id = rec.get("png_id") or f"{page_num:04d}"

    if not words or not bboxes:
        stats["dropped_empty"] += 1
        return None
    if labels is None or len(labels) == 0:
        labels = ["O"] * len(words)
        stats["filled_missing_labels"] += 1
    if len(words) != len(bboxes) or len(words) != len(labels):
        stats["dropped_mismatch"] += 1
        return None
    if min_tokens and len(words) < min_tokens:
        stats["dropped_short"] += 1
        return None

    norm_labels = collapse_labels(labels, collapse_note=collapse_note)
    stats["note_collapsed"] += sum(1 for lbl in labels if lbl == "NOTE")

    label_ids: List[int] = []
    for lbl in norm_labels:
        if lbl not in label2id:
            stats["dropped_unknown"] += 1
            return None
        label_ids.append(label2id[lbl])

    try:
        norm_bboxes = normalize_bboxes(bboxes, width=width, height=height)
    except Exception:
        stats["dropped_bad_bbox"] += 1
        return None

    # Compute label coverage statistics
    total_tokens = len(norm_labels)
    non_o_count = sum(1 for lbl in norm_labels if lbl != "O")
    non_o_ratio = non_o_count / total_tokens if total_tokens > 0 else 0.0
    ingredient_count = sum(1 for lbl in norm_labels if lbl == "INGREDIENT_LINE")
    instruction_count = sum(1 for lbl in norm_labels if lbl == "INSTRUCTION_STEP")
    title_count = sum(1 for lbl in norm_labels if lbl == "TITLE")

    example = PageExample(
        page_num=page_num,
        id=f"boston_page_{page_num:04d}",
        image_path=str(rec.get("image_path")),
        words=list(words),
        bboxes=norm_bboxes,
        labels=label_ids,
        label_names=norm_labels,
        width=width,
        height=height,
        has_labels=bool(rec.get("has_labels", provided_labels)),
        png_id=png_id,
        non_o_ratio=non_o_ratio,
        non_o_token_count=non_o_count,
        ingredient_token_count=ingredient_count,
        instruction_token_count=instruction_count,
        title_token_count=title_count,
    )
    return example


def load_pages(
    jsonl_path: Path,
    collapse_note: bool,
    max_pages: int,
    min_tokens: int,
    min_non_o_ratio: float = 0.0,
    min_non_o_tokens: int = 0,
    min_ingredient_tokens: int = 0,
    min_instruction_tokens: int = 0,
    filter_recipe_only: bool = False,
) -> tuple[List[PageExample], dict]:
    """Load pages from JSONL with optional filtering for recipe content.

    Args:
        filter_recipe_only: If True, only keep pages meeting minimum supervision thresholds
        min_non_o_ratio: Minimum ratio of non-O labels (default 0.0 = no filter)
        min_non_o_tokens: Minimum count of non-O tokens (alternative to ratio)
        min_ingredient_tokens: Minimum ingredient tokens
        min_instruction_tokens: Minimum instruction tokens
    """
    stats = Counter(
        dropped_empty=0,
        dropped_mismatch=0,
        dropped_short=0,
        dropped_unknown=0,
        dropped_bad_bbox=0,
        note_collapsed=0,
        filled_missing_labels=0,
        dropped_low_coverage=0,
    )
    pages: List[PageExample] = []
    with jsonl_path.open() as f:
        for line in tqdm(f, desc="Loading pages"):
            if not line.strip():
                continue
            rec = json.loads(line)
            example = process_record(rec, collapse_note=collapse_note, min_tokens=min_tokens, stats=stats)
            if example:
                # Apply recipe-only filter if enabled
                if filter_recipe_only:
                    passes_coverage = (
                        example.non_o_ratio >= min_non_o_ratio
                        or example.non_o_token_count >= min_non_o_tokens
                        or example.ingredient_token_count >= min_ingredient_tokens
                        or example.instruction_token_count >= min_instruction_tokens
                    )
                    if not passes_coverage:
                        stats["dropped_low_coverage"] += 1
                        continue

                pages.append(example)
            if max_pages and len(pages) >= max_pages:
                break
    pages.sort(key=lambda x: x.page_num)
    return pages, stats


def rank_and_select_pages(
    pages: List[PageExample],
    max_pages: int,
    rank_by_instructions: bool = True,
    min_ingredient_coverage: float = 0.3,
) -> List[PageExample]:
    """Rank pages by label richness and select top N with ingredient coverage constraint.

    Args:
        pages: List of pages to rank
        max_pages: Maximum number of pages to keep (0 = no limit)
        rank_by_instructions: If True, use blended ranking (0.6 inst + 0.4 ing)
        min_ingredient_coverage: Minimum fraction of selected pages with ingredient_token_count >= 10

    Returns:
        Selected pages (ranked and limited to max_pages)
    """
    if max_pages == 0 or len(pages) <= max_pages:
        return pages

    # Compute blended ranking score: 0.6 * instruction + 0.4 * ingredient
    if rank_by_instructions:
        # Blended score for balanced selection
        def blended_score(p: PageExample) -> tuple:
            score = 0.6 * p.instruction_token_count + 0.4 * p.ingredient_token_count
            # Use page_num as tiebreaker for deterministic sorting
            return (score, p.page_num)

        pages_sorted = sorted(pages, key=blended_score, reverse=True)
    else:
        # Fallback: sort by total label density
        pages_sorted = sorted(
            pages,
            key=lambda p: (p.non_o_token_count, p.ingredient_token_count, p.instruction_token_count, p.page_num),
            reverse=True
        )

    # Initial selection: top N by ranking
    selected = pages_sorted[:max_pages]
    remaining = pages_sorted[max_pages:]

    # Enforce minimum ingredient coverage constraint
    min_ing_tokens = 10
    pages_with_ingredients = [p for p in selected if p.ingredient_token_count >= min_ing_tokens]
    ingredient_coverage = len(pages_with_ingredients) / len(selected) if selected else 0.0

    logger.info(f"Initial selection: {len(selected)} pages, ingredient coverage: {ingredient_coverage:.1%} ({len(pages_with_ingredients)}/{len(selected)} pages)")

    # If ingredient coverage is too low, backfill from remaining pages
    if ingredient_coverage < min_ingredient_coverage:
        logger.info(f"⚠️  Ingredient coverage ({ingredient_coverage:.1%}) below target ({min_ingredient_coverage:.1%})")
        logger.info(f"   Backfilling with ingredient-rich pages...")

        target_ing_pages = int(max_pages * min_ingredient_coverage)
        needed_ing_pages = target_ing_pages - len(pages_with_ingredients)

        # Find ingredient-rich pages from remaining pool
        remaining_with_ing = [p for p in remaining if p.ingredient_token_count >= min_ing_tokens]
        remaining_with_ing.sort(key=lambda p: (p.ingredient_token_count, p.page_num), reverse=True)

        # Backfill: swap low-ingredient pages for high-ingredient pages
        selected_ids = {p.id for p in selected}
        pages_without_ing = [p for p in selected if p.ingredient_token_count < min_ing_tokens]
        pages_without_ing.sort(key=lambda p: (p.instruction_token_count, p.page_num))  # Remove lowest instruction pages

        backfilled = 0
        for i in range(min(needed_ing_pages, len(remaining_with_ing), len(pages_without_ing))):
            # Remove page with no ingredients
            removed = pages_without_ing[i]
            selected_ids.remove(removed.id)

            # Add ingredient-rich page
            added = remaining_with_ing[i]
            selected_ids.add(added.id)
            backfilled += 1

        # Reconstruct selected list from all pages
        all_pages_by_id = {p.id: p for p in pages_sorted}
        selected = [all_pages_by_id[pid] for pid in selected_ids]
        selected.sort(key=lambda p: p.page_num)

        # Recompute coverage
        pages_with_ingredients = [p for p in selected if p.ingredient_token_count >= min_ing_tokens]
        ingredient_coverage = len(pages_with_ingredients) / len(selected) if selected else 0.0

        logger.info(f"   Backfilled {backfilled} pages")
        logger.info(f"   New ingredient coverage: {ingredient_coverage:.1%} ({len(pages_with_ingredients)}/{len(selected)} pages)")

    # Log detailed stats about final selection
    logger.info(f"\nPage selection summary:")
    logger.info(f"  Total pages kept: {len(selected)}/{len(pages)}")
    logger.info(f"  Pages with ingredient_tokens >= {min_ing_tokens}: {len(pages_with_ingredients)} ({ingredient_coverage:.1%})")

    pages_with_instructions = [p for p in selected if p.instruction_token_count >= 10]
    inst_coverage = len(pages_with_instructions) / len(selected) if selected else 0.0
    logger.info(f"  Pages with instruction_tokens >= 10: {len(pages_with_instructions)} ({inst_coverage:.1%})")

    # Compute label distribution across selected pages
    total_tokens = sum(len(p.words) for p in selected)
    total_ingredient_tokens = sum(p.ingredient_token_count for p in selected)
    total_instruction_tokens = sum(p.instruction_token_count for p in selected)
    total_title_tokens = sum(p.title_token_count for p in selected)
    total_o_tokens = total_tokens - total_ingredient_tokens - total_instruction_tokens - total_title_tokens

    logger.info(f"\nLabel distribution across selected pages:")
    logger.info(f"  Total tokens: {total_tokens}")
    logger.info(f"  INGREDIENT_LINE:  {total_ingredient_tokens:6d} ({total_ingredient_tokens/total_tokens*100:5.2f}%)")
    logger.info(f"  INSTRUCTION_STEP: {total_instruction_tokens:6d} ({total_instruction_tokens/total_tokens*100:5.2f}%)")
    logger.info(f"  TITLE:            {total_title_tokens:6d} ({total_title_tokens/total_tokens*100:5.2f}%)")
    logger.info(f"  O:                {total_o_tokens:6d} ({total_o_tokens/total_tokens*100:5.2f}%)")

    # Compute averages
    avg_ing = sum(p.ingredient_token_count for p in selected) / len(selected) if selected else 0.0
    avg_inst = sum(p.instruction_token_count for p in selected) / len(selected) if selected else 0.0
    logger.info(f"\nMean tokens per page:")
    logger.info(f"  Ingredient tokens:  {avg_ing:.1f}")
    logger.info(f"  Instruction tokens: {avg_inst:.1f}")

    return selected


def split_pages_stratified(
    pages: List[PageExample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    val_min_ingredient_pages: int = 10,
    val_min_instruction_pages: int = 10,
) -> Dict[str, List[PageExample]]:
    """Stratified split ensuring validation has pages with all label types.

    Args:
        pages: List of page examples to split
        val_ratio: Ratio of (train+val) to use for validation
        test_ratio: Ratio of total to use for test
        seed: Random seed for reproducibility
        val_min_ingredient_pages: Minimum validation pages with INGREDIENT_LINE
        val_min_instruction_pages: Minimum validation pages with INSTRUCTION_STEP

    Returns:
        Dict with 'train', 'val', 'test' splits
    """
    rng = random.Random(seed)

    n = len(pages)
    n_test = int(n * test_ratio)
    n_val = int((n - n_test) * val_ratio)

    # First, shuffle and separate test set
    shuffled = list(pages)
    rng.shuffle(shuffled)
    test_pages = shuffled[:n_test]
    remaining = shuffled[n_test:]

    # Categorize remaining pages by label content
    pages_with_ingredients = []
    pages_with_instructions = []
    pages_with_title = []
    pages_other = []

    for p in remaining:
        has_ing = p.ingredient_token_count > 0
        has_inst = p.instruction_token_count > 0
        has_title = p.title_token_count > 0

        if has_ing:
            pages_with_ingredients.append(p)
        if has_inst:
            pages_with_instructions.append(p)
        if has_title:
            pages_with_title.append(p)
        if not (has_ing or has_inst or has_title):
            pages_other.append(p)

    # Shuffle each category
    rng.shuffle(pages_with_ingredients)
    rng.shuffle(pages_with_instructions)
    rng.shuffle(pages_with_title)
    rng.shuffle(pages_other)

    # Build validation set with guaranteed label coverage
    val_pages = []
    used_page_ids = set()

    # Ensure minimum ingredient pages in validation
    target_ing = min(val_min_ingredient_pages, len(pages_with_ingredients))
    for p in pages_with_ingredients[:target_ing]:
        if p.id not in used_page_ids:
            val_pages.append(p)
            used_page_ids.add(p.id)

    # Ensure minimum instruction pages in validation
    target_inst = min(val_min_instruction_pages, len(pages_with_instructions))
    for p in pages_with_instructions[:target_inst]:
        if p.id not in used_page_ids and len(val_pages) < n_val:
            val_pages.append(p)
            used_page_ids.add(p.id)

    # Fill remaining validation slots from all categories
    remaining_pool = [p for p in remaining if p.id not in used_page_ids]
    rng.shuffle(remaining_pool)

    for p in remaining_pool:
        if len(val_pages) >= n_val:
            break
        val_pages.append(p)
        used_page_ids.add(p.id)

    # Remaining pages go to train
    train_pages = [p for p in remaining if p.id not in used_page_ids]

    # Log stratification results
    val_ing_count = sum(1 for p in val_pages if p.ingredient_token_count > 0)
    val_inst_count = sum(1 for p in val_pages if p.instruction_token_count > 0)
    val_title_count = sum(1 for p in val_pages if p.title_token_count > 0)

    train_ing_count = sum(1 for p in train_pages if p.ingredient_token_count > 0)
    train_inst_count = sum(1 for p in train_pages if p.instruction_token_count > 0)
    train_title_count = sum(1 for p in train_pages if p.title_token_count > 0)

    logger.info(f"Stratified split results:")
    logger.info(f"  Train: {len(train_pages)} pages (ING:{train_ing_count}, INST:{train_inst_count}, TITLE:{train_title_count})")
    logger.info(f"  Val:   {len(val_pages)} pages (ING:{val_ing_count}, INST:{val_inst_count}, TITLE:{val_title_count})")
    logger.info(f"  Test:  {len(test_pages)} pages")

    return {"train": train_pages, "validation": val_pages, "test": test_pages}


def split_pages(
    pages: List[PageExample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[PageExample]]:
    """Simple random split (deprecated - use split_pages_stratified for recipe datasets)."""
    rng = random.Random(seed)
    shuffled = list(pages)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(n * test_ratio)
    n_val = int((n - n_test) * val_ratio)

    test_pages = shuffled[:n_test]
    val_pages = shuffled[n_test : n_test + n_val]
    train_pages = shuffled[n_test + n_val :]

    return {"train": train_pages, "validation": val_pages, "test": test_pages}


def to_dataset_dict(splits: Dict[str, List[PageExample]]) -> datasets.DatasetDict:
    hf_splits = {}
    for split_name, examples in splits.items():
        data = {
            "id": [ex.id for ex in examples],
            "page_num": [ex.page_num for ex in examples],
            "image_path": [ex.image_path for ex in examples],
            "words": [ex.words for ex in examples],
            "bboxes": [ex.bboxes for ex in examples],
            "labels": [ex.labels for ex in examples],
            "label_names": [ex.label_names for ex in examples],
            "width": [ex.width for ex in examples],
            "height": [ex.height for ex in examples],
            "has_labels": [ex.has_labels for ex in examples],
            "png_id": [ex.png_id for ex in examples],
            "non_o_ratio": [ex.non_o_ratio for ex in examples],
            "non_o_token_count": [ex.non_o_token_count for ex in examples],
            "ingredient_token_count": [ex.ingredient_token_count for ex in examples],
            "instruction_token_count": [ex.instruction_token_count for ex in examples],
            "title_token_count": [ex.title_token_count for ex in examples],
        }
        if not examples:
            data = {k: [] for k in FEATURES}
        hf_splits[split_name] = datasets.Dataset.from_dict(data, features=FEATURES)
    return datasets.DatasetDict(hf_splits)


def write_label_map(out_dir: Path) -> None:
    data = {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}
    (out_dir / "label_map.json").write_text(json.dumps(data, indent=2))


def compute_stats(pages: Iterable[PageExample], drop_stats: dict) -> dict:
    pages = list(pages)
    token_counts = [len(p.words) for p in pages]
    total_tokens = sum(token_counts)
    label_counter = Counter()
    for p in pages:
        label_counter.update(p.label_names)
    stats = {
        "total_pages": len(pages),
        "avg_tokens_per_page": total_tokens / max(1, len(pages)),
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "label_distribution": {
            lbl: {"count": cnt, "percent": cnt / max(1, total_tokens)}
            for lbl, cnt in sorted(label_counter.items())
        },
        "percent_O_tokens": label_counter.get("O", 0) / max(1, total_tokens),
        "dropped": drop_stats,
    }
    return stats


def write_splits(out_dir: Path, splits: Dict[str, List[PageExample]], seed: int, val_ratio: float, test_ratio: float) -> dict:
    data: dict = {
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "sizes": {k: len(v) for k, v in splits.items()},
        "page_nums": {k: [p.page_num for p in v] for k, v in splits.items()},
    }
    (out_dir / "splits.json").write_text(json.dumps(data, indent=2))
    return data


def write_readme(out_dir: Path, args: argparse.Namespace, stats: dict, splits_meta: dict) -> None:
    lines = [
        "# LayoutLMv3 Dataset",
        "",
        f"- Source JSONL: {args.in_jsonl}",
        f"- Collapse NOTE: {args.collapse_note}",
        f"- Min tokens: {args.min_tokens}",
        f"- Max pages: {args.max_pages or 'all'}",
        f"- Splits (seed {splits_meta.get('seed')}): {splits_meta.get('sizes')}",
        f"- Labels: {', '.join(LABELS)} (NOTE collapsed into O)",
        "",
        "BBoxes are normalized to 0-1000 (LayoutLM convention). NOTE labels are collapsed into O to avoid noisy supervision.",
        "",
        "## Quickstart",
        f"python -m training.datasets.sanity_check --dataset_dir {out_dir} --num_samples 3",
        "",
        "## Phase 4 TODO",
        "- Align word-level labels to subwords with LayoutLMv3Processor",
        "- Cache pixel_values for faster training",
        "- Consider downsampling pages dominated by O",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LayoutLMv3 DatasetDict from weak labels")
    parser.add_argument("--in_jsonl", required=True, help="Weak-labeled JSONL path")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--test_ratio", type=float, default=0.0, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--collapse_note",
        dest="collapse_note",
        action="store_true",
        default=True,
        help="Collapse NOTE into O (default)",
    )
    parser.add_argument(
        "--no_collapse_note",
        dest="collapse_note",
        action="store_false",
        help="Disable NOTE collapse",
    )
    parser.add_argument("--max_pages", type=int, default=0, help="Limit number of pages (0 = all)")
    parser.add_argument("--min_tokens", type=int, default=40, help="Drop pages with fewer tokens")

    # Recipe-only filtering arguments
    parser.add_argument("--filter_recipe_only", action="store_true", help="Only keep pages with sufficient label coverage")
    parser.add_argument("--min_non_o_ratio", type=float, default=0.01, help="Minimum non-O label ratio (default: 0.01 = 1%%)")
    parser.add_argument("--min_non_o_tokens", type=int, default=10, help="Minimum non-O token count (default: 10)")
    parser.add_argument("--min_ingredient_tokens", type=int, default=5, help="Minimum ingredient tokens (default: 5)")
    parser.add_argument("--min_instruction_tokens", type=int, default=5, help="Minimum instruction tokens (default: 5)")

    # Page ranking and selection
    parser.add_argument("--max_recipe_pages", type=int, default=0, help="Maximum pages for recipe-only dataset (0 = no limit)")
    parser.add_argument("--rank_by_instructions", action="store_true", help="Rank pages by instruction count (prioritize instruction-rich pages)")

    # Stratified validation split arguments
    parser.add_argument("--use_stratified_split", action="store_true", help="Use stratified split to ensure validation has all label types")
    parser.add_argument("--val_min_ingredient_pages", type=int, default=10, help="Minimum validation pages with INGREDIENT_LINE (default: 10)")
    parser.add_argument("--val_min_instruction_pages", type=int, default=10, help="Minimum validation pages with INSTRUCTION_STEP (default: 10)")

    return parser.parse_args()


def validate_label_distribution(stats: dict, min_non_o_ratio: float = 0.02) -> None:
    """Fail fast when labels collapse to all-O or lack core recipe labels."""
    label_distribution = stats.get("label_distribution", {})
    total_tokens = sum(v.get("count", 0) for v in label_distribution.values())
    if total_tokens == 0:
        raise RuntimeError("Dataset contains zero tokens after preprocessing.")
    o_count = label_distribution.get("O", {}).get("count", 0)
    non_o_ratio = 1 - (o_count / max(1, total_tokens))
    missing_ingredient = label_distribution.get("INGREDIENT_LINE", {}).get("count", 0) == 0
    missing_instruction = label_distribution.get("INSTRUCTION_STEP", {}).get("count", 0) == 0

    errors = []
    if non_o_ratio < min_non_o_ratio:
        errors.append(
            f"Non-O token ratio is too low ({non_o_ratio:.4f}). Expected at least {min_non_o_ratio:.2%} with recipe labels present."
        )
    if missing_ingredient:
        errors.append("Label INGREDIENT_LINE is missing from the dataset.")
    if missing_instruction:
        errors.append("Label INSTRUCTION_STEP is missing from the dataset.")
    if errors:
        details = json.dumps(
            {"percent_O_tokens": stats.get("percent_O_tokens"), "label_distribution": label_distribution},
            indent=2,
            default=str,
        )
        raise RuntimeError("Invalid label distribution: " + " ".join(errors) + f"\nDetails: {details}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading pages from %s", args.in_jsonl)
    pages, drop_stats = load_pages(
        jsonl_path=Path(args.in_jsonl),
        collapse_note=args.collapse_note,
        max_pages=args.max_pages,
        min_tokens=args.min_tokens,
        min_non_o_ratio=args.min_non_o_ratio,
        min_non_o_tokens=args.min_non_o_tokens,
        min_ingredient_tokens=args.min_ingredient_tokens,
        min_instruction_tokens=args.min_instruction_tokens,
        filter_recipe_only=args.filter_recipe_only,
    )
    logger.info("Kept %s pages (dropped: %s)", len(pages), dict(drop_stats))

    # Rank and select top N pages if max_recipe_pages is set
    if args.filter_recipe_only and args.max_recipe_pages > 0:
        pages = rank_and_select_pages(pages, args.max_recipe_pages, args.rank_by_instructions)

    stats = compute_stats(pages, dict(drop_stats))
    validate_label_distribution(stats)

    # Use stratified split if requested (recommended for recipe-only datasets)
    if args.use_stratified_split:
        splits = split_pages_stratified(
            pages,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            val_min_ingredient_pages=args.val_min_ingredient_pages,
            val_min_instruction_pages=args.val_min_instruction_pages,
        )
    else:
        splits = split_pages(pages, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)

    ds_dict = to_dataset_dict(splits)
    ds_path = out_dir / "dataset_dict"
    ds_dict.save_to_disk(ds_path)
    write_label_map(out_dir)
    splits_meta = write_splits(out_dir, splits, seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    write_readme(out_dir, args, stats, splits_meta)
    logger.info("Saved dataset to %s", ds_path)
    logger.info("Stats: %s", stats)


if __name__ == "__main__":
    main()
