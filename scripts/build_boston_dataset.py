#!/usr/bin/env python3
"""Rebuild the Boston cookbook HuggingFace dataset from PDF pages (OCR + weak labels).

Steps:
1) Ensure pages are rendered (see scripts/render_boston_pages.py).
2) OCR pages to JSONL.
3) Merge weak labels from multiple sources (gold > corrections > curated > weak).
4) Build DatasetDict.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        sys.exit(res.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HF dataset from rendered Boston pages")
    parser.add_argument("--pdf", type=Path, default=Path("data/raw/boston-cooking-school-1918.pdf"))
    parser.add_argument("--pages-dir", type=Path, default=Path("data/pages/boston"))
    parser.add_argument("--ocr-jsonl", type=Path, default=Path("data/ocr/boston_pages.jsonl"))
    parser.add_argument(
        "--labels-jsonl",
        type=Path,
        default=Path("data/labels/boston_weak_labeled.jsonl"),
        help="Weak-labeled JSONL to merge (optional; falls back to OCR-only)",
    )
    parser.add_argument(
        "--curated-labels-jsonl",
        type=Path,
        default=Path("data/labels/boston_curated_weak_labeled.jsonl"),
        help="Curated weak labels from manual recipes (higher priority than weak labels)",
    )
    parser.add_argument(
        "--gold-labels-jsonl",
        type=Path,
        default=Path("data/gold/boston_gold.jsonl"),
        help="Gold/ground-truth labels (highest priority)",
    )
    parser.add_argument(
        "--corrections-jsonl",
        type=Path,
        default=Path("data/corrections/boston_corrections.jsonl"),
        help="User corrections from Compare Mode (priority: gold > corrections > curated > weak)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/datasets/boston_layoutlmv3_full"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-pages", type=int, default=None, help="Limit pages (dev only)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ocr-backend", choices=["tesseract", "layoutlmv3"], default="tesseract")
    parser.add_argument("--build-recipe-only", action="store_true", help="Also build recipe-only filtered dataset")
    parser.add_argument("--add-heuristic-instructions", action="store_true", default=True, help="Add heuristic INSTRUCTION_STEP labels")
    parser.add_argument("--no-heuristic-instructions", action="store_false", dest="add_heuristic_instructions", help="Disable heuristic instruction labeling")
    parser.add_argument("--add-heuristic-ingredients", action="store_true", default=True, help="Add heuristic INGREDIENT_LINE labels")
    parser.add_argument("--no-heuristic-ingredients", action="store_false", dest="add_heuristic_ingredients", help="Disable heuristic ingredient labeling")
    return parser.parse_args()


def env_max_pages(cli_max: int | None) -> int | None:
    env_val = os.getenv("COOKBOOKAI_MAX_PAGES")
    if cli_max:
        return cli_max if cli_max != 0 else None
    if env_val:
        try:
            return int(env_val)
        except ValueError:
            return None
    return None


def load_label_map(labels_jsonl: Path, source_name: str = "labels") -> dict[int, dict]:
    """Load weak labels keyed by page_num."""
    if not labels_jsonl.exists():
        print(f"No {source_name} found at {labels_jsonl}")
        return {}
    label_map: dict[int, dict] = {}
    with labels_jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            page_num = int(rec.get("page_num", -1))
            if page_num < 0:
                continue
            label_map[page_num] = rec
    print(f"Loaded {source_name} for {len(label_map)} pages from {labels_jsonl}")
    return label_map


def merge_labels_multi_source(
    ocr_jsonl: Path,
    weak_labels_jsonl: Path,
    curated_labels_jsonl: Path,
    gold_labels_jsonl: Path,
    corrections_jsonl: Path,
    merged_jsonl: Path,
    max_pages: int | None
) -> tuple[int, int, dict]:
    """
    Merge multiple label sources with priority: gold > corrections > curated > weak.

    Returns:
        (labeled_count, total_written, source_stats)
    """
    # Load all label sources
    weak_map = load_label_map(weak_labels_jsonl, "weak labels")
    curated_map = load_label_map(curated_labels_jsonl, "curated labels")
    gold_map = load_label_map(gold_labels_jsonl, "gold labels")
    corrections_map = load_label_map(corrections_jsonl, "corrections")

    # Track which source was used for each page
    source_stats = Counter()

    merged_jsonl.parent.mkdir(parents=True, exist_ok=True)
    labeled = 0
    written = 0

    with ocr_jsonl.open() as fin, merged_jsonl.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            page_num = int(rec.get("page_num", -1))

            # Determine label source (priority: gold > corrections > curated > weak)
            label_source = None
            if page_num in gold_map:
                merged = dict(gold_map[page_num])
                label_source = "gold"
            elif page_num in corrections_map:
                merged = dict(corrections_map[page_num])
                label_source = "corrections"
            elif page_num in curated_map:
                merged = dict(curated_map[page_num])
                label_source = "curated"
            elif page_num in weak_map:
                merged = dict(weak_map[page_num])
                label_source = "weak"
            else:
                merged = rec
                label_source = "ocr_only"

            # Backfill any missing metadata from OCR
            if label_source != "ocr_only":
                merged.setdefault("image_path", rec.get("image_path"))
                merged.setdefault("width", rec.get("width"))
                merged.setdefault("height", rec.get("height"))
                merged.setdefault("words", rec.get("words", []))
                merged.setdefault("bboxes", rec.get("bboxes", []))
                merged["has_labels"] = True
                merged["label_source"] = label_source
                labeled += 1
            else:
                merged.setdefault("has_labels", False)
                merged["label_source"] = label_source

            source_stats[label_source] += 1

            fout.write(json.dumps(merged) + "\n")
            written += 1
            if max_pages and written >= max_pages:
                break

    print(f"\nMerge summary:")
    print(f"  Total pages: {written}")
    print(f"  Pages with labels: {labeled}")
    print(f"  Label sources:")
    for source, count in sorted(source_stats.items()):
        pct = count / written * 100 if written > 0 else 0
        print(f"    {source:15s}: {count:4d} ({pct:5.1f}%)")
    print(f"  Output: {merged_jsonl}")

    return labeled, written, dict(source_stats)


def validate_label_distribution(merged_jsonl: Path) -> None:
    """
    Validate label distribution and warn about potential issues.

    Checks:
    - INSTRUCTION_STEP should be at least 1/5 of INGREDIENT_LINE
    - Reports top pages with instruction tokens
    """
    print(f"\nValidating label distribution in {merged_jsonl}")

    # Import label schema
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from training.datasets.labels import LABELS, label2id

    total_counts = Counter()
    pages_with_instructions = []

    with merged_jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)

            if not rec.get("has_labels", False):
                continue

            labels = rec.get("labels", [])
            page_num = rec.get("page_num", -1)

            # Count labels (can be strings or integers)
            page_counts = Counter(labels)
            for label_id, count in page_counts.items():
                if isinstance(label_id, str):
                    label_name = label_id
                elif isinstance(label_id, int) and label_id < len(LABELS):
                    label_name = LABELS[label_id]
                else:
                    continue
                total_counts[label_name] += count

            # Track pages with instructions (check both string and int forms)
            inst_count = page_counts.get("INSTRUCTION_STEP", 0) + page_counts.get(label2id["INSTRUCTION_STEP"], 0)
            if inst_count > 0:
                pages_with_instructions.append((page_num, inst_count))

    # Print distribution
    print(f"\nLabel distribution:")
    total = sum(total_counts.values())
    for label in LABELS:
        count = total_counts.get(label, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {label:20s}: {count:6d} ({pct:5.2f}%)")

    # Validate INSTRUCTION_STEP ratio
    ing_count = total_counts.get("INGREDIENT_LINE", 0)
    inst_count = total_counts.get("INSTRUCTION_STEP", 0)

    if ing_count > 0:
        inst_ratio = inst_count / ing_count
        expected_min_ratio = 0.2  # 1/5

        print(f"\nInstruction/Ingredient ratio: {inst_ratio:.3f} (expected >{expected_min_ratio:.3f})")

        if inst_ratio < expected_min_ratio:
            print(f"⚠️  WARNING: INSTRUCTION_STEP count ({inst_count}) is less than 1/5 of INGREDIENT_LINE ({ing_count})")
            print(f"   This may indicate missing instruction labels.")
            print(f"   Consider running: make generate-curated-labels")
        else:
            print(f"✓ Instruction ratio looks healthy")

    # Report top pages with instructions
    if pages_with_instructions:
        pages_with_instructions.sort(key=lambda x: x[1], reverse=True)
        print(f"\nTop 10 pages with INSTRUCTION_STEP tokens:")
        for page_num, count in pages_with_instructions[:10]:
            print(f"  Page {page_num:4d}: {count:4d} tokens")
    else:
        print(f"\n⚠️  WARNING: No pages found with INSTRUCTION_STEP labels!")


def main() -> None:
    args = parse_args()
    max_pages = env_max_pages(args.max_pages)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = out_dir / "dataset_dict"
    if args.overwrite and dataset_dir.exists():
        import shutil

        print(f"Removing existing dataset dir at {dataset_dir}")
        shutil.rmtree(dataset_dir)

    # 1) Render pages if needed
    if not args.pages_dir.exists() or not list(args.pages_dir.glob("*.png")):
        render_cmd = [
            sys.executable,
            "scripts/render_boston_pages.py",
            "--pdf",
            str(args.pdf),
            "--out-dir",
            str(args.pages_dir),
            "--dpi",
            str(args.dpi),
            "--start",
            "1",
        ]
        if max_pages:
            render_cmd += ["--max-pages", str(max_pages)]
        if args.overwrite:
            render_cmd.append("--overwrite")
        run(render_cmd)

    # 2) OCR pages (skip if exists unless overwrite)
    if args.ocr_jsonl.exists() and not args.overwrite:
        print(f"OCR JSONL exists at {args.ocr_jsonl}, skipping (use --overwrite to rebuild).")
    else:
        ocr_cmd = [
            sys.executable,
            "data/scripts/ocr_pages.py",
            "--images_dir",
            str(args.pages_dir),
            "--out",
            str(args.ocr_jsonl),
            "--ocr_backend",
            args.ocr_backend,
            "--lang",
            "eng",
        ]
        if max_pages:
            ocr_cmd += ["--max_pages", str(max_pages)]
        run(ocr_cmd)

    # 3) Merge weak labels from multiple sources (gold > corrections > curated > weak)
    merged_jsonl = out_dir / "merged_pages.jsonl"
    labeled, written, source_stats = merge_labels_multi_source(
        args.ocr_jsonl,
        args.labels_jsonl,
        args.curated_labels_jsonl,
        args.gold_labels_jsonl,
        args.corrections_jsonl,
        merged_jsonl,
        max_pages=max_pages
    )

    # 3.5) Add heuristic instruction labels if enabled
    if args.add_heuristic_instructions:
        print("\n" + "="*80)
        print("Adding heuristic INSTRUCTION_STEP labels...")
        print("="*80)
        heuristic_inst_jsonl = out_dir / "merged_pages_with_inst_heuristics.jsonl"
        heuristic_cmd = [
            sys.executable,
            "scripts/add_heuristic_instruction_labels.py",
            "--in_jsonl",
            str(merged_jsonl),
            "--out_jsonl",
            str(heuristic_inst_jsonl),
            "--label-after-ingredients",
            "--label-verb-lines",
            "--min-ingredient-tokens",
            "0",  # Process ALL pages, not just those with ingredients
        ]
        run(heuristic_cmd)
        # Use heuristic-enhanced file for subsequent steps
        merged_jsonl = heuristic_inst_jsonl

    # 3.6) Add heuristic ingredient labels if enabled
    if args.add_heuristic_ingredients:
        print("\n" + "="*80)
        print("Adding heuristic INGREDIENT_LINE labels...")
        print("="*80)
        heuristic_ing_jsonl = out_dir / "merged_pages_with_heuristics.jsonl"
        heuristic_ing_cmd = [
            sys.executable,
            "scripts/add_heuristic_ingredient_labels.py",
            "--input",
            str(merged_jsonl),
            "--output",
            str(heuristic_ing_jsonl),
            "--min-line-length",
            "3",
        ]
        run(heuristic_ing_cmd)
        # Use heuristic-enhanced file for subsequent steps
        merged_jsonl = heuristic_ing_jsonl

    # 3.7) Validate label distribution
    validate_label_distribution(merged_jsonl)

    # 4) Build full dataset (all pages with labels)
    print("\n" + "="*80)
    print("Building FULL dataset (all labeled pages)...")
    print("="*80)
    build_cmd_full = [
        sys.executable,
        "-m",
        "training.datasets.build_layoutlm_dataset",
        "--in_jsonl",
        str(merged_jsonl),
        "--out_dir",
        str(out_dir),
        "--val_ratio",
        "0.0",
        "--test_ratio",
        "0.0",
        "--min_tokens",
        "5",
        "--max_pages",
        str(max_pages or 0),
    ]
    run(build_cmd_full)

    # 5) Build recipe-only dataset if requested
    if args.build_recipe_only:
        print("\n" + "="*80)
        print("Building RECIPE-ONLY dataset (dense supervision only)...")
        print("="*80)
        recipe_only_dir = out_dir.parent / "boston_layoutlmv3_recipe_only"
        recipe_only_dir.mkdir(parents=True, exist_ok=True)

        build_cmd_recipe = [
            sys.executable,
            "-m",
            "training.datasets.build_layoutlm_dataset",
            "--in_jsonl",
            str(merged_jsonl),
            "--out_dir",
            str(recipe_only_dir),
            "--val_ratio",
            "0.15",  # Use 15% validation split
            "--test_ratio",
            "0.0",
            "--min_tokens",
            "5",
            "--max_pages",
            str(max_pages or 0),
            "--filter_recipe_only",
            "--min_non_o_ratio",
            "0.02",
            "--min_non_o_tokens",
            "10",
            "--min_ingredient_tokens",
            "10",
            "--min_instruction_tokens",
            "10",
            "--max_recipe_pages",
            "250",
            "--rank_by_instructions",
            "--use_stratified_split",  # Enable stratified split
            "--val_min_ingredient_pages",
            "10",
            "--val_min_instruction_pages",
            "10",
        ]
        run(build_cmd_recipe)
        print(f"\n✓ Recipe-only dataset created at {recipe_only_dir}")
        print(f"   Run sanity checks: make sanity-labels DATASET_DIR={recipe_only_dir}/dataset_dict")


if __name__ == "__main__":
    main()
