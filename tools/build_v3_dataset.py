#!/usr/bin/env python3
"""
Build LayoutLMv3 v3 dataset with header-aware labels and demo eval set.

This script:
1. Loads relabeled v3 JSONL data
2. Creates train/val/test splits (no page leakage across splits)
3. Creates a fixed demo_eval_set of ~20 pages representative of demo use case
4. Saves as HuggingFace dataset with manifest

Usage:
    python tools/build_v3_dataset.py \\
        --input data/processed/v3_headers_titles/boston_v3_suggested.jsonl \\
        --output data/datasets/boston_layoutlmv3_v3 \\
        --demo-pages 79,96,100,105,110

Output structure:
    output/
        dataset_dict/
            train/
            validation/
            test/
            demo_eval/
        dataset_manifest.json
"""

import argparse
import json
import logging
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import datasets
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from tqdm import tqdm

# Add ml module to path
import sys
ml_path = Path(__file__).parent.parent / "ml"
if str(ml_path) not in sys.path:
    sys.path.insert(0, str(ml_path))

from config.labels import get_label_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURES = Features({
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
    # v3-specific metadata
    "non_o_ratio": Value("float32"),
    "page_header_count": Value("int32"),
    "section_header_count": Value("int32"),
    "recipe_title_count": Value("int32"),
    "ingredient_count": Value("int32"),
    "instruction_count": Value("int32"),
})


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    logger.info(f"Loading {path}...")
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} records")
    return records


def process_record(rec: Dict, label_config: Dict) -> Dict:
    """Convert JSONL record to HF dataset format."""
    label2id = label_config["label2id"]

    page_num = rec.get("page_num", -1)
    words = rec.get("words", [])
    bboxes_raw = rec.get("bboxes", [])
    label_names = rec.get("labels", [])
    width = rec.get("width", 1000)
    height = rec.get("height", 1000)
    image_path = rec.get("image_path", "")

    # Normalize bboxes to [0, 1000] range (required by LayoutLMv3)
    def normalize_bbox(bbox, width, height):
        x1, y1, x2, y2 = bbox
        # Scale to 1000x1000 coordinate space
        x1_norm = int((x1 / width) * 1000)
        y1_norm = int((y1 / height) * 1000)
        x2_norm = int((x2 / width) * 1000)
        y2_norm = int((y2 / height) * 1000)
        # Clamp to [0, 1000]
        return [
            max(0, min(1000, x1_norm)),
            max(0, min(1000, y1_norm)),
            max(0, min(1000, x2_norm)),
            max(0, min(1000, y2_norm)),
        ]

    bboxes = [normalize_bbox(bbox, width, height) for bbox in bboxes_raw]

    # Convert label names to IDs
    labels = [label2id.get(lbl, label2id["O"]) for lbl in label_names]

    # Compute statistics
    label_counts = Counter(label_names)
    total_tokens = len(labels)
    non_o_count = sum(1 for lbl in label_names if lbl != "O")

    return {
        "id": f"page_{page_num:04d}",
        "page_num": page_num,
        "image_path": image_path,
        "words": words,
        "bboxes": bboxes,
        "labels": labels,
        "label_names": label_names,
        "width": width,
        "height": height,
        "has_labels": non_o_count > 0,
        "non_o_ratio": non_o_count / total_tokens if total_tokens > 0 else 0.0,
        "page_header_count": label_counts.get("PAGE_HEADER", 0),
        "section_header_count": label_counts.get("SECTION_HEADER", 0),
        "recipe_title_count": label_counts.get("RECIPE_TITLE", 0),
        "ingredient_count": label_counts.get("INGREDIENT_LINE", 0),
        "instruction_count": label_counts.get("INSTRUCTION_STEP", 0),
    }


def create_splits(
    records: List[Dict],
    demo_pages: Set[int],
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Create train/val/test/demo_eval splits without page leakage.

    Args:
        records: All page records
        demo_pages: Set of page numbers for demo_eval split
        val_ratio: Fraction for validation
        test_ratio: Fraction for test
        seed: Random seed

    Returns:
        Dictionary with split names -> records
    """
    random.seed(seed)

    # Separate demo pages
    demo_records = [r for r in records if r["page_num"] in demo_pages]
    non_demo_records = [r for r in records if r["page_num"] not in demo_pages]

    logger.info(f"Demo eval set: {len(demo_records)} pages")
    logger.info(f"Remaining for train/val/test: {len(non_demo_records)} pages")

    # Shuffle non-demo records
    random.shuffle(non_demo_records)

    # Calculate split sizes
    n_total = len(non_demo_records)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_test - n_val

    # Create splits
    test_records = non_demo_records[:n_test]
    val_records = non_demo_records[n_test:n_test + n_val]
    train_records = non_demo_records[n_test + n_val:]

    logger.info(f"Split sizes:")
    logger.info(f"  Train:      {len(train_records)}")
    logger.info(f"  Validation: {len(val_records)}")
    logger.info(f"  Test:       {len(test_records)}")
    logger.info(f"  Demo eval:  {len(demo_records)}")

    return {
        "train": train_records,
        "validation": val_records,
        "test": test_records,
        "demo_eval": demo_records,
    }


def build_dataset(input_path: Path, output_path: Path, demo_pages: Set[int]):
    """Build v3 dataset with all splits."""
    # Load label config
    label_config = get_label_config(version="v3")
    logger.info(f"Using label config v3 with {label_config['num_labels']} labels")

    # Load records
    records = load_jsonl(input_path)

    # Create splits
    splits = create_splits(records, demo_pages)

    # Process each split
    logger.info("Processing splits...")
    dataset_dict = {}

    for split_name, split_records in splits.items():
        logger.info(f"Processing {split_name} split...")

        processed = [process_record(rec, label_config) for rec in tqdm(split_records)]

        # Create HF dataset
        ds = Dataset.from_list(processed, features=FEATURES)
        dataset_dict[split_name] = ds

        # Print label distribution
        label_dist = Counter()
        for ex in processed:
            for lbl in ex["label_names"]:
                label_dist[lbl] += 1

        logger.info(f"{split_name} label distribution:")
        for label, count in sorted(label_dist.items(), key=lambda x: -x[1])[:8]:
            logger.info(f"  {str(label):20s}: {count:6d}")

    # Create DatasetDict
    ds_dict = DatasetDict(dataset_dict)

    # Save dataset
    logger.info(f"Saving dataset to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    ds_dict.save_to_disk(output_path / "dataset_dict")

    # Create manifest
    manifest = {
        "version": "v3_header_aware",
        "label_config": label_config,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_file": str(input_path),
        "num_labels": label_config["num_labels"],
        "splits": {
            split: len(ds) for split, ds in ds_dict.items()
        },
        "label_distribution": {},
        "demo_eval_pages": sorted(demo_pages),
    }

    # Overall label distribution
    total_label_dist = Counter()
    for split_ds in ds_dict.values():
        for ex in split_ds:
            for lbl in ex["label_names"]:
                total_label_dist[lbl] += 1

    manifest["label_distribution"] = dict(total_label_dist)

    manifest_path = output_path / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"✓ Dataset saved to {output_path}")
    logger.info(f"✓ Manifest saved to {manifest_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build v3 dataset with demo eval set")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL (v3 labels)")
    parser.add_argument("--output", type=Path, required=True, help="Output dataset directory")
    parser.add_argument(
        "--demo-pages",
        type=str,
        default="79,96",
        help="Comma-separated page numbers for demo_eval set"
    )
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.10, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Parse demo pages
    demo_pages = {int(p.strip()) for p in args.demo_pages.split(",")}
    logger.info(f"Demo eval pages: {sorted(demo_pages)}")

    manifest = build_dataset(args.input, args.output, demo_pages)

    # Print summary
    print("\n" + "=" * 80)
    print("DATASET BUILD SUMMARY")
    print("=" * 80)
    print(f"Version: {manifest['version']}")
    print(f"Total labels: {manifest['num_labels']}")
    print(f"\nSplits:")
    for split, count in manifest['splits'].items():
        print(f"  {split:12s}: {count:4d} pages")
    print(f"\nLabel distribution (top 10):")
    sorted_labels = sorted(manifest['label_distribution'].items(), key=lambda x: -x[1])[:10]
    for label, count in sorted_labels:
        print(f"  {label:20s}: {count:6d}")
    print("=" * 80)


if __name__ == "__main__":
    main()
