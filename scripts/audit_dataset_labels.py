#!/usr/bin/env python
"""
Audit label distributions in the Boston LayoutLMv3 dataset.

Usage:
  python scripts/audit_dataset_labels.py [--split train|validation|test] [--sample N] [--dataset-dir PATH]
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from collections import Counter
from typing import Dict, Tuple

from datasets import load_from_disk, DatasetDict, ClassLabel, Sequence, Value
import json

DEFAULT_DATASET_DIR = Path("data/datasets/boston_layoutlmv3_full/dataset_dict")
DEFAULT_MODEL_DIRS = [
    Path("models/layoutlmv3_boston_stageB_full_latest/layoutlmv3_boston_final"),
    Path("models/layoutlmv3_boston_final"),
    Path("models/layoutlmv3_boston_stageB_full_*/layoutlmv3_boston_final"),
]
FALLBACK_ID2LABEL = {
    0: "TITLE",
    1: "INGREDIENT_LINE",
    2: "INSTRUCTION_STEP",
    3: "TIME",
    4: "TEMP",
    5: "SERVINGS",
    6: "O",
}


def _first_existing(candidates):
    for cand in candidates:
        if "*" in str(cand):
            matches = sorted(Path().glob(str(cand)))
            if matches:
                return matches[0]
        elif Path(cand).exists():
            return Path(cand)
    return None


def resolve_label_map(dataset, model_dir: Path | None = None) -> Tuple[Dict[int, str], Dict[str, int]]:
    # detect labels column
    feature_key = None
    for k in ["labels", "ner_tags", "label_ids"]:
        if k in dataset.features:
            feature_key = k
            break
    if feature_key is None:
        return dict(FALLBACK_ID2LABEL), {v: k for k, v in FALLBACK_ID2LABEL.items()}

    feat = dataset.features[feature_key]
    # handle Sequence
    if isinstance(feat, Sequence):
        feat = feat.feature

    # ClassLabel
    if isinstance(feat, ClassLabel):
        id2label = {i: name for i, name in enumerate(feat.names)}
        label2id = {v: k for k, v in id2label.items()}
        return id2label, label2id

    # Value -> try model config
    if isinstance(feat, Value):
        if model_dir is None:
            model_dir = _first_existing(DEFAULT_MODEL_DIRS)
        config_path = model_dir / "config.json" if model_dir else None
        if config_path and config_path.exists():
            cfg = json.loads(config_path.read_text())
            id2label_cfg = cfg.get("id2label") or {}
            id2label = {int(k): str(v) for k, v in id2label_cfg.items()}
            if id2label:
                label2id = {v: k for k, v in id2label.items()}
                return id2label, label2id
        id2label = dict(FALLBACK_ID2LABEL)
        label2id = {v: k for k, v in id2label.items()}
        return id2label, label2id

    # default fallback
    id2label = dict(FALLBACK_ID2LABEL)
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id


def load_dataset(dataset_dir: Path):
    ds = load_from_disk(str(dataset_dir))
    if isinstance(ds, DatasetDict):
        return ds
    return DatasetDict({"train": ds})


def audit_split(ds, split: str, model_dir: Path | None = None):
    if split not in ds:
        raise ValueError(f"Split '{split}' not found. Available: {list(ds.keys())}")
    data = ds[split]
    id2label, label2id = resolve_label_map(data, model_dir=model_dir)

    total_tokens = 0
    counts_by_id = Counter()
    per_page = []
    labels_key = None
    for k in ["labels", "ner_tags", "label_ids"]:
        if k in data.features:
            labels_key = k
            break

    for i in range(len(data)):
        labels = data[i].get(labels_key, [])
        # ensure list of ints
        labels = [int(l) for l in labels]
        total_tokens += len(labels)
        c = Counter(labels)
        counts_by_id.update(c)
        per_page.append((data[i].get("page_id", data[i].get("page_num", i)), c))

    counts_by_name = {id2label.get(k, str(k)): v for k, v in counts_by_id.items()}
    return {
        "total_tokens": total_tokens,
        "counts_by_id": counts_by_id,
        "counts_by_name": counts_by_name,
        "id2label": id2label,
        "per_page": per_page,
    }


def print_table(counts_by_name: dict, total_tokens: int):
    rows = []
    for lbl, cnt in counts_by_name.items():
        pct = (cnt / total_tokens) * 100 if total_tokens else 0.0
        rows.append((cnt, pct, lbl))
    rows.sort(reverse=True)
    print("Label distribution (sorted by frequency):")
    for cnt, pct, lbl in rows:
        print(f"{lbl:<20} {cnt:>10} ({pct:5.2f}%)")


def check_distribution(counts_by_name: dict, total_tokens: int, min_non_o_ratio: float, fail_on_invalid: bool) -> None:
    if not total_tokens:
        msg = "No tokens found."
        if fail_on_invalid:
            raise RuntimeError(msg)
        print(f"WARNING: {msg}")
        return

    o_count = counts_by_name.get("O", 0)
    non_o_ratio = 1 - (o_count / total_tokens)
    missing = [lbl for lbl in ("INGREDIENT_LINE", "INSTRUCTION_STEP") if counts_by_name.get(lbl, 0) == 0]
    errors = []
    if non_o_ratio < min_non_o_ratio:
        errors.append(f"Non-O token ratio too low ({non_o_ratio:.4f}); expected at least {min_non_o_ratio:.2%}.")
    if missing:
        errors.append(f"Missing labels: {', '.join(missing)}.")
    if errors:
        msg = " ".join(errors)
        if fail_on_invalid:
            raise RuntimeError(msg)
        print(f"WARNING: {msg}")
        return

    top_lbl, top_cnt = max(counts_by_name.items(), key=lambda kv: kv[1])
    if (top_cnt / total_tokens) > 0.9:
        print(f"WARNING: Label '{top_lbl}' accounts for {top_cnt/total_tokens:.2%} of tokens.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", help="Dataset split to audit")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="Path to dataset")
    parser.add_argument("--sample", type=int, default=0, help="Sample N pages with highest ingredient/instruction counts")
    parser.add_argument("--model-dir", default=None, help="Model dir to resolve label map when dataset lacks names")
    parser.add_argument("--min-non-o-ratio", type=float, default=0.02, help="Fail when non-O ratio falls below this")
    parser.add_argument("--fail-on-invalid", action="store_true", help="Exit non-zero if label distribution is invalid")
    args = parser.parse_args()

    ds_dir = Path(args.dataset_dir)
    ds = load_dataset(ds_dir)
    model_dir = Path(args.model_dir) if args.model_dir else None
    audit = audit_split(ds, args.split, model_dir=model_dir)

    print(f"Dataset: {ds_dir} (split: {args.split})")
    print(f"Total tokens: {audit['total_tokens']}")
    print_table(audit["counts_by_name"], audit["total_tokens"])
    check_distribution(
        audit["counts_by_name"],
        audit["total_tokens"],
        min_non_o_ratio=args.min_non_o_ratio,
        fail_on_invalid=args.fail_on_invalid,
    )

    if args.sample > 0:
        print("\nSample pages (high ingredient/instruction counts):")
        pages = audit["per_page"]
        per_label_sorted = sorted(
            pages,
            key=lambda p: (
                p[1].get("INGREDIENT_LINE", 0)
                + p[1].get("INSTRUCTION_STEP", 0)
            ),
            reverse=True,
        )
        chosen = per_label_sorted[: args.sample]
        for pid, counts in chosen:
            ing = counts.get("INGREDIENT_LINE", 0)
            instr = counts.get("INSTRUCTION_STEP", 0)
            title = counts.get("TITLE", 0)
            print(f"- Page {pid}: ING={ing}, INSTR={instr}, TITLE={title}, total={sum(counts.values())}")


if __name__ == "__main__":
    main()
