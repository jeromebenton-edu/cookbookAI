"""Generate weak labels for OCR pages using heuristic rules."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List

import numpy as np

import pandas as pd
from tqdm import tqdm

from .line_grouper import group_lines
from .spacy_labeler import assign_token_labels, label_lines
from .confidence import average_confidences


def load_recipe_candidates(path: pathlib.Path, threshold: float) -> set[int]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(df.loc[df["recipe_score"] >= threshold, "page_num"].astype(int).tolist())


def _json_default(obj: Any):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError


def write_jsonl(records: List[Dict[str, Any]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, default=_json_default) + "\n")


def process_page(row: pd.Series) -> Dict[str, Any]:
    words = list(row["words"])
    bboxes = [tuple(map(int, bbox)) for bbox in row["bboxes"]]
    confs_raw = row.get("confidences")
    confidences = list(confs_raw) if isinstance(confs_raw, (list, np.ndarray)) else None
    page_height = int(row["height"])

    lines = group_lines(words, bboxes, page_height)
    line_predictions = label_lines(lines)
    token_labels, token_conf = assign_token_labels(len(words), line_predictions, confidences)

    avg_line_conf = average_confidences([ln["confidence"] for ln in line_predictions])
    avg_token_conf = average_confidences(token_conf)
    coverage = sum(1 for lbl in token_labels if lbl != "O") / max(1, len(token_labels))

    record = {
        "book": row.get("book", "Boston Cooking-School Cook Book"),
        "year": row.get("year", 1918),
        "page_num": int(row["page_num"]),
        "image_path": row["image_path"],
        "width": int(row["width"]),
        "height": int(row["height"]),
        "words": words,
        "bboxes": bboxes,
        "labels": token_labels,
        "token_confidence": token_conf,
        "lines": line_predictions,
        "page_label_quality": {
            "avg_line_confidence": avg_line_conf,
            "avg_token_confidence": avg_token_conf,
            "coverage": coverage,
        },
    }
    return record


def filter_highconf(records: List[Dict[str, Any]], min_avg_line_conf: float, min_ing: int, min_instr: int) -> List[Dict[str, Any]]:
    filtered = []
    for rec in records:
        avg_conf = rec["page_label_quality"]["avg_line_confidence"]
        labels = rec["labels"]
        if avg_conf < min_avg_line_conf:
            continue
        ing_count = labels.count("INGREDIENT_LINE")
        instr_count = labels.count("INSTRUCTION_STEP")
        if ing_count < min_ing or instr_count < min_instr:
            continue
        filtered.append(rec)
    return filtered


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Weak labeling pipeline")
    parser.add_argument("--parquet", required=True, help="OCR parquet path")
    parser.add_argument("--recipe_candidates", help="CSV of recipe candidates", default=None)
    parser.add_argument("--candidate_threshold", type=float, default=0.65, help="Threshold for recipe candidates")
    parser.add_argument("--use_all_pages", action="store_true", help="Ignore recipe gating; label all pages")
    parser.add_argument("--out", required=True, help="Output weak labels JSONL")
    parser.add_argument("--out_highconf", help="Output high-confidence JSONL", default=None)
    parser.add_argument("--min_score", type=float, default=0.65, help="Minimum recipe score to include (used if candidate file missing)")
    parser.add_argument("--min_avg_line_conf", type=float, default=0.80, help="Minimum avg line confidence for highconf")
    parser.add_argument("--min_ing", type=int, default=2, help="Min ingredient lines for highconf")
    parser.add_argument("--min_instr", type=int, default=2, help="Min instruction lines for highconf")
    parser.add_argument("--max_pages", type=int, help="Process at most N pages")
    parser.add_argument("--debug_sample", default="data/labels/debug_samples", help="Dir to write one debug page JSON")
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.parquet)
    total_available = len(df)
    candidate_pages: set[int] = set()
    if args.recipe_candidates and not args.use_all_pages:
        candidate_pages = load_recipe_candidates(pathlib.Path(args.recipe_candidates), args.candidate_threshold)
    if candidate_pages and not args.use_all_pages:
        df = df[df["page_num"].isin(candidate_pages)]
    if args.max_pages:
        df = df.head(args.max_pages)
    print(f"Pages available in parquet: {total_available} | processing: {len(df)} | use_all_pages={args.use_all_pages}")

    records: List[Dict[str, Any]] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Labeling pages"):
        rec = process_page(row)
        records.append(rec)

    out_path = pathlib.Path(args.out)
    write_jsonl(records, out_path)
    print(f"Wrote {len(records)} weak-labeled pages -> {out_path}")

    if args.out_highconf:
        high = filter_highconf(records, args.min_avg_line_conf, args.min_ing, args.min_instr)
        write_jsonl(high, pathlib.Path(args.out_highconf))
        print(f"High-confidence subset: {len(high)} pages -> {args.out_highconf}")

    debug_dir = pathlib.Path(args.debug_sample)
    if records:
        debug_dir.mkdir(parents=True, exist_ok=True)
        sample_path = debug_dir / f"page_{records[0]['page_num']:04d}.json"
        sample_path.write_text(json.dumps(records[0], indent=2), encoding="utf-8")
        print(f"Wrote debug sample -> {sample_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
