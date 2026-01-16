#!/usr/bin/env python3
"""Convert a Label Studio export (JSON) into gold JSONL aligned to weak labels.

Assumes the LS tasks came from the weak-label export, so line bboxes largely
overlap. We match annotated rectangles back to weak-label lines via IoU and
update line labels and token labels accordingly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


def load_weak(path: Path) -> Dict[int, dict]:
    data: Dict[int, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            data[int(rec["page_num"])] = rec
    return data


def iou(b1: List[float], b2: List[float]) -> float:
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter <= 0:
        return 0.0
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / max(1e-9, area1 + area2 - inter)


def rect_to_bbox(value: dict, orig_w: float, orig_h: float) -> List[float]:
    x0 = value["x"] / 100 * orig_w
    y0 = value["y"] / 100 * orig_h
    x1 = x0 + value["width"] / 100 * orig_w
    y1 = y0 + value["height"] / 100 * orig_h
    return [x0, y0, x1, y1]


def apply_annotations(weak_rec: dict, results: Iterable[dict]) -> dict:
    lines = weak_rec.get("lines", []) or []
    labels = weak_rec.get("labels", []) or []
    width = weak_rec.get("width")
    height = weak_rec.get("height")
    for res in results:
        if res.get("type") != "rectanglelabels":
            continue
        rect_lbls = res.get("value", {}).get("rectanglelabels") or []
        if not rect_lbls:
            continue
        label = rect_lbls[0]
        bbox = rect_to_bbox(res["value"], width, height)

        # Find best matching line by IoU
        best_idx = None
        best_iou = 0.0
        for idx, line in enumerate(lines):
            lb = line.get("line_bbox") or []
            if len(lb) != 4:
                continue
            score = iou(lb, bbox)
            if score > best_iou:
                best_iou = score
                best_idx = idx
        if best_idx is None:
            continue

        lines[best_idx]["label"] = label
        # Update token-level labels for words in this line
        for wi in lines[best_idx].get("word_indices", []):
            if wi < len(labels):
                labels[wi] = label

    weak_rec["lines"] = lines
    weak_rec["labels"] = labels
    return weak_rec


def convert(ls_export: Path, weak_labels: Path, out_path: Path) -> int:
    tasks = json.loads(ls_export.read_text())
    weak_map = load_weak(weak_labels)

    gold_records = []
    for task in tasks:
        page_num = task.get("data", {}).get("page_num")
        if page_num is None:
            continue
        anns = task.get("annotations") or []
        if not anns:
            continue
        results = anns[-1].get("result") or []
        if not results:
            continue
        weak_rec = weak_map.get(int(page_num))
        if not weak_rec:
            continue
        gold_records.append(apply_annotations(weak_rec, results))

    with out_path.open("w") as f:
        for rec in gold_records:
            f.write(json.dumps(rec) + "\n")
    return len(gold_records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LS export JSON to gold JSONL")
    parser.add_argument("--ls_export", required=True, help="Path to Label Studio export JSON")
    parser.add_argument("--weak_labels", required=True, help="Weak labels JSONL used to seed tasks")
    parser.add_argument("--out", required=True, help="Output gold JSONL")
    args = parser.parse_args()

    count = convert(Path(args.ls_export), Path(args.weak_labels), Path(args.out))
    print(f"Wrote {count} gold records -> {args.out}")


if __name__ == "__main__":
    main()
