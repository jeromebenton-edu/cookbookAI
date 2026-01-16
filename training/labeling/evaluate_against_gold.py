"""Evaluate weak labels against gold annotations."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import sklearn.metrics as skm


def read_jsonl(path: pathlib.Path) -> Dict[int, dict]:
    data: Dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            data[int(rec["page_num"])] = rec
    return data


def align_tokens(pred: dict, gold: dict) -> Tuple[List[str], List[str]]:
    p_labels = pred.get("labels", [])
    g_labels = gold.get("labels", [])
    n = min(len(p_labels), len(g_labels))
    return p_labels[:n], g_labels[:n]


def align_lines(pred: dict, gold: dict) -> Tuple[List[str], List[str]]:
    p_lines = pred.get("lines", []) or []
    g_lines = gold.get("lines", []) or []
    n = min(len(p_lines), len(g_lines))
    return [p_lines[i]["label"] for i in range(n)], [g_lines[i]["label"] for i in range(n)]


def classify_report(y_true: List[str], y_pred: List[str]) -> Dict:
    labels = sorted(set(y_true) | set(y_pred))
    report = skm.classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    cm = skm.confusion_matrix(y_true, y_pred, labels=labels)
    return {"labels": labels, "report": report, "confusion_matrix": cm.tolist()}


def worst_pages(token_acc: Dict[int, float], k: int = 5):
    return sorted(token_acc.items(), key=lambda kv: kv[1])[:k]


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Evaluate weak labels against gold")
    parser.add_argument("--gold", required=True, help="Gold JSONL path")
    parser.add_argument("--pred", required=True, help="Predicted JSONL path")
    parser.add_argument("--out_md", required=True, help="Markdown report path")
    parser.add_argument("--out_json", required=True, help="Metrics JSON path")
    args = parser.parse_args(argv)

    gold = read_jsonl(pathlib.Path(args.gold))
    pred = read_jsonl(pathlib.Path(args.pred))

    token_true: List[str] = []
    token_pred: List[str] = []
    line_true: List[str] = []
    line_pred: List[str] = []
    token_acc: Dict[int, float] = {}

    for page_num, gold_rec in gold.items():
        if page_num not in pred:
            continue
        p_tokens, g_tokens = align_tokens(pred[page_num], gold_rec)
        token_pred.extend(p_tokens)
        token_true.extend(g_tokens)
        correct = sum(1 for a, b in zip(p_tokens, g_tokens) if a == b)
        token_acc[page_num] = correct / max(1, len(g_tokens))

        p_lines, g_lines = align_lines(pred[page_num], gold_rec)
        line_pred.extend(p_lines)
        line_true.extend(g_lines)

    token_metrics = classify_report(token_true, token_pred)
    line_metrics = classify_report(line_true, line_pred) if line_true else {}

    # Build markdown
    md_lines = ["# Gold Evaluation", ""]
    md_lines.append(f"Gold pages: {len(gold)} | Evaluated pages: {len(token_acc)}")
    md_lines.append("\n## Token-level metrics")
    md_lines.append(str(token_metrics["report"]))
    if line_metrics:
        md_lines.append("\n## Line-level metrics")
        md_lines.append(str(line_metrics["report"]))

    md_lines.append("\n## Worst pages (token accuracy)")
    for page, acc in worst_pages(token_acc):
        md_lines.append(f"- Page {page}: {acc:.2f}")

    pathlib.Path(args.out_md).write_text("\n".join(md_lines), encoding="utf-8")
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"token": token_metrics, "line": line_metrics}, f, indent=2)

    print(f"Token micro F1: {token_metrics['report'].get('accuracy', 0):.3f}")
    if line_metrics:
        print(f"Line micro F1: {line_metrics['report'].get('accuracy', 0):.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
