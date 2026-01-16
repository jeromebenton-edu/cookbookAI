#!/usr/bin/env python
"""
Quick sanity check for model label mappings and predictions.

Runs on a few pages and reports label distributions.
Exits non-zero if INGREDIENT_LINE or INSTRUCTION_STEP are missing or never predicted.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "backend"))

from app.services import get_service  # noqa: E402
from collections import Counter  # noqa: E402


def count_labels(tokens):
    c = Counter()
    for t in tokens or []:
        lbl = t.get("pred_label") or t.get("label") or "O"
        c[lbl] += 1
    return c


def main():
    svc = get_service()
    svc.load()
    if not svc._model:
        print("Model not loaded", file=sys.stderr)
        sys.exit(1)
    id2label = {int(k): v for k, v in svc._model.config.id2label.items()}
    print("Model dir:", svc.model_dir)
    print("Labels:", id2label)
    required = {"INGREDIENT_LINE", "INSTRUCTION_STEP"}
    if not required.issubset(set(id2label.values())):
        print("Missing required labels:", required - set(id2label.values()), file=sys.stderr)
        sys.exit(2)

    pages = [96, 97, 386]
    available = svc.list_pages()
    if not available:
        print("No pages available.", file=sys.stderr)
        sys.exit(3)
    # replace with first available pages if defaults missing
    pages = [p for p in pages if p in available] or available[:3]

    global_counts = Counter()
    for pid in pages:
        try:
            overlay = svc.predict_page(pid, grouped=False, min_conf=0.0, refresh=True)
        except Exception as exc:
            print(f"Error on page {pid}: {exc}", file=sys.stderr)
            continue
        counts = count_labels(overlay.get("tokens", []))
        global_counts.update(counts)
        print(f"\nPage {pid} label counts:", dict(counts))

    if global_counts.get("INGREDIENT_LINE", 0) == 0 or global_counts.get("INSTRUCTION_STEP", 0) == 0:
        print("INGREDIENT_LINE or INSTRUCTION_STEP never predicted in sample pages.", file=sys.stderr)
        sys.exit(4)

    print("\nAggregate counts:", dict(global_counts))
    print("Sanity check passed.")


if __name__ == "__main__":
    main()
