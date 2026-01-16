#!/usr/bin/env python3
"""Precompute prediction and recipe caches for demo pages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.layoutlm_service import LayoutLMService  # noqa: E402
from backend.app.utils.recipe_extraction import cache_recipe, recipe_from_prediction  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute demo caches")
    p.add_argument("--model_dir", help="Override model dir")
    p.add_argument("--dataset_dir", help="Override dataset dir")
    p.add_argument("--out_dir", default="backend/cache", help="Cache root directory")
    p.add_argument("--limit", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    svc = LayoutLMService()
    if args.model_dir:
        svc.model_dir = Path(args.model_dir)
    if args.dataset_dir:
        svc.dataset_dir = Path(args.dataset_dir)
    svc.load()

    pages = svc.list_pages()[: args.limit]
    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    recipe_dir = out_dir / "recipes"
    pred_dir.mkdir(parents=True, exist_ok=True)
    recipe_dir.mkdir(parents=True, exist_ok=True)

    for num in pages:
        pred = svc.predict_page(num, grouped=True, min_conf=0.0, refresh=True)
        (pred_dir / f"page_{num:04d}.json").write_text(json.dumps(pred, indent=2))
        recipe = recipe_from_prediction(pred, include_raw=False)
        cache_recipe(recipe_dir / f"page_{num:04d}_recipe.json", recipe)
        print(f"Cached page {num}")

    print(f"Done. Cached {len(pages)} pages to {out_dir}")


if __name__ == "__main__":
    main()
