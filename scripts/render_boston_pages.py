#!/usr/bin/env python3
"""Render the full Boston cookbook PDF to page PNGs.

Usage:
  python scripts/render_boston_pages.py \
    --pdf data/raw/boston-cooking-school-1918.pdf \
    --out-dir data/pages/boston \
    --dpi 200

Defaults aim for the full book. Limit pages during dev with:
  COOKBOOKAI_MAX_PAGES=50 python scripts/render_boston_pages.py [...]
  python scripts/render_boston_pages.py --max-pages 50 [...]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from cookbookai.utils.pdf_render import render_pdf_to_pngs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Boston cookbook PDF to PNGs")
    parser.add_argument("--pdf", type=Path, default=Path("data/raw/boston-cooking-school-1918.pdf"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/pages/boston"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-pages", type=int, default=None, help="Limit pages (dev only)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_max = os.getenv("COOKBOOKAI_MAX_PAGES")
    max_pages = args.max_pages
    if env_max and not max_pages:
        try:
            max_pages = int(env_max)
        except ValueError:
            max_pages = None
    if max_pages == 0:
        max_pages = None

    rendered = render_pdf_to_pngs(
        pdf_path=args.pdf,
        out_dir=args.out_dir,
        dpi=args.dpi,
        start=args.start,
        end=args.end,
        max_pages=max_pages,
        overwrite=args.overwrite,
    )
    print(f"Rendered {rendered} pages to {args.out_dir}")


if __name__ == "__main__":
    main()
