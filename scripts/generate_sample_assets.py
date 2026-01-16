#!/usr/bin/env python3
"""Generate sample images for demo/smoke from existing dataset pages."""

from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image, ImageFilter


def pick_pages(src_dir: Path, count: int = 3):
    files = sorted(src_dir.glob("*.png"))
    return files[:count]


def make_variants(img: Image.Image):
    skewed = img.rotate(2, expand=True, fillcolor="white")
    low = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    return skewed, low


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    dest = Path("docs/samples")
    dest.mkdir(parents=True, exist_ok=True)

    candidates = [Path("data/pages/boston"), Path("data/_generated/pages/boston")]
    src_dir = next((p for p in candidates if p.exists()), None)
    if not src_dir:
        print("No source pages found; expected data/pages/boston/*.png")
        return

    pages = pick_pages(src_dir)
    if not pages:
        print("No pages to copy.")
        return

    clean = Image.open(pages[0]).convert("RGB")
    skewed, low = make_variants(Image.open(pages[1]).convert("RGB")) if len(pages) > 1 else (clean, clean)
    low = low if len(pages) > 2 else clean

    outputs = [
        (dest / "sample_clean_printed.png", clean),
        (dest / "sample_skewed_scan.png", skewed),
        (dest / "sample_low_quality.png", low),
    ]
    for path, img in outputs:
        if path.exists() and not args.overwrite:
            print(f"Skipping existing {path}")
            continue
        img.save(path, format="PNG")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
