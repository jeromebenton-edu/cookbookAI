#!/usr/bin/env python3
"""Verify sample assets exist and are loadable."""

from pathlib import Path
from PIL import Image

def main():
    dest = Path("docs/samples")
    files = [
        dest / "sample_clean_printed.png",
        dest / "sample_skewed_scan.png",
        dest / "sample_low_quality.png",
    ]
    missing = [f for f in files if not f.exists()]
    if missing:
        print("Missing samples:", missing)
        exit(1)
    for f in files:
        size = f.stat().st_size
        if size < 20000:
            print(f"{f} too small ({size} bytes)")
            exit(1)
        try:
            img = Image.open(f)
            img.verify()
        except Exception as exc:
            print(f"{f} failed to load: {exc}")
            exit(1)
    print("Sample assets verified.")

if __name__ == "__main__":
    main()
