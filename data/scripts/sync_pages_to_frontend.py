"""Sync rendered pages into frontend/public/recipes/boston/pages.

Example:
python data/scripts/sync_pages_to_frontend.py \
  --src data/pages/boston \
  --dst frontend/public/recipes/boston/pages \
  --mode copy
"""

import argparse
import filecmp
import pathlib
import shutil
import sys
from typing import Literal

Mode = Literal["copy", "symlink"]


def sync_pages(src: pathlib.Path, dst: pathlib.Path, mode: Mode) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")

    dst.mkdir(parents=True, exist_ok=True)
    updated = 0
    skipped = 0

    for image in sorted(src.glob("*.png")):
        target = dst / image.name
        if mode == "symlink":
            if target.exists():
                if target.is_symlink() and target.resolve() == image.resolve():
                    skipped += 1
                    continue
                target.unlink()
            target.symlink_to(image.resolve())
            updated += 1
        else:
            if target.exists() and filecmp.cmp(image, target, shallow=False):
                skipped += 1
                continue
            shutil.copy2(image, target)
            updated += 1

    print(f"Synced pages -> {dst} (updated {updated}, skipped {skipped})")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync rendered page PNGs into frontend")
    parser.add_argument("--src", required=True, help="Source directory of rendered pages")
    parser.add_argument("--dst", required=True, help="Destination directory in frontend")
    parser.add_argument("--mode", choices=["copy", "symlink"], default="copy", help="Copy or symlink files")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    sync_pages(pathlib.Path(args.src), pathlib.Path(args.dst), args.mode)


if __name__ == "__main__":
    main(sys.argv[1:])
