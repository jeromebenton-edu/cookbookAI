"""Render PDF pages to PNG images.

Example:
python data/scripts/render_pdf_pages.py \
  --pdf data/raw/boston-cooking-school-1918.pdf \
  --out data/pages/boston \
  --dpi 300 \
  --start 1 --end 50
"""

import argparse
import pathlib
import sys
from typing import Optional

import fitz  # PyMuPDF
from tqdm import tqdm


def render_pdf(
    pdf_path: pathlib.Path,
    out_dir: pathlib.Path,
    dpi: int,
    start: int,
    end: Optional[int],
    max_pages: Optional[int],
    overwrite: bool,
) -> None:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)

    page_indices = list(range(start - 1, min(end or doc.page_count, doc.page_count)))
    if max_pages is not None:
        page_indices = page_indices[:max_pages]

    for idx in tqdm(page_indices, desc="Rendering pages"):
        page = doc.load_page(idx)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        filename = f"{idx + 1:04d}.png"
        out_path = out_dir / filename
        if out_path.exists() and not overwrite:
            continue
        pix.save(out_path)

    print(f"Rendered {len(page_indices)} pages to {out_dir}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render PDF pages to PNG")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--out", required=True, help="Output directory for PNGs")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI (default 300)")
    parser.add_argument("--start", type=int, default=1, help="Start page (1-indexed)")
    parser.add_argument("--end", type=int, help="End page (inclusive, 1-indexed)")
    parser.add_argument("--max_pages", type=int, help="Process at most this many pages")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing images")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    render_pdf(
        pdf_path=pathlib.Path(args.pdf),
        out_dir=pathlib.Path(args.out),
        dpi=args.dpi,
        start=args.start,
        end=args.end,
        max_pages=args.max_pages,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
