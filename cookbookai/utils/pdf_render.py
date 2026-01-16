from __future__ import annotations

import pathlib
from typing import Optional

try:
    import fitz  # PyMuPDF
except Exception as exc:  # pragma: no cover
    fitz = None
    _import_error = exc
else:
    _import_error = None


def render_pdf_to_pngs(
    pdf_path: pathlib.Path,
    out_dir: pathlib.Path,
    dpi: int = 200,
    max_pages: Optional[int] = None,
    start: int = 1,
    end: Optional[int] = None,
    overwrite: bool = False,
) -> int:
    """
    Render a PDF to individual PNGs (0001.png, 0002.png, ...).

    Returns the number of pages rendered.
    """
    if fitz is None:
        raise ImportError(
            "PyMuPDF (fitz) is required to render PDF pages. "
            "Install with: pip install PyMuPDF"
        ) from _import_error

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)

    page_indices = list(range(start - 1, min(end or doc.page_count, doc.page_count)))
    if max_pages is not None:
        page_indices = page_indices[:max_pages]

    rendered = 0
    for idx in page_indices:
        page = doc.load_page(idx)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        filename = f"{idx + 1:04d}.png"
        out_path = out_dir / filename
        if out_path.exists() and not overwrite:
            continue
        pix.save(out_path)
        rendered += 1

    return rendered

