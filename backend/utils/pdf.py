import io
from typing import Any

import fitz
from PIL import Image


def pdf_first_page_to_image(pdf_bytes: bytes, dpi: int = 200) -> Image.Image:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count == 0:
        raise ValueError("PDF has no pages")

    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=dpi)
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    return image
