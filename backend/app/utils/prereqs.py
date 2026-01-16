from __future__ import annotations

import shutil
from typing import Tuple

import pytesseract


def check_tesseract_available() -> Tuple[bool, str]:
    """
    Returns (available, message).
    Checks binary presence and pytesseract version.
    """
    if not shutil.which("tesseract"):
        return False, "tesseract binary not found. Install with `sudo apt-get install tesseract-ocr` or `brew install tesseract`."
    try:
        version = pytesseract.get_tesseract_version()
        return True, f"tesseract {version}"
    except Exception as exc:  # pragma: no cover - best effort
        return False, f"tesseract check failed: {exc}"
