"""Build line-level labels from OCR output."""

from pathlib import Path
from typing import List


def build_lines(ocr_path: Path) -> List[str]:
    """
    TODO: Parse OCR JSON and cluster tokens into lines.
    """
    _ = ocr_path
    return []


if __name__ == "__main__":
    print("TODO: implement line builder")
