from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

from PIL import Image
import pytesseract
from pytesseract import Output

try:
    from backend.app.utils.ocr_enhanced import postprocess_word
    HAS_POSTPROCESSING = True
except ImportError:
    HAS_POSTPROCESSING = False


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def image_to_ocr_tokens(image_path: Path, apply_postprocessing: bool = True) -> Dict:
    """Extract OCR tokens from image with optional post-processing.

    Args:
        image_path: Path to image file
        apply_postprocessing: Apply OCR error corrections (default: True)

    Returns:
        Dictionary with words, bboxes, confs, width, height
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    words: List[str] = []
    bboxes: List[List[int]] = []
    confs: List[float] = []
    for text, conf, left, top, w, h in zip(
        data["text"], data["conf"], data["left"], data["top"], data["width"], data["height"]
    ):
        if not text or text.strip() == "" or int(conf) < 0:
            continue

        # Apply post-processing corrections if enabled
        word = text.strip()
        if apply_postprocessing and HAS_POSTPROCESSING:
            word = postprocess_word(word)

        x0 = int(left / width * 1000)
        y0 = int(top / height * 1000)
        x1 = int((left + w) / width * 1000)
        y1 = int((top + h) / height * 1000)
        words.append(word)
        bboxes.append([x0, y0, x1, y1])
        confs.append(_clamp01(float(conf) / 100.0))

    # stable ordering top-to-bottom then left-to-right
    order = sorted(range(len(words)), key=lambda i: (bboxes[i][1], bboxes[i][0]))
    words = [words[i] for i in order]
    bboxes = [bboxes[i] for i in order]
    confs = [confs[i] for i in order]

    return {
        "words": words,
        "bboxes": bboxes,
        "confs": confs,
        "width": width,
        "height": height,
    }
