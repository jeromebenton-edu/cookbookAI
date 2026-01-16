"""Enhanced OCR with preprocessing and post-processing for historical cookbooks.

This module provides improved OCR quality through:
1. Image preprocessing (denoise, deskew, contrast enhancement)
2. Post-processing text correction (fix common OCR errors)
3. Configurable Tesseract options for better accuracy
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract
from pytesseract import Output

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# Common OCR errors in historical cookbooks (pattern -> correction)
OCR_CORRECTIONS = {
    # Fractions commonly misread
    r'\b11([/ ]*)2\b': r'1½',  # "11/2" or "11 2" -> "1½"
    r'\b1([/ ]*)2\b': r'½',     # "1/2" or "1 2" -> "½"
    r'\b1([/ ]*)4\b': r'¼',     # "1/4" or "1 4" -> "¼"
    r'\b3([/ ]*)4\b': r'¾',     # "3/4" or "3 4" -> "¾"
    r'\b21([/ ]*)2\b': r'2½',   # "21/2" or "21 2" -> "2½"

    # Common word substitutions
    r'\bBeggs?\b': 'eggs',       # "Beggs" -> "eggs"
    r'\btegg\b': 'egg',          # "tegg" -> "egg"
    r'\bcgg\b': 'egg',           # "cgg" -> "egg"
    r'\blb\.?\b': 'lb',          # "lb." -> "lb"
    r'\bIbs\.?\b': 'lbs',        # "Ibs" -> "lbs"
    r'\bcup\.?\b': 'cup',        # "cup." -> "cup"
    r'\bcups\.?\b': 'cups',      # "cups." -> "cups"
    r'\bteaspoon\.?\b': 'teaspoon',
    r'\btablespoon\.?\b': 'tablespoon',

    # Common character substitutions
    r'\bII\b': '11',             # Roman numeral II -> 11
    r'\bIII\b': '111',           # Roman numeral III -> 111
    r'\bl\b': '1',               # Lowercase L -> 1 (in numeric context)
    r'\bO\b': '0',               # Uppercase O -> 0 (in numeric context)
}


def preprocess_image(
    image: Image.Image,
    denoise: bool = True,
    deskew: bool = True,
    enhance_contrast: bool = True,
    sharpen: bool = True,
) -> Image.Image:
    """Preprocess image for better OCR quality.

    Args:
        image: Input PIL Image
        denoise: Apply denoising filter
        deskew: Correct image skew/rotation
        enhance_contrast: Enhance contrast and brightness
        sharpen: Apply sharpening filter

    Returns:
        Preprocessed PIL Image
    """
    # Convert to grayscale
    img = ImageOps.grayscale(image)

    if HAS_CV2:
        # Use OpenCV for advanced preprocessing
        img_array = np.array(img)

        # Denoise
        if denoise:
            img_array = cv2.fastNlMeansDenoising(img_array, None, 10, 7, 21)

        # Deskew (correct rotation)
        if deskew:
            coords = np.column_stack(np.where(img_array > 0))
            if len(coords) > 0:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = 90 + angle
                elif angle > 45:
                    angle = angle - 90

                # Only correct if angle is significant (> 0.5 degrees)
                if abs(angle) > 0.5:
                    (h, w) = img_array.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    img_array = cv2.warpAffine(
                        img_array, M, (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE
                    )

        # Adaptive thresholding for better contrast
        if enhance_contrast:
            img_array = cv2.adaptiveThreshold(
                img_array, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                25, 15
            )

        # Morphological operations to clean up
        if sharpen:
            kernel = np.ones((2, 2), np.uint8)
            img_array = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)

        img = Image.fromarray(img_array)
    else:
        # Fallback to PIL-only preprocessing
        if enhance_contrast:
            img = ImageOps.autocontrast(img)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)

        if sharpen:
            img = img.filter(ImageFilter.SHARPEN)

        if denoise:
            img = img.filter(ImageFilter.MedianFilter(size=3))

    return img


def postprocess_text(text: str) -> str:
    """Apply post-processing corrections to OCR text.

    Args:
        text: Raw OCR text output

    Returns:
        Corrected text
    """
    # Apply each correction pattern
    for pattern, replacement in OCR_CORRECTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def postprocess_word(word: str) -> str:
    """Apply post-processing corrections to a single OCR word.

    Args:
        word: Single word from OCR

    Returns:
        Corrected word
    """
    original = word

    # Apply word-level corrections
    for pattern, replacement in OCR_CORRECTIONS.items():
        word = re.sub(pattern, replacement, word, flags=re.IGNORECASE)

    return word


def image_to_ocr_tokens(
    image_path: Path,
    preprocess: bool = True,
    postprocess: bool = True,
    tesseract_config: str = "--psm 6 --oem 3",
) -> Dict:
    """Extract OCR tokens with enhanced preprocessing and post-processing.

    Args:
        image_path: Path to image file
        preprocess: Enable image preprocessing
        postprocess: Enable text post-processing
        tesseract_config: Tesseract configuration string
            --psm options:
                6 = Assume a single uniform block of text (default, good for cookbook pages)
                3 = Fully automatic page segmentation (if 6 doesn't work well)
                11 = Sparse text. Find as much text as possible
            --oem options:
                3 = Default (Legacy + LSTM engines)
                1 = Neural nets LSTM only (faster, often better)
                2 = Legacy only

    Returns:
        Dictionary with keys: words, bboxes, confs, width, height
    """
    image = Image.open(image_path).convert("RGB")

    # Preprocess image
    if preprocess:
        processed_image = preprocess_image(
            image,
            denoise=True,
            deskew=True,
            enhance_contrast=True,
            sharpen=True
        )
    else:
        processed_image = image

    width, height = image.size  # Use original dimensions for bbox coordinates

    # Run Tesseract with custom config
    data = pytesseract.image_to_data(
        processed_image,
        config=tesseract_config,
        output_type=Output.DICT
    )

    words: List[str] = []
    bboxes: List[List[int]] = []
    confs: List[float] = []

    for text, conf, left, top, w, h in zip(
        data["text"],
        data["conf"],
        data["left"],
        data["top"],
        data["width"],
        data["height"]
    ):
        if not text or text.strip() == "" or int(conf) < 0:
            continue

        # Post-process word
        word = text.strip()
        if postprocess:
            word = postprocess_word(word)

        # Convert to normalized coordinates (0-1000 range)
        x0 = int(left / width * 1000)
        y0 = int(top / height * 1000)
        x1 = int((left + w) / width * 1000)
        y1 = int((top + h) / height * 1000)

        words.append(word)
        bboxes.append([x0, y0, x1, y1])
        confs.append(max(0.0, min(1.0, float(conf) / 100.0)))

    # Stable ordering top-to-bottom then left-to-right
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


# Tesseract configuration presets
TESSERACT_CONFIGS = {
    "default": "--psm 6 --oem 3",  # Single block, both engines
    "fast": "--psm 6 --oem 1",     # Single block, LSTM only (faster)
    "accurate": "--psm 6 --oem 3",  # Single block, both engines (high quality)
    "sparse": "--psm 11 --oem 3",  # Sparse text detection
    "cookbook": "--psm 6 --oem 1 -c preserve_interword_spaces=1",  # Optimized for cookbook layout
}
