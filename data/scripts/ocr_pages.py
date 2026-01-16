"""Run OCR over rendered page images and emit JSONL.

Example:
python data/scripts/ocr_pages.py \
  --images_dir data/pages/boston \
  --out data/ocr/boston_pages.jsonl \
  --ocr_backend tesseract \
  --lang eng \
  --max_pages 50
"""

import argparse
import json
import pathlib
import re
import sys
import time
from typing import List, Tuple

from PIL import Image, ImageOps
from tqdm import tqdm

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import pytesseract
    from pytesseract import Output as TessOutput
except ImportError:
    pytesseract = None  # type: ignore
    TessOutput = None  # type: ignore

# TODO: optionally support LayoutLMv3Processor OCR if needed.

BBox = Tuple[int, int, int, int]


def preprocess_image(image: Image.Image, enable: bool) -> Image.Image:
    if not enable:
        return image

    gray = ImageOps.grayscale(image)
    if cv2 is None:
        # Fallback simple contrast + threshold
        contrasted = ImageOps.autocontrast(gray)
        return contrasted.point(lambda p: 255 if p > 180 else 0)

    if np is None:
        contrasted = ImageOps.autocontrast(gray)
        return contrasted

    img = cv2.cvtColor(np.array(gray), cv2.COLOR_GRAY2BGR)  # type: ignore[name-defined]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 15)
    return Image.fromarray(img)


def infer_page_num(path: pathlib.Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def run_tesseract(image: Image.Image, lang: str) -> Tuple[List[str], List[BBox], List[int]]:
    if pytesseract is None or TessOutput is None:
        raise ImportError("pytesseract is not installed; install or choose a different backend")

    data = pytesseract.image_to_data(image, lang=lang, output_type=TessOutput.DICT)
    words: List[str] = []
    bboxes: List[BBox] = []
    confidences: List[int] = []

    n_boxes = len(data["text"])
    for i in range(n_boxes):
        text = data["text"][i].strip()
        if not text:
            continue
        conf = int(float(data.get("conf", ["0"][0])[i])) if data.get("conf") else 0
        x, y, w, h = (
            int(data["left"][i]),
            int(data["top"][i]),
            int(data["width"][i]),
            int(data["height"][i]),
        )
        words.append(text)
        bboxes.append((x, y, x + w, y + h))
        confidences.append(conf)
    return words, bboxes, confidences


def run_layoutlmv3(image: Image.Image, lang: str) -> Tuple[List[str], List[BBox], List[int]]:  # pragma: no cover - optional path
    try:
        from transformers import LayoutLMv3Processor
    except Exception as exc:  # noqa: BLE001
        raise ImportError("transformers/LayoutLMv3Processor not available") from exc

    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
    # apply_ocr=True uses pytesseract internally if available.
    encoding = processor(image, return_offsets_mapping=True, return_tensors="pt", lang=lang)
    words = encoding.words
    boxes = encoding.bboxes
    if words is None or boxes is None:
        return [], [], []
    filtered_words: List[str] = []
    bboxes: List[BBox] = []
    for word, box in zip(words, boxes):
        if not word or word.isspace():
            continue
        x0, y0, x1, y1 = box.tolist()
        filtered_words.append(word)
        bboxes.append((int(x0), int(y0), int(x1), int(y1)))
    return filtered_words, bboxes, []


def process_images(
    images_dir: pathlib.Path,
    out_path: pathlib.Path,
    ocr_backend: str,
    lang: str,
    max_pages: int | None,
    preprocess: bool,
    include_text: bool,
) -> None:
    images = sorted(images_dir.glob("*.png"))
    if max_pages:
        images = images[:max_pages]

    if not images:
        raise FileNotFoundError(f"No PNG images found in {images_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    total_tokens = 0

    with out_path.open("w", encoding="utf-8") as f:
        for image_path in tqdm(images, desc="OCR pages"):
            page_num = infer_page_num(image_path)
            image = Image.open(image_path)
            width, height = image.size
            processed_image = preprocess_image(image, preprocess)
            full_text = None

            if ocr_backend == "tesseract":
                words, bboxes, confidences = run_tesseract(processed_image, lang)
                if include_text and pytesseract is not None:
                    full_text = pytesseract.image_to_string(processed_image, lang=lang)
            elif ocr_backend == "layoutlmv3":
                words, bboxes, confidences = run_layoutlmv3(processed_image, lang)
            else:
                raise ValueError("Unsupported ocr_backend. Choose 'tesseract' or 'layoutlmv3'.")

            page = {
                "book": "Boston Cooking-School Cook Book",
                "year": 1918,
                "page_num": page_num,
                "image_path": str(image_path),
                "width": width,
                "height": height,
                "words": words,
                "bboxes": bboxes,
                "confidences": confidences,
                "text": full_text,
            }
            f.write(json.dumps(page) + "\n")
            total_tokens += len(words)

    elapsed = time.time() - start_time
    avg_tokens = total_tokens / len(images) if images else 0
    print(
        f"Processed {len(images)} pages -> {out_path}\n"
        f"Average tokens/page: {avg_tokens:.1f}\n"
        f"Runtime: {elapsed:.1f}s"
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR rendered pages to JSONL")
    parser.add_argument("--images_dir", required=True, help="Directory of page PNGs")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--ocr_backend", choices=["tesseract", "layoutlmv3"], default="tesseract")
    parser.add_argument("--lang", default="eng", help="OCR language (default eng)")
    parser.add_argument("--max_pages", type=int, help="Process at most this many pages")
    parser.add_argument("--preprocess", action="store_true", help="Enable simple preprocessing before OCR")
    parser.add_argument("--include_text", action="store_true", help="Also store full OCR text per page")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    process_images(
        images_dir=pathlib.Path(args.images_dir),
        out_path=pathlib.Path(args.out),
        ocr_backend=args.ocr_backend,
        lang=args.lang,
        max_pages=args.max_pages,
        preprocess=args.preprocess,
        include_text=args.include_text,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
