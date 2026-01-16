"""Enhanced OCR script with preprocessing and post-processing.

This is an improved version of ocr_pages.py that uses the enhanced OCR module
with better preprocessing, post-processing, and Tesseract configurations.

Example:
python data/scripts/ocr_pages_enhanced.py \
  --images_dir data/pages/boston \
  --out data/ocr/boston_pages_enhanced.jsonl \
  --preset cookbook \
  --start_page 69 \
  --max_pages 50
"""

import argparse
import json
import pathlib
import re
import sys
import time
from typing import List

from PIL import Image
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from backend.app.utils.ocr_enhanced import (
    image_to_ocr_tokens,
    preprocess_image,
    TESSERACT_CONFIGS,
)


def infer_page_num(path: pathlib.Path) -> int:
    """Extract page number from filename."""
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def process_images(
    images_dir: pathlib.Path,
    out_path: pathlib.Path,
    preset: str,
    start_page: int,
    max_pages: int | None,
    save_preprocessed: bool,
) -> None:
    """Process images with enhanced OCR.

    Args:
        images_dir: Directory containing page images
        out_path: Output JSONL path
        preset: Tesseract config preset name
        start_page: Skip pages before this number
        max_pages: Maximum number of pages to process
        save_preprocessed: Save preprocessed images for debugging
    """
    images = sorted(images_dir.glob("*.png"))

    # Filter by start_page
    if start_page > 0:
        images = [img for img in images if infer_page_num(img) >= start_page]

    if max_pages:
        images = images[:max_pages]

    if not images:
        raise FileNotFoundError(f"No PNG images found in {images_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Create preprocessed directory if requested
    preprocessed_dir = None
    if save_preprocessed:
        preprocessed_dir = out_path.parent / "preprocessed"
        preprocessed_dir.mkdir(exist_ok=True)

    # Get Tesseract config
    tesseract_config = TESSERACT_CONFIGS.get(preset, TESSERACT_CONFIGS["default"])

    start_time = time.time()
    total_tokens = 0

    print(f"Processing {len(images)} pages with preset '{preset}'")
    print(f"Tesseract config: {tesseract_config}")
    print(f"Starting from page {start_page if start_page > 0 else 'first'}")

    with out_path.open("w", encoding="utf-8") as f:
        for image_path in tqdm(images, desc="OCR pages"):
            page_num = infer_page_num(image_path)

            # Save preprocessed image if requested
            if save_preprocessed and preprocessed_dir:
                image = Image.open(image_path).convert("RGB")
                processed = preprocess_image(image)
                preprocessed_path = preprocessed_dir / f"page_{page_num:04d}.png"
                processed.save(preprocessed_path)

            # Run OCR with enhanced preprocessing
            result = image_to_ocr_tokens(
                image_path,
                preprocess=True,
                postprocess=True,
                tesseract_config=tesseract_config,
            )

            page = {
                "book": "Boston Cooking-School Cook Book",
                "year": 1918,
                "page_num": page_num,
                "image_path": str(image_path),
                "width": result["width"],
                "height": result["height"],
                "words": result["words"],
                "bboxes": result["bboxes"],
                "confidences": result["confs"],
            }
            f.write(json.dumps(page) + "\n")
            total_tokens += len(result["words"])

    elapsed = time.time() - start_time
    avg_tokens = total_tokens / len(images) if images else 0

    print(f"\n✓ Processed {len(images)} pages -> {out_path}")
    print(f"  Average tokens/page: {avg_tokens:.1f}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Runtime: {elapsed:.1f}s ({elapsed/len(images):.2f}s/page)")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhanced OCR with preprocessing and post-processing"
    )
    parser.add_argument(
        "--images_dir",
        required=True,
        help="Directory of page PNGs"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSONL path"
    )
    parser.add_argument(
        "--preset",
        choices=list(TESSERACT_CONFIGS.keys()),
        default="cookbook",
        help="Tesseract configuration preset (default: cookbook)"
    )
    parser.add_argument(
        "--start_page",
        type=int,
        default=69,
        help="Skip pages before this number (default: 69)"
    )
    parser.add_argument(
        "--max_pages",
        type=int,
        help="Process at most this many pages"
    )
    parser.add_argument(
        "--save_preprocessed",
        action="store_true",
        help="Save preprocessed images for debugging"
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    process_images(
        images_dir=pathlib.Path(args.images_dir),
        out_path=pathlib.Path(args.out),
        preset=args.preset,
        start_page=args.start_page,
        max_pages=args.max_pages,
        save_preprocessed=args.save_preprocessed,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
