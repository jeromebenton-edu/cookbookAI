"""Compare different OCR configurations on sample pages.

This script tests multiple Tesseract configurations and preprocessing options
to find the best settings for cookbook OCR quality.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.utils.ocr_enhanced import image_to_ocr_tokens, TESSERACT_CONFIGS
from PIL import Image


def compare_configs(image_path: Path) -> None:
    """Compare different Tesseract configurations on a single page."""
    print(f"Testing OCR configurations on: {image_path.name}\n")
    print("=" * 80)

    results = {}

    # Test each preset
    for preset_name, config in TESSERACT_CONFIGS.items():
        print(f"\n{preset_name.upper()} (config: {config})")
        print("-" * 80)

        # Test with preprocessing
        result_pre = image_to_ocr_tokens(
            image_path,
            preprocess=True,
            postprocess=True,
            tesseract_config=config
        )

        # Test without preprocessing
        result_no_pre = image_to_ocr_tokens(
            image_path,
            preprocess=False,
            postprocess=True,
            tesseract_config=config
        )

        results[preset_name] = {
            "with_preprocessing": result_pre,
            "without_preprocessing": result_no_pre
        }

        print(f"  With preprocessing:    {len(result_pre['words']):3d} words, "
              f"avg conf: {sum(result_pre['confs'])/len(result_pre['confs']):.2f}")
        print(f"  Without preprocessing: {len(result_no_pre['words']):3d} words, "
              f"avg conf: {sum(result_no_pre['confs'])/len(result_no_pre['confs']):.2f}")

        # Show first 20 words with preprocessing
        print(f"\n  First 20 words (with preprocessing):")
        for i, word in enumerate(result_pre['words'][:20], 1):
            print(f"    {i:2d}. {word}")

    print("\n" + "=" * 80)
    print("\nSUMMARY")
    print("-" * 80)
    for preset_name, data in results.items():
        pre = data["with_preprocessing"]
        no_pre = data["without_preprocessing"]
        print(f"{preset_name:12s}: "
              f"pre={len(pre['words']):3d} words (conf={sum(pre['confs'])/len(pre['confs']):.2f}), "
              f"no-pre={len(no_pre['words']):3d} words (conf={sum(no_pre['confs'])/len(no_pre['confs']):.2f})")


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/compare_ocr_configs.py <image_path>")
        print("\nExample:")
        print("  python tools/compare_ocr_configs.py data/pages/boston/0079.png")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    compare_configs(image_path)


if __name__ == "__main__":
    main()
