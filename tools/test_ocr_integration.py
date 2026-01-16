#!/usr/bin/env python3
"""Test that OCR post-processing is working correctly."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.utils.ocr_enhanced import postprocess_word


def test_postprocessing():
    """Test common OCR error corrections."""
    test_cases = [
        # (input, expected_output)
        ("Beggs", "eggs"),
        ("11/2", "1½"),
        ("1/2", "½"),
        ("1/4", "¼"),
        ("3/4", "¾"),
        ("21/2", "2½"),
        ("Ibs", "lbs"),
        ("tegg", "egg"),
        ("cups", "cups"),  # Should not change
        ("tablespoon", "tablespoon"),  # Should not change
    ]

    print("Testing OCR Post-Processing")
    print("=" * 60)

    all_passed = True
    for input_word, expected in test_cases:
        result = postprocess_word(input_word)
        passed = result == expected
        status = "✓" if passed else "✗"

        print(f"{status} '{input_word}' → '{result}' (expected: '{expected}')")

        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


def test_integration():
    """Test that the integration works in the parsing pipeline."""
    from backend.app.utils.ocr_tesseract import image_to_ocr_tokens

    print("\nTesting Integration with OCR Module")
    print("=" * 60)

    # Check if postprocessing is available
    from backend.app.utils.ocr_tesseract import HAS_POSTPROCESSING

    if HAS_POSTPROCESSING:
        print("✓ OCR post-processing module loaded successfully")
    else:
        print("✗ OCR post-processing module not available")
        return 1

    # Test that function signature includes postprocessing parameter
    import inspect
    sig = inspect.signature(image_to_ocr_tokens)
    if 'apply_postprocessing' in sig.parameters:
        print("✓ image_to_ocr_tokens() supports apply_postprocessing parameter")
        default = sig.parameters['apply_postprocessing'].default
        if default is True:
            print(f"✓ Post-processing enabled by default: {default}")
        else:
            print(f"! Post-processing default value: {default}")
    else:
        print("✗ image_to_ocr_tokens() missing apply_postprocessing parameter")
        return 1

    print("=" * 60)
    print("✓ Integration tests passed!")
    return 0


if __name__ == "__main__":
    ret1 = test_postprocessing()
    ret2 = test_integration()

    sys.exit(max(ret1, ret2))
