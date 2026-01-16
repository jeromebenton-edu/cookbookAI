# OCR Quality Improvements

This document explains the OCR enhancements added to improve text extraction quality from historical cookbook scans.

## Overview

Historical cookbook scans (like the 1918 Fanny Farmer cookbook) present unique OCR challenges:
- **Aging paper**: Yellowed, stained, or damaged pages
- **Historical typography**: Old-fashioned fonts and ligatures
- **Fractions and special characters**: ½, ¼, ¾ often misread as "11/2", "1/4", etc.
- **Print quality**: Blurred or faded text from scanning
- **Layout complexity**: Multiple columns, varied text sizes

## Solution: Three-Pronged Approach

### 1. Image Preprocessing

Located in `backend/app/utils/ocr_enhanced.py: preprocess_image()`

**Enhancements:**
- **Denoising**: Removes scan artifacts and paper texture noise
- **Deskewing**: Corrects page rotation to improve character recognition
- **Contrast enhancement**: Adaptive thresholding for better text/background separation
- **Sharpening**: Enhances edge definition of characters

**Implementation:**
- Uses OpenCV when available for advanced processing
- Falls back to PIL filters if OpenCV unavailable
- All preprocessing is optional and can be toggled

### 2. Post-Processing Text Correction

Located in `backend/app/utils/ocr_enhanced.py: OCR_CORRECTIONS`

**Common Error Patterns:**

| OCR Error | Correction | Reason |
|-----------|------------|--------|
| `11/2` or `11 2` | `1½` | Fraction misread |
| `1/2` or `1 2` | `½` | Fraction misread |
| `Beggs` | `eggs` | 'B' mistaken for '3' |
| `Ibs` | `lbs` | 'I' mistaken for 'l' |
| `tegg` | `egg` | 't' prefix error |

**Features:**
- Pattern-based regex replacements
- Case-insensitive matching
- Context-aware corrections (only in appropriate contexts)

### 3. Tesseract Configuration Optimization

Located in `backend/app/utils/ocr_enhanced.py: TESSERACT_CONFIGS`

**Available Presets:**

- **`default`**: `--psm 6 --oem 3` - Balanced quality/speed
- **`fast`**: `--psm 6 --oem 1` - LSTM-only for faster processing
- **`accurate`**: `--psm 6 --oem 3` - Both engines for best quality
- **`sparse`**: `--psm 11 --oem 3` - For pages with scattered text
- **`cookbook`**: `--psm 6 --oem 1 -c preserve_interword_spaces=1` - Optimized for recipe layout

**PSM (Page Segmentation Mode) Options:**
- `6`: Assume single uniform block (best for cookbook pages)
- `11`: Sparse text detection (for title pages, indices)
- `3`: Fully automatic (when layout is unpredictable)

**OEM (OCR Engine Mode) Options:**
- `1`: Neural network LSTM only (faster, often better for printed text)
- `3`: Legacy + LSTM (more robust for challenging scans)

## Usage

### Enhanced OCR Script

```bash
# Process pages with enhanced OCR
python data/scripts/ocr_pages_enhanced.py \
  --images_dir data/pages/boston \
  --out data/ocr/boston_pages_enhanced.jsonl \
  --preset cookbook \
  --start_page 69 \
  --max_pages 50
```

**Options:**
- `--preset`: Choose from `default`, `fast`, `accurate`, `sparse`, `cookbook`
- `--start_page`: Skip pages before this number (default: 69 to skip front matter)
- `--max_pages`: Limit number of pages to process
- `--save_preprocessed`: Save preprocessed images for debugging

### Compare Configurations

Test different configs to find the best for your use case:

```bash
python tools/compare_ocr_configs.py data/pages/boston/0079.png
```

This will:
1. Test all 5 configuration presets
2. Run with and without preprocessing
3. Show word count and average confidence
4. Display first 20 words from each config

## Integration with Parsing Pipeline

The enhanced OCR can be integrated into the recipe parsing workflow:

1. **Update `tools/parse_full_cookbook.py`** to use `ocr_enhanced` instead of `ocr_tesseract`
2. Enable preprocessing for all pages >= 69 (recipe pages)
3. Apply post-processing corrections before token classification
4. Use the `cookbook` preset for optimal results

## Testing & Validation

### Before & After Comparison

**Original OCR errors on page 79:**
- "13g cups" → Should be "1½ cups"
- "34 cup" → Should be "¾ cup"
- "43g teaspoon" → Should be "½ teaspoon"
- "Beggs" → Should be "eggs"

**With enhanced OCR:**
- Fractions correctly extracted (if printed clearly)
- "eggs" correctly recognized
- Better confidence scores overall

### Metrics to Track

- **Word count**: More words = better detection
- **Average confidence**: Higher = more accurate
- **Manual validation**: Spot-check 10-20 pages
- **Recipe extraction quality**: Do we get better titles/ingredients?

## Next Steps

1. **Re-run OCR** on all recipe pages (69-623) with enhanced pipeline
2. **Re-parse recipes** using improved OCR data
3. **Validate improvements** by checking recipe titles and ingredient lists
4. **Fine-tune corrections** based on common remaining errors

## Files Modified/Created

### New Files:
- `backend/app/utils/ocr_enhanced.py` - Enhanced OCR module
- `data/scripts/ocr_pages_enhanced.py` - Enhanced OCR script
- `tools/compare_ocr_configs.py` - Configuration comparison tool
- `docs/OCR_IMPROVEMENTS.md` - This documentation

### Integration Points:
- Replace imports in `tools/parse_full_cookbook.py`
- Update API endpoints to use enhanced OCR
- Frontend demo pages already work (using pre-generated OCR data)
