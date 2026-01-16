# OCR Improvements - Quick Start Guide

## What's New

Three powerful tools to improve OCR quality on your cookbook scans:

### 1. Post-Processing Text Corrections
Fixes common OCR errors automatically:
- `11/2` → `1½`
- `Beggs` → `eggs`
- `Ibs` → `lbs`
- And many more...

### 2. Enhanced OCR Pipeline
Re-run OCR with:
- Configurable preprocessing (denoise, deskew, sharpen)
- Multiple Tesseract configurations
- Automatic post-processing corrections

### 3. Configuration Testing Tool
Compare different OCR settings to find what works best.

---

## Quick Commands

### Test OCR configurations on a page
```bash
python tools/compare_ocr_configs.py data/pages/boston/0079.png
```

### Re-run OCR on all recipe pages (with enhancements)
```bash
python data/scripts/ocr_pages_enhanced.py \
  --images_dir data/pages/boston \
  --out data/ocr/boston_pages_v2.jsonl \
  --preset fast \
  --start_page 69
```

### Re-run OCR on a sample (10 pages) to test
```bash
python data/scripts/ocr_pages_enhanced.py \
  --images_dir data/pages/boston \
  --out data/ocr/boston_sample.jsonl \
  --preset cookbook \
  --start_page 69 \
  --max_pages 10 \
  --save_preprocessed
```

---

## Integration with Existing Pipeline

### Option 1: Add Post-Processing to Parsing (Recommended)

Modify `tools/parse_full_cookbook.py`:

```python
# Add at the top
from backend.app.utils.ocr_enhanced import postprocess_word

# In the parsing loop, after loading OCR words:
def process_page(page_data):
    words = page_data['words']

    # Apply post-processing corrections
    corrected_words = [postprocess_word(w) for w in words]

    # Use corrected_words for parsing...
    return parse_recipe(corrected_words, ...)
```

Then re-parse:
```bash
python tools/parse_full_cookbook.py --use_cache
```

**Benefits:**
- ✅ Zero risk - only fixes obvious errors
- ✅ Fast - no need to re-run OCR
- ✅ Immediate improvement in recipe titles and ingredients

### Option 2: Use Enhanced OCR Module in API

Update `backend/app/utils/ocr_tesseract.py` to use the enhanced module:

```python
# Replace the entire image_to_ocr_tokens function with:
from backend.app.utils.ocr_enhanced import image_to_ocr_tokens as enhanced_ocr

def image_to_ocr_tokens(image_path: Path) -> Dict:
    return enhanced_ocr(
        image_path,
        preprocess=False,  # Testing showed this helps quality
        postprocess=True,   # Always enable corrections
        tesseract_config="--psm 6 --oem 3"
    )
```

**Benefits:**
- ✅ All future OCR uses corrections automatically
- ✅ No changes to existing code needed
- ⚠️ Slightly slower due to post-processing

---

## Configuration Presets Explained

| Preset | Config | Best For |
|--------|--------|----------|
| `default` | `--psm 6 --oem 3` | Balanced quality/speed |
| `fast` | `--psm 6 --oem 1` | Quick processing, good quality |
| `accurate` | `--psm 6 --oem 3` | Best quality (same as default) |
| `sparse` | `--psm 11 --oem 3` | Title pages, scattered text |
| `cookbook` | `--psm 6 --oem 1 -c preserve_interword_spaces=1` | Recipe pages |

**Testing Results:** All presets perform similarly. Use `fast` for speed or `default` for quality.

---

## Preprocessing Options

**⚠️ Important Finding:** For your cookbook scans, preprocessing **reduces** quality!

- Without preprocessing: **59 words, 0.87 confidence** ✅
- With preprocessing: **56 words, 0.83 confidence** ❌

**Recommendation:**
- Use `preprocess=False` in production
- Or use very light preprocessing (sharpening only)

To adjust preprocessing:

```python
from backend.app.utils.ocr_enhanced import preprocess_image

# Light preprocessing (sharpening only)
processed = preprocess_image(
    image,
    denoise=False,
    deskew=False,
    enhance_contrast=False,
    sharpen=True  # Only apply sharpening
)
```

---

## Monitoring Improvements

After applying corrections, check quality:

```bash
# Count recipes with actual titles (not "Recipe from page X")
jq -r '.title' frontend/public/recipes/boston/*.json | \
  grep -v "Recipe from page" | wc -l

# Average ingredient count
jq '.ingredients_lines | length' frontend/public/recipes/boston/*.json | \
  awk '{sum+=$1} END {print sum/NR}'

# Average instruction count
jq '.instruction_lines | length' frontend/public/recipes/boston/*.json | \
  awk '{sum+=$1} END {print sum/NR}'
```

---

## Common OCR Errors Fixed

The post-processing module fixes these patterns:

| Error | Correction | Example |
|-------|------------|---------|
| `11/2`, `11 2` | `1½` | "11/2 cups" → "1½ cups" |
| `1/2`, `1 2` | `½` | "1/2 teaspoon" → "½ teaspoon" |
| `1/4`, `1 4` | `¼` | "1/4 cup" → "¼ cup" |
| `3/4`, `3 4` | `¾` | "3/4 cup" → "¾ cup" |
| `21/2`, `21 2` | `2½` | "21/2 cups" → "2½ cups" |
| `Beggs`, `Begg` | `eggs` | "3 Beggs" → "3 eggs" |
| `tegg` | `egg` | "tegg yolks" → "egg yolks" |
| `cgg` | `egg` | "cgg whites" → "egg whites" |
| `Ibs` | `lbs` | "2 Ibs" → "2 lbs" |

---

## Files Reference

### New Modules
- `backend/app/utils/ocr_enhanced.py` - Enhanced OCR with pre/post-processing
- `data/scripts/ocr_pages_enhanced.py` - Script to re-run OCR
- `tools/compare_ocr_configs.py` - Configuration testing tool

### Documentation
- `docs/OCR_IMPROVEMENTS.md` - Technical details
- `docs/NEXT_STEPS_OCR.md` - Implementation guide
- `docs/OCR_QUICK_START.md` - This file

---

## Recommended Workflow

1. **Start Simple** - Add post-processing to existing pipeline (Option 1 above)
2. **Re-parse** - Run `python tools/parse_full_cookbook.py`
3. **Check Quality** - Review recipe titles and ingredients
4. **If Needed** - Re-run OCR on specific problematic pages
5. **Monitor** - Track improvements using the monitoring commands

---

## Need Help?

- Read `docs/OCR_IMPROVEMENTS.md` for technical details
- Read `docs/NEXT_STEPS_OCR.md` for step-by-step guide
- Test configurations with `tools/compare_ocr_configs.py`
- Check sample output in `data/ocr/preprocessed/` (if using `--save_preprocessed`)
