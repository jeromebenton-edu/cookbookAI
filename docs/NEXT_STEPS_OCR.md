# Next Steps: Implementing OCR Improvements

## What We've Built

Three new tools for improving OCR quality:

1. **`backend/app/utils/ocr_enhanced.py`** - Enhanced OCR module with preprocessing and post-processing
2. **`data/scripts/ocr_pages_enhanced.py`** - Script to re-run OCR with enhancements
3. **`tools/compare_ocr_configs.py`** - Tool to test different configurations

## Key Findings from Testing

Testing on page 79 (Griddle Cakes) revealed:

**Preprocessing Results:**
- ❌ Aggressive preprocessing (denoise + deskew + contrast) **reduced** quality
  - 56 words with preprocessing vs 59 without
  - 0.83 confidence with preprocessing vs 0.87 without
- ✅ These scans are already high quality - preprocessing removes useful information

**Best Configuration:**
- **No preprocessing** or very light preprocessing
- **PSM 6** (single block) works well
- **OEM 3** (Legacy + LSTM) has slightly better results than OEM 1
- **Post-processing** is still valuable for fixing common errors

## Recommended Approach

### Option 1: Quick Fix - Post-Processing Only

Apply post-processing corrections to existing OCR data without re-running OCR:

```python
from backend.app.utils.ocr_enhanced import postprocess_word

# In your parsing pipeline, before feeding to model:
for word in ocr_words:
    corrected = postprocess_word(word)
    # Use corrected word
```

**Pros:**
- Fast - no need to re-run OCR
- Fixes common errors like "Beggs" → "eggs", fractions
- Low risk

**Cons:**
- Won't improve underlying OCR quality
- Can't fix missing/wrong characters

### Option 2: Re-run OCR with Light Preprocessing

Re-run OCR on all recipe pages with optimized settings:

```bash
python data/scripts/ocr_pages_enhanced.py \
  --images_dir data/pages/boston \
  --out data/ocr/boston_pages_v2.jsonl \
  --preset fast \
  --start_page 69
```

Then modify the `preprocess_image()` function to use **lighter** preprocessing:

```python
# In ocr_enhanced.py, modify preprocess_image defaults:
def preprocess_image(
    image: Image.Image,
    denoise: bool = False,      # Changed from True
    deskew: bool = False,       # Changed from True
    enhance_contrast: bool = False,  # Changed from True
    sharpen: bool = True,       # Keep this
) -> Image.Image:
```

**Pros:**
- Can experiment with different Tesseract configs
- Sharpening might help slightly
- Post-processing corrections applied

**Cons:**
- Takes time to re-run OCR (555 pages × ~2s/page = ~20 minutes)
- May not improve quality much over existing OCR

### Option 3: Hybrid - Selective Enhancement

Apply enhancements only where needed:

1. **Identify problematic pages** (low confidence, missing titles)
2. **Re-run OCR** on just those pages with different configs
3. **Apply post-processing** to all pages
4. **Merge results** back into main dataset

**Pros:**
- Targets effort where needed
- Lower risk of making things worse
- Faster than full re-run

**Cons:**
- More complex implementation
- Requires quality analysis first

## Immediate Actions You Can Take

### 1. Test Post-Processing on Current Data

Update `tools/parse_full_cookbook.py` to use post-processing:

```python
# At the top
from backend.app.utils.ocr_enhanced import postprocess_word

# In the parsing loop, after loading OCR data:
for i, word in enumerate(page_words):
    page_words[i] = postprocess_word(word)
```

Then re-parse recipes:

```bash
python tools/parse_full_cookbook.py --use_cache
```

This will:
- Fix "Beggs" → "eggs"
- Fix fractions like "11/2" → "1½"
- Clean up other common OCR errors
- **Zero risk** - only affects text cleaning

### 2. Compare Current OCR vs Enhanced OCR

Run enhanced OCR on a sample (say, 10 pages) and compare:

```bash
# Run enhanced OCR on pages 69-78
python data/scripts/ocr_pages_enhanced.py \
  --images_dir data/pages/boston \
  --out data/ocr/boston_sample_enhanced.jsonl \
  --preset fast \
  --start_page 69 \
  --max_pages 10 \
  --save_preprocessed
```

Then manually review:
- Check `data/ocr/preprocessed/` folder to see preprocessing effects
- Compare word lists between original and enhanced
- Count how many titles are now extracted vs before

### 3. Update Specific Problem Pages

If certain pages have persistent OCR issues:

```bash
# Test different configs on a problematic page
python tools/compare_ocr_configs.py data/pages/boston/XXXX.png

# Pick the best config and re-run just that page
python -c "
from pathlib import Path
from backend.app.utils.ocr_enhanced import image_to_ocr_tokens
import json

result = image_to_ocr_tokens(
    Path('data/pages/boston/XXXX.png'),
    preprocess=False,  # Based on testing results
    postprocess=True,
    tesseract_config='--psm 6 --oem 3'
)
print(json.dumps(result, indent=2))
" > data/ocr/page_XXXX_fixed.json
```

## Monitoring Quality Improvements

After applying any changes, check:

1. **Recipe title extraction rate**:
   ```bash
   jq -r '.title' frontend/public/recipes/boston/*.json | grep -v "Recipe from page" | wc -l
   ```

2. **Average ingredient count** (should increase if OCR improves):
   ```bash
   jq '.ingredients_lines | length' frontend/public/recipes/boston/*.json | awk '{sum+=$1} END {print sum/NR}'
   ```

3. **Manual spot-checks** - pick 10 random recipes and verify quality

## What NOT to Do

❌ **Don't** apply aggressive preprocessing to these scans - testing shows it hurts quality
❌ **Don't** re-run all OCR without testing on samples first
❌ **Don't** assume more preprocessing = better results
❌ **Don't** change multiple variables at once - test one thing at a time

## Summary Recommendation

**Start with Option 1** (post-processing only):
1. Add `postprocess_word()` to your parsing pipeline
2. Re-parse all recipes with corrected OCR words
3. Check if recipe titles and ingredients improve
4. If satisfied, stop here

**If you need more improvement:**
1. Identify the 10-20 worst pages (missing titles, garbled text)
2. Test different Tesseract configs on those pages specifically
3. Re-run OCR only on those problem pages
4. Merge back into main dataset

This balanced approach gives you immediate benefits (post-processing) while allowing targeted improvements where needed, without the risk of degrading overall quality with aggressive preprocessing.
