# Curated Recipe Label Generation

## Overview

This system converts manually curated recipe JSON files from `/frontend/public/recipes/boston/` into token-level weak labels for LayoutLMv3 training. The curated recipes serve as high-quality ground truth that is aligned with OCR tokens using fuzzy matching.

## Implementation

### New Script: `scripts/generate_curated_weak_labels.py`

**Features:**
- Loads curated recipe JSON files with title, ingredients, and instructions
- Aligns each text field to OCR tokens using fuzzy matching
- Generates token-level labels (TITLE, INGREDIENT_LINE, INSTRUCTION_STEP)
- Reports match coverage and quality metrics
- Filters pages by minimum coverage threshold

**Algorithm:**
1. **Text Normalization**: Lowercase, strip punctuation, collapse whitespace
2. **Fuzzy Matching**: Uses rapidfuzz (fallback to token overlap if unavailable)
3. **Span Finding**: Searches OCR token windows for best match
4. **Label Assignment**: Assigns label IDs to matched token spans
5. **Coverage Tracking**: Reports % of recipe fields successfully matched

**Usage:**
```bash
python scripts/generate_curated_weak_labels.py \
  --curated-dir frontend/public/recipes/boston \
  --ocr-jsonl data/ocr/boston_pages.jsonl \
  --out data/labels/boston_curated_weak_labeled.jsonl \
  --match-threshold 0.5 \
  --min-coverage 0.05
```

**Parameters:**
- `--match-threshold`: Minimum similarity for fuzzy match (0-1, default: 0.7)
- `--min-coverage`: Minimum % of recipe fields matched to include page (default: 0.5)

### Updated: `scripts/build_boston_dataset.py`

**Multi-Source Label Merging:**

Priority hierarchy (highest to lowest):
1. **Gold labels** (`data/gold/boston_gold.jsonl`) - Hand-annotated ground truth
2. **Corrections** (`data/corrections/boston_corrections.jsonl`) - User fixes from Compare Mode
3. **Curated labels** (`data/labels/boston_curated_weak_labeled.jsonl`) - NEW: From curated recipes
4. **Weak labels** (`data/labels/boston_weak_labeled.jsonl`) - Automated weak labeling
5. **OCR only** - Unlabeled pages (all "O")

**New CLI Arguments:**
```bash
--curated-labels-jsonl    # Path to curated labels (default: data/labels/boston_curated_weak_labeled.jsonl)
--gold-labels-jsonl       # Path to gold labels (default: data/gold/boston_gold.jsonl)
--corrections-jsonl       # Path to corrections (default: data/corrections/boston_corrections.jsonl)
```

**Label Distribution Validation:**

The script now validates that `INSTRUCTION_STEP` count is at least 1/5 of `INGREDIENT_LINE`:

```python
if inst_ratio < 0.2:
    print("⚠️  WARNING: INSTRUCTION_STEP count is less than 1/5 of INGREDIENT_LINE")
    print("   Consider running: make generate-curated-labels")
```

Also reports top 10 pages with instruction tokens for debugging.

### Makefile Integration

**New Target: `make generate-curated-labels`**

Generates token-level labels from curated recipes:
```bash
make generate-curated-labels
```

**Updated Targets:**

- `make build-dataset` - Now runs `generate-curated-labels` automatically
- `make rebuild-data` - Full pipeline: render → curated labels → build dataset

```bash
# Full rebuild with curated labels
make rebuild-data

# Just regenerate curated labels
make generate-curated-labels

# Build dataset (includes curated label generation)
make build-dataset
```

## Results

### Current Status (Initial Run)

**With threshold=0.5, min_coverage=0.05:**

```
Total curated recipes: 30
Pages with OCR data: 30
Pages written: 18
Pages skipped (low coverage): 12

Coverage statistics:
  Average: 11.5%
  Min: 0.0%
  Max: 41.7%

Total label distribution:
  TITLE               :      7 ( 0.16%)
  INGREDIENT_LINE     :    127 ( 2.91%)
  INSTRUCTION_STEP    :     22 ( 0.50%)
  O                   :   4207 (96.42%)
```

**Successfully matched recipes:**
- graham-muffins: 35.7% coverage (20 INGREDIENT_LINE tokens)
- oatmeal-cookies: 41.7% coverage (18 INGREDIENT_LINE tokens)
- Plus 16 other recipes with 5-35% coverage

### Why Coverage is Low

**Mismatch Challenges:**
1. **Modernized Text**: Curated recipes use modern measurements ("1 cup", "1/2 tsp") vs 1918 originals
2. **Formatting Differences**: Original text may have different spacing, abbreviations
3. **OCR Errors**: Tesseract may misread old typography
4. **Page Content**: Some pages have headers, footers, other text that dilutes recipe content

**Example:**
- Curated: "1 cup flour"
- 1918 OCR: "One cup of flour" or "1 c. flour"

### Improvement Strategies

**1. Better Text Normalization:**
- Expand abbreviations (c. → cup, tbsp. → tablespoon)
- Convert spelled numbers (One → 1, two → 2)
- Handle fraction formats (1-1/2 vs 1½)

**2. Smarter Matching:**
- Use edit distance for OCR error tolerance
- Window-based partial matching
- Skip common words (the, of, a)

**3. Curated Recipe Quality:**
- Verify source page numbers are correct
- Add "original_text" field with 1918 wording
- Include alternate phrasings

**4. Hybrid Approach:**
- Use curated labels for TITLE (high confidence, short)
- Use weak labels for instructions (longer, more variation)
- Combine both sources with weighted confidence

## Files Modified

1. **scripts/generate_curated_weak_labels.py** (NEW, 401 lines)
   - Fuzzy matching algorithm
   - Token alignment logic
   - Coverage tracking and reporting

2. **scripts/build_boston_dataset.py** (+162 lines, 187→349 total)
   - Multi-source label merging
   - Validation warnings for instruction ratio
   - Source tracking (gold/corrections/curated/weak/ocr_only)

3. **Makefile** (+3 targets)
   - `generate-curated-labels`
   - Updated `build-dataset` (depends on curated labels)
   - Updated `rebuild-data` (includes curated labels)

## Expected Impact

**Before** (weak labels only):
```
INGREDIENT_LINE:    8070 tokens
INSTRUCTION_STEP:    837 tokens
Ratio: 0.104 (10.4%)
```

**After** (weak + curated):
```
INGREDIENT_LINE:    8197 tokens  (+127)
INSTRUCTION_STEP:    859 tokens  (+22)
Ratio: 0.105 (10.5%)
```

**Marginal improvement** due to low coverage. To get significant gains:
- Increase curated recipe count (30 → 100+)
- Improve matching algorithm (better fuzzy logic)
- Add original 1918 text to curated files

## Future Enhancements

### Short Term
1. **rapidfuzz integration**: Install rapidfuzz for better fuzzy matching
2. **Text expansion**: Convert abbreviations and numbers
3. **Logging**: Add --verbose flag for debugging matches

### Medium Term
1. **Original text field**: Add `original_ingredients`/`original_instructions` to curated JSON
2. **Confidence scores**: Weight labels by match similarity
3. **Visual QA**: Generate overlay images showing matches

### Long Term
1. **Active learning loop**: Export corrections from Compare Mode
2. **Human verification**: Label Studio integration for gold labels
3. **Cross-validation**: Use curated labels to validate weak labeler quality

## Usage Recommendations

**For immediate training:**
```bash
# Generate curated labels with relaxed thresholds
python scripts/generate_curated_weak_labels.py \
  --match-threshold 0.5 \
  --min-coverage 0.05

# Rebuild dataset with multi-source merging
python scripts/build_boston_dataset.py
```

**For better quality:**
1. Install rapidfuzz: `pip install rapidfuzz`
2. Add original text to 10-20 curated recipes
3. Increase match threshold to 0.7
4. Verify output with audit script

## Acceptance Criteria Status

✅ **A) Script Implementation**
- [x] `scripts/generate_curated_weak_labels.py` created
- [x] Loads curated recipes from frontend/public/recipes/boston
- [x] Aligns with OCR tokens using fuzzy matching
- [x] Outputs JSONL with labels, source="curated", match coverage
- [x] Reports label distribution and coverage stats

✅ **B) Dataset Merge Logic**
- [x] `build_boston_dataset.py` merges 4 label sources
- [x] Priority: gold > corrections > curated > weak
- [x] Tracks label source per page
- [x] Reports merge statistics

✅ **C) Validation Improvements**
- [x] Warns if INSTRUCTION_STEP < INGREDIENT_LINE/5
- [x] Reports top pages with instruction tokens
- [x] Shows label distribution after merge

✅ **D) Makefile Integration**
- [x] `make generate-curated-labels` target added
- [x] `make rebuild-dataset` calls generation automatically
- [x] `make rebuild-data` includes full pipeline

## Notes

**Current limitation**: Low match coverage (11.5% average) due to text differences between modernized curated recipes and 1918 original text. This still provides value for the recipes that DO match well (35-40% coverage), especially for ingredient labels.

**Recommended next step**: Add "original_text" fields to curated recipes to bridge the gap between modern and historical text, which should dramatically improve matching.
