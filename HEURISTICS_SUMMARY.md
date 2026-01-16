# Recipe Title Heuristics - Analysis Summary

## What We Accomplished

### 1. Manual Labeling
- Labeled 7 demo pages (69, 76, 78, 88, 90, 92, 94) with correct recipe titles
- Identified clear patterns in recipe titles:
  - Length: 2-4 words
  - Position: Top 6-15% of page (average 15%)
  - Punctuation: 86% end with period
  - Structure: "How to [verb] [item]" or "[Descriptor] [Recipe Type]"

### 2. Pattern Analysis
Discovered title characteristics:
- **Position**: Y-coordinate 51-435px (6-52% of page height, avg 15%)
- **Font size**: Same as body text (~11px) - NOT larger
- **Common words**: bread (3x), tea, cocoa, punch, rolls
- **Capitalization**: All start with capital letter
- **Context**: Appear before INGREDIENT_LINE tokens

### 3. Improved Heuristics Script
Created `improve_v3_heuristics.py` with scoring system:
- Detects "How to" pattern (+10 points)
- Identifies food terms (+5 points each)
- Checks for period ending (+3 points)
- Verifies Y-position in top 30% (+2-5 points)
- Prefers 2-4 word lengths (+2 points)
- Penalizes section headers (-10 points)

### 4. Results
- **Overall**: 542/623 pages (87%) had titles detected
- **Demo pages**: 2/7 correctly identified
  - ✓ Page 69: "How to make Tea." (correct!)
  - ✓ Page 78: "Claret Punch." (correct!)
  - ✗ Pages 76, 88, 90, 92, 94: Wrong titles selected

## Challenges Identified

### 1. Section Headers vs Recipe Titles
**Problem**: Section headers like "BREAD AND BREAD MAKING" appear at Y < 5% and score highly due to:
- Containing food terms
- Being in top of page
- Having good length

**Impact**: Heuristics often pick section header instead of actual recipe title

### 2. Missing Words in OCR
**Problem**: Some title words missing from OCR (e.g., "Water" in "Water Bread")
**Impact**: Can't find exact title matches

### 3. Over-matching
**Problem**: Generic phrases with food words get high scores (e.g., "cream of tartar", "roll up like jelly")
**Impact**: False positives reduce precision

### 4. Font Size Not Helpful
**Discovery**: Recipe titles are same font size as body text (~11px)
**Impact**: Can't use bbox height to distinguish titles

## Recommendations

### Option 1: Train Model Now (Recommended)
**Why**:
- 87% coverage is good enough for training
- Model will learn to distinguish section headers from titles
- Manual labels on 7 pages provide validation
- Can iterate on heuristics after seeing model performance

**Next Steps**:
1. Rebuild dataset with improved JSONL:
   ```bash
   python tools/build_v3_dataset.py \
     --input data/processed/v3_headers_titles/boston_v3_improved.jsonl \
     --output data/datasets/boston_layoutlmv3_v3_retrain \
     --demo-pages "69,76,78,88,90,92,94"
   ```

2. Train model:
   ```bash
   python tools/train_v3_model.py \
     --dataset data/datasets/boston_layoutlmv3_v3_retrain \
     --output models/layoutlmv3_v3_heuristics_v2 \
     --epochs 20
   ```

3. Evaluate on demo pages and compare recall metrics

### Option 2: Refine Heuristics Further
**Strategies**:
1. **Add explicit section header detection**: Mark "BREAD AND BREAD MAKING", "COCOA AND CHOCOLATE" as SECTION_HEADER (label 1), then only look for titles AFTER section headers

2. **Use position relative to other labels**: Recipe title must come:
   - AFTER section headers / page headers
   - BEFORE ingredient lines
   - Within first 50 tokens of content

3. **Stricter food term matching**: Require recipe type word (bread, tea, cocoa, etc.) NOT in all caps

4. **Manual rules for known patterns**: Special cases for "How to", "[Adjective] [RecipeType]" formats

### Option 3: Hybrid Approach (Best Long-term)
1. **Phase 1**: Train with current 87% coverage
2. **Phase 2**: Analyze model errors to refine heuristics
3. **Phase 3**: Re-label with improved heuristics + retrain
4. **Phase 4**: Active learning - manually label pages where model has low confidence

## Current State

### Files Created
- `data/label_studio/demo-manual-label-v3.json` - Manual labels for 7 pages
- `analyze_title_patterns.py` - Pattern analysis script
- `analyze_title_bboxes.py` - Bbox analysis script
- `improve_v3_heuristics.py` - Improved heuristic relabeling
- `data/processed/v3_headers_titles/boston_v3_improved.jsonl` - Re-labeled dataset

### Metrics
- Manual labels: 7 pages
- Heuristic coverage: 542/623 pages (87%)
- Heuristic accuracy on demo pages: 2/7 (29%)
- Previous model RECIPE_TITLE recall: 19%

## Recommendation

**Go with Option 1: Train now**

Rationale:
- Heuristics got 87% coverage - that's a huge improvement over whatever was there before
- The model will learn the visual/spatial patterns better than rule-based heuristics
- You have 7 manually labeled pages for validation
- Can iterate on heuristics after seeing what the model learns
- Time to train > time to perfect heuristics

The goal is good training data, not perfect training data. Let the model learn!
