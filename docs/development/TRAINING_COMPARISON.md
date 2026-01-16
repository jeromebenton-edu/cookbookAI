# Training Results Comparison: v3 Original vs Improved Heuristics

## Executive Summary

Training with improved heuristics provided **9x more RECIPE_TITLE training examples** (1,780 vs ~200). However, validation metrics show **NO improvement** in RECIPE_TITLE recall compared to the original model, likely because the demo pages don't have RECIPE_TITLE labels in ground truth.

---

## Training Data Comparison

### Original Model (v3_production)
- **Training set**: 463 pages with ~200 RECIPE_TITLE labels
- **Validation set**: 92 pages with 6 pages containing titles
- **RECIPE_TITLE coverage**: ~30-40% of pages

### Improved Model (v3_improved)
- **Training set**: 463 pages with **1,780 RECIPE_TITLE labels**
- **Validation set**: 92 pages with 6 pages containing titles
- **RECIPE_TITLE coverage**: 87% of pages (542/623 total)

**Improvement**: 9x more RECIPE_TITLE training examples!

---

## Validation Set Results (92 pages)

### RECIPE_TITLE Performance

| Metric | Original | Improved | Change |
|--------|----------|----------|--------|
| **Recall** | 19.5% | 19.6% | +0.1% ⚠️ |
| **Precision** | 57.1% | 65.4% | +8.3% ✓ |
| **F1 Score** | - | 30.2% | - |
| **Title Anchor Accuracy** | 100% (6/6) | 100% (6/6) | No change ✓ |

**Analysis**:
- ✓ Precision improved slightly
- ⚠️ **Recall unchanged** - model still misses 80% of titles
- ✓ Anchor accuracy still perfect on labeled pages

### Overall Token Classification

| Metric | Original | Improved | Change |
|--------|----------|----------|--------|
| **Accuracy** | 81.3% | 76.8% | -4.5% ⚠️ |
| **Precision** | 91.8% | 84.6% | -7.2% ⚠️ |
| **Recall** | 67.0% | 57.5% | -9.5% ⚠️ |

**Analysis**: Overall performance decreased, possibly due to:
- Noisy heuristic labels confusing the model
- Heuristics incorrectly labeling non-titles as RECIPE_TITLE
- Model learning from imperfect heuristic patterns

---

## Test Set Results (61 pages)

### RECIPE_TITLE Performance

| Metric | Improved | Notes |
|--------|----------|-------|
| **Recall** | 4.9% | Very low - model rarely predicts titles |
| **Precision** | 47.4% | When it does predict, ~50% correct |
| **F1 Score** | 9.0% | Poor overall |
| **Title Anchor Accuracy** | 20% (1/5) | Only 1 page with correct title bbox |

**Analysis**: Test set performance is much worse than validation, suggesting:
- Model overfitting to validation data
- Test set may have different title patterns
- Heuristics may have worked better on validation pages

---

## Demo Eval Results (7 manually labeled pages)

### Critical Finding

| Metric | Value | Issue |
|--------|-------|-------|
| **Title Anchor Accuracy** | 0% | Model found 0 pages with titles |
| **Pages with titles** | 0/7 | No RECIPE_TITLE predictions |
| **Precision** | 0.0% | N/A - no predictions |
| **Recall** | 0.0% | N/A - no ground truth titles |

**Root Cause**: The demo pages DON'T HAVE RECIPE_TITLE LABELS in the ground truth!

Looking at the demo_eval metrics:
```json
"eval_true_label_distribution": {
  "O": 1280,
  "INGREDIENT_LINE": 150,
  "INSTRUCTION_STEP": 231
  // NO RECIPE_TITLE!
}
```

The 7 demo pages only have:
- INGREDIENT_LINE
- INSTRUCTION_STEP
- O (Other)

**This means we can't evaluate RECIPE_TITLE performance on demo pages** because they're continuation pages without recipe titles in the current ground truth labeling.

---

## Why Didn't Recall Improve?

### Hypothesis 1: Heuristic Label Quality ✓ (Most Likely)
- Heuristics found 1,780 RECIPE_TITLE examples (87% coverage)
- But only 2/7 manually labeled demo pages were correctly identified by heuristics
- **Accuracy**: 29% on demo pages we verified
- Model trained on **noisy/incorrect labels** may have learned wrong patterns

### Hypothesis 2: Model Confusion
- 9x more examples, but many were false positives (section headers, random phrases with food words)
- Model may have learned to be even MORE conservative due to conflicting signals

### Hypothesis 3: Architecture Limitation
- LayoutLMv3 may not be learning the visual/spatial patterns needed
- Font size doesn't help (titles same size as body text)
- Position alone may not be sufficient

### Hypothesis 4: Ground Truth Issues
- Validation pages may have had titles all along, not from our heuristics
- Our "improvement" didn't actually add new correct labels where they were missing

---

## Key Insights

### What Worked ✓
1. **Heuristics increased coverage**: 87% of pages now have RECIPE_TITLE candidates
2. **Precision improved slightly**: 57% → 65% on validation
3. **Anchor accuracy maintained**: Still 100% on pages with correct labels
4. **Training completed successfully**: No crashes, proper class weighting

### What Didn't Work ⚠️
1. **Recall unchanged**: Still only catching ~20% of titles
2. **Overall metrics decreased**: Lower accuracy/precision/recall across all labels
3. **Demo pages unusable**: Ground truth doesn't have RECIPE_TITLE labels
4. **Heuristic accuracy low**: Only 29% correct on manually verified pages

### What We Learned 📚
1. **User insight was key**: "Titles are centered and bold, same as section headers" - this is why heuristics struggled
2. **More data ≠ better data**: 9x more examples didn't help when they were noisy
3. **Ground truth matters**: Demo eval is useless without proper labels
4. **Spatial patterns are hard**: Model can't distinguish titles from headers based on position/font alone

---

## Recommendations

### Option 1: Use Original Model ✓ (Recommended)
**Why**:
- Original model achieved same 19.5% recall with 100% anchor accuracy
- Fewer false positives
- Better overall token classification metrics
- Trained on cleaner data (even if less of it)

**When to use improved model**:
- Never - it's objectively worse

### Option 2: Fix Heuristics and Retrain
**What to fix**:
1. **Add SECTION_HEADER detection first**: Explicitly label "BREAD AND BREAD MAKING", "COCOA AND CHOCOLATE" as section headers
2. **Only look for titles AFTER section headers**: Recipe titles come after section intros
3. **Stricter food term matching**: Require recipe type word (bread, tea, etc.) in non-caps
4. **Position relative to other labels**: Title must be before first INGREDIENT_LINE

**Effort**: Medium (2-3 hours to refine heuristics)
**Expected improvement**: 40-60% recall if heuristics reach 60%+ accuracy

### Option 3: Manual Labeling + Active Learning
1. **Manually label 50-100 pages** with proper RECIPE_TITLE, SECTION_HEADER, PAGE_HEADER
2. **Train model on clean data**
3. **Use model to suggest labels** on remaining pages
4. **Manually correct** high-confidence errors
5. **Retrain** with corrected dataset

**Effort**: High (10-20 hours of labeling)
**Expected improvement**: 60-80% recall with high-quality training data

### Option 4: Hybrid Approach
1. **Keep using original model** for now (19.5% recall)
2. **Gradually improve heuristics** based on error analysis
3. **Manually label edge cases** the model misses
4. **Incremental retraining** as data quality improves

**Effort**: Low ongoing effort
**Expected improvement**: Gradual improvement over time

---

## Conclusion

The improved heuristics experiment revealed that:
1. **Quantity ≠ Quality**: 9x more training examples didn't help
2. **Heuristics are hard**: Only 29% accuracy on verified pages
3. **Model needs clean data**: Training on noisy labels made things worse
4. **Visual patterns are subtle**: Can't distinguish titles from headers with rules

**Recommendation**: Stick with the original v3_production model and invest effort in improving heuristics or manual labeling before next training run.

The good news: We now understand WHY the problem is hard (centered+bold titles that look like headers) and have a clear path forward (better section header detection + manual labeling).

---

## Files Generated

- `data/processed/v3_headers_titles/boston_v3_improved.jsonl` - Heuristically labeled dataset
- `models/layoutlmv3_v3_improved/` - Trained model (worse than original)
- `data/label_studio/demo-manual-label-v3.json` - Manual labels for 7 pages
- `analyze_title_patterns.py` - Pattern analysis script
- `analyze_title_bboxes.py` - Bbox analysis script
- `improve_v3_heuristics.py` - Heuristic relabeling script
- This comparison document

## Next Session Recommendations

1. **Don't use v3_improved model** - use original v3_production
2. **Focus on improving heuristics** with section header detection
3. **Consider manual labeling** for higher quality training data
4. **Update demo pages ground truth** to include RECIPE_TITLE labels for proper evaluation
