# Critical Issue: RECIPE_TITLE Detection Accuracy

## Problem Discovered

The current model (`layoutlmv3_v3_manual_59pages_balanced`) has **catastrophic failure** in detecting recipe titles.

### Evaluation Results

Tested on 20 random pages with RECIPE_TITLE labels:
- **Perfect accuracy: 0%** (0/20 pages)
- **Partial matches: 5%** (1/20 pages)
- **Complete failures: 95%** (19/20 pages)

### Example: Page 519 (Gherkins)

**Ground Truth:**
- Title tokens: "Unripe", "Cucumber", "Pickles", "(Gherkins)."
- Combined: "Unripe Cucumber Pickles (Gherkins)."

**Model Prediction:**
- Title tokens: "quarts", "cucumbers", "boiling", "and", "let", "stand", "three", "days.", "Drain", "water", "brine,", "from", "bring"
- These are random words from the recipe instructions, NOT the title!

## Root Cause: Severe Class Imbalance

Dataset label distribution:
```
Training set (465 pages):
  O (background):       67.59%  (74,905 tokens)
  INGREDIENT_LINE:      15.81%  (17,518 tokens)
  INSTRUCTION_STEP:     12.60%  (13,960 tokens)
  PAGE_HEADER:           2.13%   (2,357 tokens)
  RECIPE_TITLE:          1.80%   (2,000 tokens)  ⚠️ TOO LOW
  SECTION_HEADER:        0.03%      (35 tokens)
  SERVINGS:              0.03%      (30 tokens)
  TEMP:                  0.02%      (22 tokens)
```

**RECIPE_TITLE represents only 1.8% of training tokens** - the model can't learn from such limited examples, especially when competing with 67% background tokens and 28% ingredient/instruction tokens.

## Impact on Production

This explains why:
1. **Titles are often wrong** - Model labels random instruction words as titles
2. **Title extraction fails** - Post-processing can't fix fundamentally wrong predictions
3. **Many "Recipe from page X" fallbacks** - When model finds no RECIPE_TITLE tokens
4. **Concatenated titles** - Model labels multiple unrelated tokens as titles

**No amount of post-processing logic can fix this** - the model's predictions are fundamentally incorrect.

## Solutions

### Option 1: Retrain with Class Balancing (Recommended)

Use weighted loss or oversampling to give RECIPE_TITLE more importance:

```python
# In training script
from torch import nn

# Calculate class weights (inverse frequency)
class_weights = {
    'O': 1.0,
    'INGREDIENT_LINE': 1.0,
    'INSTRUCTION_STEP': 1.0,
    'PAGE_HEADER': 5.0,
    'RECIPE_TITLE': 30.0,  # 30x weight for rare class
    'SECTION_HEADER': 100.0,
    'SERVINGS': 100.0,
    'TEMP': 100.0,
}

# Use weighted CrossEntropyLoss
criterion = nn.CrossEntropyLoss(weight=torch.tensor(list(class_weights.values())))
```

### Option 2: Add More RECIPE_TITLE Examples

Annotate more pages, focusing on pages with clear, distinct titles. Aim for at least 5,000-10,000 RECIPE_TITLE tokens (3-5x current amount).

### Option 3: Use Heuristic Fallback

When model confidence for RECIPE_TITLE is low, fall back to heuristics:
- Look for bold/large text near top of page
- First line of text after page header
- Text matching pattern: `^[A-Z][a-z]+ .+ \.$`

### Option 4: Two-Stage Approach

1. **Stage 1**: Use LayoutLMv3 for ingredients/instructions only
2. **Stage 2**: Use separate title extraction model or rules

## Immediate Actions

1. **Document limitation** - Add to README that title extraction is unreliable
2. **Manual curation** - Allow users to edit/correct titles
3. **Confidence threshold** - Only show titles with >0.8 confidence
4. **Fall back to heuristics** - When no RECIPE_TITLE found, extract first bold text

## Long-term Plan

1. **Retrain model** with class balancing (1-2 days)
2. **Add more annotations** focusing on title-heavy pages (2-3 days)
3. **Evaluate on held-out test set** to verify improvement
4. **Re-parse all recipes** with improved model

## Files for Reference

- **Diagnostic tool**: `tools/inspect_page_tokens.py` - Inspect token-level predictions for any page
- **Evaluation tool**: `tools/evaluate_title_accuracy.py` - Measure accuracy across dataset
- **Model path**: `models/layoutlmv3_v3_manual_59pages_balanced/`
- **Dataset path**: `data/datasets/boston_layoutlmv3_v3_manual_expanded2/dataset_dict/`

## Example Commands

```bash
# Inspect specific page
python tools/inspect_page_tokens.py 519

# Evaluate title accuracy on 50 pages
python tools/evaluate_title_accuracy.py 50

# Check dataset statistics
python -c "from datasets import load_from_disk; ds = load_from_disk('data/datasets/boston_layoutlmv3_v3_manual_expanded2/dataset_dict'); print(ds)"
```

## Conclusion

The model is **not fit for production** for title extraction. Either:
1. Retrain with proper class balancing, OR
2. Implement heuristic-based title extraction, OR
3. Accept that most titles will be wrong and require manual correction

Current post-processing logic (stopping at first period, limiting tokens, etc.) is **a band-aid** that doesn't address the root cause.
