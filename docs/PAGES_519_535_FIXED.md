# Pages 519 and 535 - Title Extraction Fixed

## Summary

Successfully fixed title extraction for both problematic pages identified by the user.

## Issues Resolved

### Page 519: Unripe Cucumber Pickles
**Before**: `(Gherkins).`  
**After**: `Unripe Cucumber Pickles (Gherkins).` ✓  
**Method**: Heuristic fallback (confidence: 0.5)

### Page 535: Rennet Custard  
**Before**: `(Junket).`  
**After**: `Rennet Custard (Junket).` ✓  
**Method**: ML model extraction (confidence: 0.67)

## Root Cause Analysis

### Page 519 - ML Model Failure
The ML model completely failed to detect the correct title tokens:
- **Expected**: "Unripe", "Cucumber", "Pickles", "(Gherkins)."
- **Actual predictions**: "quarts", "cucumbers", "boiling", "and", "let", "stand" (random instruction words)

**Solution**: Enhanced heuristic fallback that:
1. Filters capitalized words in top of page (Y: 50-400px)
2. Groups tokens on same line (within 12px Y threshold)
3. Requires minimum 2 tokens to avoid scattered marginal notes
4. Sorts by X position (left-to-right) for proper reading order

### Page 535 - Token Sorting Issue
The ML model DID detect all three title tokens correctly:
- "Rennet" @ Y=250
- "Custard" @ Y=249
- "(Junket)." @ Y=246

However, the extraction logic sorted by (Y, X) which put "(Junket)." first (smallest Y value), then stopped at the first period, producing incomplete title.

**Solution**: Modified ML extraction to:
1. Group tokens by line (similar Y positions within 12px)
2. Sort by X position (left-to-right) within the line
3. Take tokens until hitting a period (now reads left-to-right)

## Technical Changes

### 1. Enhanced Heuristic (`extract_title_heuristic`)
**File**: `backend/app/utils/recipe_extraction.py` (lines 99-186)

```python
def extract_title_heuristic(all_tokens: List[Token]) -> Tuple[Line | None, float]:
    # Filter candidates: capitalized, top of page, not scattered
    candidates = [
        t for t in all_tokens
        if t.bbox[1] < 400 and t.bbox[1] > 50
        and len(t.text) > 1
        and (t.text[0].isupper() or t.text[0] == '(')
    ]
    
    # Group by line (within 12px Y threshold)
    y_thresh = 12
    title_tokens = []
    
    for token in sorted(candidates, key=lambda t: (t.mid_y, t.mid_x)):
        if not title_tokens:
            title_tokens.append(token)
            continue
            
        avg_y = sum(t.mid_y for t in title_tokens) / len(title_tokens)
        
        if abs(token.mid_y - avg_y) <= y_thresh:
            # Same line - add it
            title_tokens.append(token)
            if token.text.rstrip().endswith('.'):
                break
        else:
            # Different line - keep if we have 2+ tokens
            if len(title_tokens) >= 2:
                if any(t.text.rstrip().endswith('.') for t in title_tokens):
                    break
                title_tokens = [token]
            else:
                title_tokens = [token]
    
    # Sort by X position for left-to-right reading order
    title_tokens = sorted(title_tokens, key=lambda t: t.bbox[0])
    
    # Clean and return
    combined_text = " ".join(t.text for t in title_tokens)
    combined_text = clean_text(combined_text.strip().lstrip("'\"\u2018\u2019\u201c\u201d\u201a\u201e"))
    
    return Line(..., confidence=0.5)
```

### 2. Fixed ML Token Sorting (`extract_title_obj`)
**File**: `backend/app/utils/recipe_extraction.py` (lines 215-244)

```python
# OLD (incorrect):
sorted_titles = sorted(titles, key=lambda t: (t.bbox[1], t.bbox[0]))  # Sorts by Y first

# NEW (correct):
# Group tokens by line (similar Y positions)
first_y = titles[0].mid_y
title_line_tokens = [t for t in titles if abs(t.mid_y - first_y) <= 12]

# Sort by X position (left to right) within the line
sorted_titles = sorted(title_line_tokens, key=lambda t: t.bbox[0])
```

### 3. Added Model Evaluation Mode
**File**: `tools/parse_full_cookbook.py` (line 189)

```python
model = LayoutLMv3ForTokenClassification.from_pretrained(model_path).to(device)
model.eval()  # Set to evaluation mode for deterministic predictions
```

## Validation Tests

### Test 1: Page 519 Heuristic
```bash
$ python tools/inspect_page_tokens.py 519
# ML model detects: "quarts", "cucumbers", "boiling"... (wrong)
# Heuristic detects: "Unripe Cucumber Pickles (Gherkins)." (correct)
```

### Test 2: Page 535 ML Extraction
```bash
$ python -c "..." # Test ML token detection
RECIPE_TITLE tokens detected:
  "Rennet" @ y=250 conf=0.6245
  "Custard" @ y=249 conf=0.6846
  "(Junket)." @ y=246 conf=0.6909

# After fix, reads left-to-right: "Rennet Custard (Junket)."
```

## Results

### Overall Statistics (391 recipes)
- **Heuristic extractions**: 230 recipes (58.8%)
- **ML extractions**: 161 recipes (41.2%)
- **Leading apostrophes**: 0 ✓
- **Concatenated titles**: 0 ✓
- **Out-of-range pages**: 0 ✓

### Verified Pages
- ✓ Page 519: "Unripe Cucumber Pickles (Gherkins)."
- ✓ Page 535: "Rennet Custard (Junket)."
- ✓ Page 100: "German Toast."
- ✓ Page 250: "Broiled Chicken."

## Frontend Access

Both recipes are now accessible:
- http://localhost:3000/recipes/unripe-cucumber-pickles-gherkins-p0519
- http://localhost:3000/recipes/rennet-custard-junket-p0535

## Files Modified

1. **backend/app/utils/recipe_extraction.py**
   - Lines 99-186: Enhanced `extract_title_heuristic()` with line grouping
   - Lines 189-213: Position-based validation (avg_y > 350, count >= 8)
   - Lines 215-244: Fixed ML token sorting to read left-to-right

2. **tools/parse_full_cookbook.py**
   - Line 189: Added `model.eval()` for deterministic predictions

## Next Steps

Phase 2: Model retraining with class balancing (see `docs/RETRAINING_PLAN.md`) will further improve ML accuracy and reduce reliance on heuristic fallback.

---

**Date**: January 15, 2026  
**Status**: RESOLVED ✓
