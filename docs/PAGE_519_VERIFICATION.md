# Page 519 Verification - Before and After

## The Problem

When you provided the screenshot of page 519, the parsed recipe showed an incorrect title:

**Incorrect Title**: `(Gherkins).`

**Correct Title** (from original scan): `Unripe Cucumber Pickles (Gherkins).`

## Root Cause

Investigation revealed:
1. ML model had **0% perfect accuracy** on RECIPE_TITLE detection
2. Model predicted random instruction words as title tokens
3. Class imbalance: RECIPE_TITLE only 1.8% of training data

### ML Model Predictions for Page 519

The model incorrectly predicted these tokens as RECIPE_TITLE:
- "quarts", "cucumbers", "boiling", "and", "let", "stand"

**Ground Truth** (what it should have detected):
- "Unripe", "Cucumber", "Pickles", "(Gherkins)."

## The Solution

### Enhanced Heuristic Algorithm

Created a position-based heuristic that:

1. **Filters candidates**: Capitalized words in top of page (Y: 50-400px)
2. **Groups by line**: Tokens within 12px Y threshold
3. **Requires minimum tokens**: At least 2 to avoid scattered words
4. **Sorts left-to-right**: By X position for reading order
5. **Validates ML predictions**: Uses heuristic when ML position is suspicious

### How It Works for Page 519

**Step 1 - Find candidates** (capitalized words near top):
```
1. "Make" @ y=64 x=532          ← Marginal note
2. "Add" @ y=135 x=466           ← Marginal note
3. "Remove" @ y=158 x=216        ← Marginal note
4. "Seald" @ y=182 x=316         ← Marginal note
5. "(Gherkins)." @ y=249 x=616   ← Title token
6. "Cucumber" @ y=250 x=312      ← Title token
7. "Unripe" @ y=251 x=192        ← Title token
8. "Pickles" @ y=251 x=486       ← Title token
```

**Step 2 - Group by line**:
- Tokens 1-4 are scattered (not on same line) → Skip
- Tokens 5-8 are on same line (Y: 249-251) → Group together

**Step 3 - Sort left-to-right**:
```
1. "Unripe" @ x=192
2. "Cucumber" @ x=312
3. "Pickles" @ x=486
4. "(Gherkins)." @ x=616
```

**Step 4 - Combine**:
**Result**: `Unripe Cucumber Pickles (Gherkins).` ✓

## Current Result

### Recipe File: `unripe-cucumber-pickles-gherkins-p0519.json`

```json
{
  "id": "unripe-cucumber-pickles-gherkins-p0519",
  "title": "Unripe Cucumber Pickles (Gherkins).",
  "source": {
    "page": 519
  },
  "ai": {
    "field_confidence": {
      "title": 0.5  // Heuristic marker
    }
  }
}
```

### Frontend URL
http://localhost:3000/recipes/unripe-cucumber-pickles-gherkins-p0519

## Verification

✅ Title correctly extracted: **"Unripe Cucumber Pickles (Gherkins)."**  
✅ No leading apostrophes  
✅ No concatenation with other recipes  
✅ Confidence 0.5 indicates heuristic was used (as expected)  

---

**Status**: RESOLVED
**Date**: January 15, 2026
