# Title Extraction Improvements - January 15, 2026

## Summary

Successfully improved recipe title extraction using an enhanced heuristic fallback system. The heuristic now correctly handles pages where the ML model fails (0% accuracy on RECIPE_TITLE class).

## Problems Solved

### 1. Incorrect Titles from ML Model Failure
**Problem**: Page 519 showed "(Gherkins)." instead of "Unripe Cucumber Pickles (Gherkins)."
- ML model had 0% perfect accuracy on RECIPE_TITLE detection
- Root cause: Severe class imbalance (1.8% of training tokens)

**Solution**: Enhanced heuristic fallback with line grouping
- Filters capitalized tokens in top portion of page (y: 50-400px)
- Groups tokens on same line (within 12px Y threshold)
- Requires minimum 2 tokens to avoid scattered marginal notes
- Sorts by X position for correct left-to-right reading order

### 2. Leading Apostrophes
**Problem**: Titles showing as 'Water Bread, 'Spun Sugar
**Solution**: Unicode quote stripping including U+2018-U+201E

### 3. Concatenated Titles
**Problem**: Multiple recipes per card like "Sweet French Rolls. Luncheon Rolls."
**Solution**: Stop at first period when extracting titles

### 4. Non-Recipe Pages
**Problem**: Pages 536-623 (back matter) parsed as recipes
**Solution**: Limited parsing to pages 69-535

## Results

### Current Statistics (391 recipes)
- **Heuristic extractions**: 230 recipes (58.8%) with conf=0.5
- **ML predictions**: 155 recipes (39.6%) with conf>0.5
- **Leading apostrophes**: 0 ✓
- **Concatenated titles**: 0 ✓
- **Out-of-range pages**: 0 ✓

### Verified Examples
- Page 519: **"Unripe Cucumber Pickles (Gherkins)."** ✓ (was "(Gherkins).")
- Page 100: "German Toast." ✓
- Page 250: "Broiled Chicken." ✓
- Page 450: "Sponge Drop." ✓

## Technical Implementation

### File: `backend/app/utils/recipe_extraction.py`

#### New Function: `extract_title_heuristic()`
```python
def extract_title_heuristic(all_tokens: List[Token]) -> Tuple[Line | None, float]:
    """
    Heuristic fallback for title extraction when ML model fails.
    Strategy: Find first line of capitalized text near top of page that ends with period.
    """
    # Filter to top portion of page (y < 400 pixels) and exclude page headers
    candidates = [
        t for t in all_tokens
        if t.bbox[1] < 400  # Top of page
        and t.bbox[1] > 50  # Below page header
        and len(t.text) > 1  # Not punctuation
        and (t.text[0].isupper() or t.text[0] == '(')
    ]

    # Find first group of tokens on same line (within 12px Y threshold)
    # This filters out scattered marginal notes
    title_tokens = []
    y_thresh = 12

    for token in candidates:
        if not title_tokens:
            title_tokens.append(token)
            continue

        # Check if token is on same line as previous tokens
        avg_y = sum(t.mid_y for t in title_tokens) / len(title_tokens)

        if abs(token.mid_y - avg_y) <= y_thresh:
            # Same line - add it
            title_tokens.append(token)
            if token.text.rstrip().endswith('.'):
                break
        else:
            # Different line - if we have 2+ tokens, we found a title
            if len(title_tokens) >= 2:
                if any(t.text.rstrip().endswith('.') for t in title_tokens):
                    break
                # Reset and start new line
                title_tokens = [token]
            else:
                # Less than 2 tokens, likely scattered words - continue
                title_tokens = [token]

    # Need at least 2 tokens to be a valid title
    if len(title_tokens) < 2:
        return None, 0.0

    # Sort tokens by X position (left to right) for proper reading order
    title_tokens = sorted(title_tokens, key=lambda t: t.bbox[0])

    # Combine and clean
    combined_text = " ".join(t.text for t in title_tokens)
    combined_text = combined_text.strip().lstrip("'\"\u2018\u2019\u201c\u201d\u201a\u201e").rstrip("'\"\u2018\u2019\u201c\u201d\u201a\u201e")
    combined_text = clean_text(combined_text)

    return Line(...), 0.5  # Mark as heuristic with 0.5 confidence
```

#### Modified: `extract_title_obj()`
```python
def extract_title_obj(grouped: Dict[str, List[Token]], all_tokens: List[Token] = None):
    titles = grouped.get("RECIPE_TITLE") or []

    # Always try heuristic for comparison
    heuristic_title, heuristic_conf = None, 0.0
    if all_tokens:
        heuristic_title, heuristic_conf = extract_title_heuristic(all_tokens)

    # If no ML titles, use heuristic
    if not titles:
        if heuristic_title:
            return heuristic_title, heuristic_conf
        return None, 0.0

    # Check if ML titles are in suspicious positions
    ml_positions = [t.bbox[1] for t in titles]
    avg_y_position = sum(ml_positions) / len(ml_positions)

    # If ML titles are too far down the page (y > 350), prefer heuristic
    if avg_y_position > 350 and heuristic_title:
        return heuristic_title, heuristic_conf

    # If ML has too many tokens (>10), likely concatenating - use heuristic
    if len(titles) > 10 and heuristic_title:
        return heuristic_title, heuristic_conf

    # ... continue with ML extraction
```

## Next Steps (Phase 2)

The heuristic is working well as a fallback, but we should still retrain the ML model with proper class balancing to improve overall accuracy:

1. Add weighted cross-entropy loss (37.5x weight for RECIPE_TITLE)
2. Retrain for 15 epochs (~2-3 hours)
3. Evaluate on test set
4. Re-parse if ML accuracy improves

See `docs/RETRAINING_PLAN.md` for detailed implementation plan.

## Files Modified

1. `backend/app/utils/recipe_extraction.py`:
   - Added `extract_title_heuristic()` function (lines 99-177)
   - Modified `extract_title_obj()` to use heuristic with position validation (lines 179-207)
   - Modified `recipe_from_prediction()` to pass all tokens (lines 244-264)

2. `tools/parse_full_cookbook.py`:
   - Added automatic cleanup of old recipe files (lines 176-180)
   - Page range filter to 69-535 (line 207)

## Diagnostic Tools Created

1. `tools/inspect_page_tokens.py`: Inspect ML predictions for any page
2. `tools/evaluate_title_accuracy.py`: Evaluate RECIPE_TITLE detection accuracy across dataset

