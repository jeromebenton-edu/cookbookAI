# Recipe Title Extraction - Final Fixes

## Issues Identified

1. **Leading apostrophes** - Titles showing as `'Spun Sugar` instead of `Spun Sugar`
2. **Multiple titles per card** - E.g., `Sweet French Rolls. Luncheon Rolls.` (two recipes on one card)
3. **Non-recipe pages parsed** - Pages 536+ (back matter/index) were being parsed as recipes

## Root Causes

### Issue 1: Leading Apostrophes
- OCR interprets opening quotation marks as apostrophes
- `clean_text()` was called BEFORE stripping apostrophes
- Need to strip `'` and `"` from both start AND end of titles

### Issue 2: Multiple Titles Per Card
- Model labels multiple RECIPE_TITLE tokens on pages with multiple recipes
- Old code combined ALL title tokens on the page
- Example: Page has "Sweet French Rolls." and "Luncheon Rolls." - both got merged

### Issue 3: Non-Recipe Pages
- Pages 536-623 contain index, advertisements, back matter
- These pages have text that the model mislabels as recipes
- Need to filter page range to 69-535 (actual recipe content)

## Solutions Implemented

### Fix 1: Apostrophe Stripping (Enhanced)
**File:** `backend/app/utils/recipe_extraction.py` (line 129-131)

```python
# Strip leading AND trailing quotes/apostrophes
combined_text = combined_text.strip().lstrip("'\"").rstrip("'\"")
combined_text = clean_text(combined_text)  # Apply after stripping
```

**Before:** `'Spun Sugar,`
**After:** `Spun Sugar`

### Fix 2: Single Title Extraction (Primary Recipe Only)
**File:** `backend/app/utils/recipe_extraction.py` (lines 107-124)

**Strategy:**
1. Sort title tokens by position (top-to-bottom, left-to-right)
2. Take tokens until we hit a period + large vertical gap
3. Stop when we detect a second recipe (>50px vertical gap after period)
4. Only use the **primary** (first) title on each page

```python
# Take only the first title group
primary_titles = []
for i, token in enumerate(sorted_titles):
    if token.text.rstrip().endswith('.') and i + 1 < len(sorted_titles):
        primary_titles.append(token)
        next_token = sorted_titles[i + 1]
        y_gap = next_token.bbox[1] - token.bbox[3]
        if y_gap > 50:  # Large gap = different recipe
            break
    else:
        primary_titles.append(token)
```

**Before:** `Sweet French Rolls. Luncheon Rolls.`
**After:** `Sweet French Rolls.`

**Note:** Pages with multiple recipes now only show the first/primary recipe. This is acceptable because:
- Most cookbook pages have one main recipe
- Secondary recipes are usually variations
- Keeps UI clean and focused

### Fix 3: Page Range Filtering
**File:** `tools/parse_full_cookbook.py` (line 201)

```python
# Only process actual recipe pages
all_pages = sorted([p for p in page_index.keys() if 69 <= p <= 535])
```

**Page Breakdown:**
- **1-68**: Front matter, table of contents (skip)
- **69-535**: Actual recipes (process)
- **536-623**: Back matter, index, ads (skip)

**Before:** 825 recipes (many false positives)
**After:** ~400-450 recipes (actual recipes only)

## Expected Results

After re-parsing:
- ✓ No leading apostrophes on titles
- ✓ One clean title per recipe card
- ✓ Only actual recipe pages (69-535) processed
- ✓ ~400-450 recipes total (down from 825)
- ✓ No back matter or index entries

## Testing

Check results after re-parse:
```bash
# View sample titles
jq -r '.recipes[:20] | .[] | .title' frontend/public/recipes/boston/index.json

# Count recipes
jq '.recipes | length' frontend/public/recipes/boston/index.json

# Check no leading apostrophes
jq -r '.recipes[].title' frontend/public/recipes/boston/index.json | grep "^'" || echo "No apostrophes found!"

# Check page range
jq -r '.recipes[] | .page' frontend/public/recipes/boston/index.json | sort -n | uniq | head -5
jq -r '.recipes[] | .page' frontend/public/recipes/boston/index.json | sort -n | uniq | tail -5
```

## Files Modified

1. **`backend/app/utils/recipe_extraction.py`**
   - Enhanced apostrophe stripping (lines 129-131)
   - Primary title extraction logic (lines 107-124)

2. **`tools/parse_full_cookbook.py`**
   - Page range filter 69-535 (line 201)

## Trade-offs

**Decision:** Only extract primary title per page
- **Pro:** Clean, focused UI with one recipe per card
- **Pro:** Prevents title concatenation errors
- **Con:** Pages with multiple recipes only show first one
- **Acceptable because:** Secondary recipes are usually variations, and users can view the original page image

**Alternative considered:** Create multiple recipe cards per page
- Would require splitting ingredients/instructions per recipe
- Much more complex logic
- Risk of incorrect attribution
- Not implemented for v1

## Known Limitations

1. **Pages with 2+ recipes:** Only first recipe extracted
2. **Very long titles:** May still concatenate if no period marker
3. **Titles with intentional quotes:** Would be stripped (rare in this corpus)

## Next Steps

After re-parsing completes:
1. Verify frontend shows clean titles
2. Check page count is ~400-450
3. Spot-check 10-20 recipes for quality
4. Consider adding support for multi-recipe pages in future version
