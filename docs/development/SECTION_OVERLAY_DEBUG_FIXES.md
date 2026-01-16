# Section Overlay Debug Fixes

## Root Cause: Incorrect Coordinate Space Conversion

### Problem
Bounding boxes were not rendering in Inspector mode after implementing section overlays.

### Root Cause
The bbox coordinates in prediction JSON files are in **original image pixel space** (500x819px), but the rendering code was incorrectly detecting them as **normalized 0-1000 coordinates** because:

```typescript
// WRONG: Treats any maxVal <= 1000 as normalized
const maxVal = Math.max(x1, y1, x2, y2);  // = 500 (from 500px image width)
const isNormalized = maxVal <= 1000;      // TRUE (incorrect!)
const left = (x1 / 1000) * renderSize.width;  // WRONG: divides by 1000 instead of 500
```

This caused boxes to render at 1/2 the correct size and position.

### Fix
Always use the **natural image dimensions** to scale coordinates:

```typescript
// CORRECT: Use actual natural image dimensions
const naturalW = imageRef.current?.naturalWidth ?? 500;  // e.g., 500px
const naturalH = imageRef.current?.naturalHeight ?? 819; // e.g., 819px
const left = (x1 / naturalW) * renderSize.width;  // Scale from 500px to displayed width
```

## Additional Fixes Applied

### 1. Improved Title Matching (sectionOverlays.ts)
**Problem**: Title matching was case-sensitive and didn't handle punctuation.

**Fix**: Robust normalization function:
```typescript
const normalize = (text: string) =>
  text
    .toLowerCase()
    .trim()
    .replace(/[.,;:!?]/g, "")      // Remove punctuation
    .replace(/\s+/g, " ");          // Collapse multiple spaces

const titleMatch = titleTokens.find((t) => {
  const normalizedToken = normalize(t.text);
  return (
    normalizedToken.includes(normalizedTarget) ||
    normalizedTarget.includes(normalizedToken)
  );
});
```

### 2. Enhanced Debugging
Added comprehensive console logging:

**sectionOverlays.ts**:
- Logs all title candidates with normalized text
- Shows selectedTitleMatch, filteredIngredientCount, filteredInstructionCount

**SectionOverlayViewer.tsx**:
- Logs overlay computation results
- Logs first box rendering with naturalW, naturalH, computed positions
- Dev banner showing section count and render size

### 3. UI Improvements (InspectorModeView.tsx)

**Mode indicator**:
```tsx
<span className="rounded-full bg-blue-100 px-2 py-0.5">
  {overlayMode === "sections" ? "Sections" : "Tokens"}
</span>
```

**Conditional controls**:
- Token label toggles + confidence slider: **only shown in tokens mode**
- Sections mode: shows simple "Show boxes" checkbox + selected title

**Better messaging**:
- Sections mode: "Viewing section-level overlays (Title, Ingredients, Instructions) for the selected recipe."
- Tokens mode: "Viewing token-level ML predictions with label and confidence filtering."

### 4. Dev Banner (SectionOverlayViewer.tsx)
Shows debug info when sections are found:
```tsx
<div className="absolute right-2 top-2 bg-blue-500 text-white">
  Sections: {sections.length} | Render: {renderSize.width}x{renderSize.height}
</div>
```

Shows helpful error message when no overlays found:
```tsx
<p>No section overlays found</p>
<p className="text-xs">
  Title match: {debug.selectedTitleMatch?.text ?? "none"} |
  Ingredients: {debug.filteredIngredientCount} |
  Instructions: {debug.filteredInstructionCount}
</p>
```

## Files Changed

1. **[frontend/components/ai/SectionOverlayViewer.tsx](frontend/components/ai/SectionOverlayViewer.tsx)**
   - Fixed coordinate space conversion (naturalWidth/Height instead of assumed normalization)
   - Added debug logging and dev banner
   - Enhanced empty state messaging

2. **[frontend/lib/overlays/sectionOverlays.ts](frontend/lib/overlays/sectionOverlays.ts)**
   - Improved title matching with robust normalization
   - Added debug logging for title matching

3. **[frontend/components/demo/InspectorModeView.tsx](frontend/components/demo/InspectorModeView.tsx)**
   - Added mode indicator badge
   - Conditional rendering of controls based on overlay mode
   - Better UX messaging for each mode
   - Simple show boxes toggle for sections mode

## Testing

### Before Fix
- No boxes rendered (coordinates scaled incorrectly)
- Token controls shown even in sections mode (confusing)
- No indication of which overlay mode was active

### After Fix
- ✅ Boxes render correctly in sections mode (3 boxes: Title, Ingredients, Instructions)
- ✅ Boxes only cover selected recipe (Waffles), not other recipes on the page
- ✅ Token controls hidden in sections mode
- ✅ Clear mode indicator showing "Sections" or "Tokens"
- ✅ Dev banner and console logs for debugging

## How to Test

1. Go to `localhost:3000/demo`
2. Click "Show how the AI found this" → enters Inspector mode
3. Default view should show **Sections mode** with:
   - Blue badge showing "Sections"
   - 3 clean boxes (Title, Ingredients, Instructions) around Waffles recipe
   - No boxes on other recipes (Bread Griddle-Cakes, Buckwheat Cakes)
4. Expand "Advanced ▾" and toggle to **Tokens mode**:
   - Badge changes to "Tokens"
   - Shows token label toggles and confidence slider
   - Displays all raw token boxes with filtering

## Coordinate Space Reference

| Space | Width | Height | Usage |
|-------|-------|--------|-------|
| Original image | 500px | 819px | Prediction bbox coordinates |
| Displayed (example) | 400px | 655px | Browser rendering size |
| Scaling factor | 0.8 | 0.8 | renderSize.width / naturalWidth |

**Correct formula**:
```
displayedX = (bboxX / naturalWidth) * renderSize.width
```

**Incorrect formula** (original bug):
```
displayedX = (bboxX / 1000) * renderSize.width  // WRONG!
```
