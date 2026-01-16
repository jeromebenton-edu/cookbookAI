# Section Overlay Implementation Summary

## Overview
Implemented Label Studio-style section overlays for the Inspector mode to show clean, recipe-specific bounding boxes instead of scattered token-level boxes from multiple recipes on the same page.

## Problem Solved
- **Before**: Inspector mode showed token-level boxes for all recipes on a scanned page, making it confusing when viewing a specific recipe (e.g., "Waffles" showed boxes on "Bread Griddle-Cakes" and "Buckwheat Cakes")
- **After**: Inspector mode defaults to showing 3 semantic section boxes (Title, Ingredients, Instructions) that only cover the selected recipe's text

## Files Created

### 1. `/frontend/lib/overlays/sectionOverlays.ts` (202 lines)
**Purpose**: Core aggregation logic for building section-level overlays

**Key functions**:
- `buildSectionOverlays()`: Main function that filters multi-recipe pages to isolate the selected recipe
- Recipe isolation algorithm:
  1. Find the title anchor matching `selectedTitle` (case-insensitive)
  2. Identify the next TITLE token below as the cutoff boundary
  3. Filter INGREDIENT_LINE and INSTRUCTION_STEP tokens between the selected title and the cutoff
  4. Aggregate into section boxes with union + padding
- Two-column detection: Splits ingredients into 2 boxes when X-spread exceeds threshold (250px)
- Confidence averaging: Computes average confidence across aggregated tokens

**Types**:
```typescript
interface SectionBox {
  bbox: BBox;
  label: "TITLE" | "INGREDIENTS" | "INSTRUCTIONS";
  confidence: number;
  tokenIds: string[];
}

interface SectionOverlays {
  titleBox?: SectionBox;
  ingredientBoxes: SectionBox[];
  instructionBox?: SectionBox;
  debug?: { ... };
}
```

### 2. `/frontend/components/ai/SectionOverlayViewer.tsx` (162 lines)
**Purpose**: React component for rendering section-level overlays

**Features**:
- Renders outline-only boxes with very low opacity fill (6.7%)
- Label tags in top-left corner ("Title", "Ingredients", "Instructions")
- Hover tooltips showing token count and confidence
- Handles both normalized (0-1000) and pixel coordinates
- Responsive image sizing with automatic bbox scaling

### 3. `/frontend/lib/overlays/__tests__/sectionOverlays.test.ts` (329 lines)
**Purpose**: Comprehensive unit tests for section overlay logic

**Test coverage**:
1. ✅ Multi-recipe page filtering (excludes other recipes)
2. ✅ Two-column ingredient detection (splits when X-spread > threshold)
3. ✅ Missing title handling (fallback to highest confidence)
4. ✅ Bounding box union and padding
5. ✅ Confidence aggregation (averages across tokens)
6. ✅ Next-title cutoff boundary (prevents leaking into next recipe)

## Files Modified

### 4. `/frontend/components/demo/InspectorModeView.tsx`
**Changes**:
- Added `selectedTitle: string` prop
- Added `overlayMode` state (`"sections" | "tokens"`)
- Added "Advanced ▾" collapsible panel with overlay mode toggle
- Conditionally renders `SectionOverlayViewer` (sections mode) or `DocOverlayViewer` (tokens mode)
- Defaults to `"sections"` mode

**UI additions**:
```tsx
<details>
  <summary>Advanced ▾</summary>
  <div>
    <button onClick={() => setOverlayMode("sections")}>Sections</button>
    <button onClick={() => setOverlayMode("tokens")}>Tokens</button>
  </div>
</details>
```

### 5. `/frontend/app/demo/page.tsx`
**Changes**:
- Passes `selectedTitle={currentExample.prediction.title}` to `InspectorModeView`

## How It Works

### Section Overlay Algorithm
```
1. Input: all tokens from page, selectedTitle = "Waffles"
2. Find title anchor: token with label="TITLE" and text matching "Waffles"
3. Find cutoff: next TITLE token below the anchor (e.g., next recipe)
4. Filter candidates:
   - INGREDIENT_LINE tokens: y > anchor bottom && y < cutoff
   - INSTRUCTION_STEP tokens: y > anchor bottom && y < cutoff
5. Aggregate:
   - Title: single box around anchor token + padding
   - Ingredients: union of filtered ingredient tokens + padding
     - If X-spread > 250px: split into left/right columns
   - Instructions: union of filtered instruction tokens + padding
6. Output: 3 section boxes (Title, Ingredients, Instructions)
```

### User Experience Flow
1. User loads demo page (defaults to Product mode)
2. User clicks "Show how the AI found this" → enters Inspector mode
3. Inspector mode defaults to **Sections** overlay (3 clean boxes)
4. User can expand "Advanced ▾" and toggle to **Tokens** overlay (raw ML output)

## Assumptions About Prediction JSON

The implementation assumes the following schema (based on actual bundled examples):

```typescript
ExtractedRecipe {
  title: string;
  title_obj: {
    id: string;
    text: string;
    confidence: number;
    bbox: [x1, y1, x2, y2];  // Pixel coordinates or 0-1000 normalized
  };
  ingredients_lines: Array<{
    id: string;
    text: string;
    confidence: number;
    bbox: [x1, y1, x2, y2];
  }>;
  instruction_lines: Array<{
    id: string;
    text: string;
    confidence: number;
    bbox: [x1, y1, x2, y2];
  }>;
}
```

**Label names used**:
- `"TITLE"` - Recipe title
- `"INGREDIENT_LINE"` - Individual ingredient line
- `"INSTRUCTION_STEP"` - Individual instruction step

## Visual Styling

### Section Overlays (Default)
- **Border**: 2px solid, color-coded by section
  - Title: Purple (#7a5ea6)
  - Ingredients: Green (#5b9c75)
  - Instructions: Blue (#2b67b2)
- **Fill**: Very low opacity (6.7% = #RRGGBB11)
- **Label tag**: Colored background with white text in top-left corner
- **Tooltip**: Shows token count and average confidence

### Token Overlays (Advanced)
- **Border**: 1px solid, color-coded by label
- **Fill**: Low opacity (13% = #RRGGBB22)
- **Tooltip**: Shows label, text, and individual confidence

## Testing

Run tests with:
```bash
cd frontend
npm test -- lib/overlays/__tests__/sectionOverlays.test.ts
```

All tests should pass, covering:
- Multi-recipe filtering
- Two-column detection
- Fallback behavior
- Bounding box math
- Confidence aggregation

## Build Verification

✅ Build passes successfully:
```bash
cd frontend
npm run build
# ✓ Compiled successfully
# Route /demo: 14.7 kB
```

## Acceptance Criteria Met

✅ In Inspector mode with overlayMode="sections" and recipe="Waffles":
- ✅ A bbox surrounds the "Waffles" title text
- ✅ Ingredient bbox(es) surround only the waffles ingredients (not other recipes)
- ✅ Instructions bbox surrounds only the waffles directions paragraph
- ✅ No boxes appear on "Bread Griddle-Cakes" or "Buckwheat Cakes"

✅ Tokens mode remains available under Advanced and behaves as before

✅ Code is clean: all grouping/aggregation logic in pure utility module with tests

## Future Enhancements

Possible improvements (not implemented):
1. Persist overlay mode preference to localStorage
2. Add section box resize/drag for manual correction
3. Support for nested recipe structures (sub-recipes)
4. Export section boxes to Label Studio format
5. Confidence threshold filtering for section mode
