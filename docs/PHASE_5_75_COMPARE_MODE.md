# Phase 5.75 – Compare Mode & Corrections

Purpose: help users see where LayoutLMv3 is right/wrong, expose confidence, and export corrections for future training.

## Backend

- `/api/parse/boston/{page}/recipe` now returns:
  - `title_obj` with id/text/confidence/bbox
  - `ingredients_lines[]` and `instruction_lines[]` with ids, confidences, bboxes, token counts
  - `ingredients`/`instructions` arrays derived from the line objects (backward compatible)
- Query params: `include_lines` (default true), `include_raw`, `refresh`.
- Featured/recipe caches live under `backend/cache/`.

## Frontend Compare Mode

- Toggle between **Curated / AI Extracted / Compare**.
- Diff highlighting (line-level):
  - match/near-match: normal
  - curated-only: red
  - AI-only: green
  - partial mismatch: yellow with similarity
- Confidence chips per AI line (High/Med/Low).
- Inline edits: title, ingredients, instructions (add/delete/edit).
- **Export Corrections** dropdown with:
  - Download as file
  - Copy to clipboard
  - Inline help explaining the workflow

## How it works (diff)

- Lines normalized (lowercase, strip punctuation, collapse spaces).
- Similarity via bag-of-words overlap.
- Thresholds: ingredients 0.75, instructions 0.70.
- Remaining AI lines = extras; missing curated lines flagged.

## Export JSON Structure

```json
{
  "page_num": 79,
  "corrected": {
    "title": "Unfermented Grape Juice",
    "ingredients": ["10 lbs. grapes", "8 lbs. sugar", "1 cup water"],
    "instructions": ["Put grapes and water in granite stew-pan..."]
  },
  "ai_original": {
    "title": "...",
    "ingredients": [...],
    "instructions": [...],
    "confidence": { "overall": 0.85, ... }
  },
  "meta": {
    "model": "layoutlmv3",
    "overall_conf": 0.85,
    "created_at": "2026-01-17T..."
  }
}
```

## Corrections Workflow

### Step 1: Review and Edit

1. Open a recipe page (e.g., `/recipes/unfermented-grape-juice-p0079`)
2. Switch to **Compare** mode
3. Review AI extractions against curated data
4. Click AI fields to edit and fix errors

### Step 2: Export

1. Click **Export Corrections** dropdown
2. Choose **Download as file** or **Copy to clipboard**
3. Files are auto-named: `boston_page_XXXX_corrected.json`

### Step 3: Save

Save exported files to `data/corrections/` folder for:
- Updating curated recipe files
- Retraining the model
- Tracking quality metrics

See [`data/corrections/README.md`](../data/corrections/README.md) for detailed usage.

## Demo Flow (UX)

1. Open `/demo` (defaults to Compare)
2. Pick a featured page
3. Toggle AI overlay, review extracted recipe
4. Compare vs curated; fix lines inline
5. Click "What do I do with this?" for workflow help
6. Export corrected JSON

## Teaching Value

This workflow demonstrates:
- **Human-in-the-loop ML** - Combining automation with human expertise
- **Active learning** - Using corrections to improve future model versions
- **Data quality management** - Systematic approach to fixing extraction errors
