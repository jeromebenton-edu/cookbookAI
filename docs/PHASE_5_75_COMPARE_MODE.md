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
- Export buttons: download/copy corrected JSON (includes AI original + metadata).

## How it works (diff)
- Lines normalized (lowercase, strip punctuation, collapse spaces).
- Similarity via bag-of-words overlap.
- Thresholds: ingredients 0.75, instructions 0.70.
- Remaining AI lines = extras; missing curated lines flagged.

## Export JSON contents
```json
{
  "page_num": 16,
  "corrected": { "title": "...", "ingredients": [...], "instructions": [...] },
  "ai_original": { ...full AI response... },
  "meta": { "model": "layoutlmv3", "created_at": "..." }
}
```
This can seed future training/active-learning loops (Phase 6).

## Demo flow (UX)
1) Open `/demo` (defaults to Compare).
2) Pick a featured page.
3) Toggle AI overlay, review extracted recipe.
4) Compare vs curated; fix lines inline.
5) Export corrected JSON.
