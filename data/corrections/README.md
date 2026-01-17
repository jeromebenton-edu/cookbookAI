# Recipe Corrections

This folder stores human-corrected recipe extractions exported from the Compare Mode UI.

## Workflow

1. **Open a recipe** in the frontend at `/recipes/[recipe-id]`
2. **Switch to Compare mode** using the toggle buttons
3. **Review AI extractions** against the curated data
4. **Edit AI fields** to fix errors (click fields to edit)
5. **Export corrections** using the "Export Corrections" dropdown
6. **Save to this folder** with the auto-generated filename

## File Naming Convention

Files are named: `boston_page_XXXX_corrected.json`

Where `XXXX` is the zero-padded page number (e.g., `boston_page_0079_corrected.json`).

## JSON Structure

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

## Using Corrections

### Update Curated Recipes

Use corrections to improve the curated recipe JSON files:

```bash
# Example: Apply correction to curated recipe
python scripts/apply_correction.py data/corrections/boston_page_0079_corrected.json
```

### Retrain the Model

Corrections can be used as additional training data:

```bash
# Convert corrections to training format
python scripts/corrections_to_training.py data/corrections/ data/training/corrections_batch/
```

### Quality Metrics

Track correction rates to measure model performance:

```bash
# Analyze correction patterns
python scripts/analyze_corrections.py data/corrections/
```

## Teaching Value

This corrections workflow demonstrates:

- **Human-in-the-loop ML** - Combining automation with human expertise
- **Active learning** - Using corrections to improve future model versions
- **Data quality management** - Systematic approach to fixing extraction errors
- **Feedback loops** - Closing the loop between prediction and ground truth

## Notes

- Corrections are additive - keep both the AI original and corrected versions
- The `meta` field tracks which model version produced the extraction
- Confidence scores help prioritize which pages need review
