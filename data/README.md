# Data Pipeline (Phase 2.5)

Artifacts for rendering the 1918 **Boston Cooking-School Cook Book** PDF into page images, OCR JSONL, Parquet, and recipe-page candidates for faster labeling.

## Layout

```
data/
  raw/                      # place the PDF here (not committed)
  pages/boston/             # rendered PNGs (0001.png, 0002.png, ...)
  ocr/
    boston_pages.jsonl      # OCR output (one JSON per line)
    boston_pages.parquet    # Parquet with derived features
    boston_recipe_candidates.{csv,jsonl,md}
  scripts/                  # CLI utilities
```

## Install

Use a dedicated environment for the data pipeline:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r data/requirements.txt
```

Install the Tesseract binary (Ubuntu: `sudo apt-get install tesseract-ocr`).

## PDF Placement

Download the 1918 PDF (public domain) and place it at `data/raw/boston-cooking-school-1918.pdf`.

## Render PDF to PNGs

```bash
python data/scripts/render_pdf_pages.py \
  --pdf data/raw/boston-cooking-school-1918.pdf \
  --out data/pages/boston \
  --dpi 300 \
  --start 1 --end 50
```

## OCR to JSONL

```bash
python data/scripts/ocr_pages.py \
  --images_dir data/pages/boston \
  --out data/ocr/boston_pages.jsonl \
  --ocr_backend tesseract \
  --lang eng \
  --max_pages 50 \
  --preprocess \
  --include_text
```

## Convert JSONL to Parquet

```bash
python data/scripts/jsonl_to_parquet.py \
  --in data/ocr/boston_pages.jsonl \
  --out data/ocr/boston_pages.parquet
```

## Detect Recipe Pages

```bash
python data/scripts/detect_recipe_pages.py \
  --in data/ocr/boston_pages.parquet \
  --out_csv data/ocr/boston_recipe_candidates.csv \
  --out_jsonl data/ocr/boston_recipe_candidates.jsonl \
  --out_md data/ocr/boston_recipe_candidates_top50.md \
  --threshold 0.65
```

Outputs include a score (0–1), boolean `likely` based on threshold, detected signals, and an excerpt. Heuristics rely on units, cooking verbs, time/temp patterns, numeric density, short lines, and penalties for TOC/index patterns or low token counts. Tune keyword lists in `data/scripts/recipe_signals.py`.

## Weak Labeling (Phase 3)

Generate weak labels and a high-confidence subset:

```bash
python training/labeling/run_weak_labeling.py \
  --parquet data/ocr/boston_pages.parquet \
  --recipe_candidates data/ocr/boston_recipe_candidates.csv \
  --out data/labels/boston_weak_labeled.jsonl \
  --out_highconf data/labels/boston_weak_labeled_highconf.jsonl \
  --candidate_threshold 0.65 \
  --min_avg_line_conf 0.80
```

Evaluate against a gold set:

```bash
python training/labeling/evaluate_against_gold.py \
  --gold data/gold/boston_gold.jsonl \
  --pred data/labels/boston_weak_labeled.jsonl \
  --out_md data/reports/gold_eval_report.md \
  --out_json data/reports/gold_eval_metrics.json
```

## Sync PNGs to Frontend

```bash
python data/scripts/sync_pages_to_frontend.py \
  --src data/pages/boston \
  --dst frontend/public/recipes/boston/pages \
  --mode copy
```

## Option 2 Pipeline (recipes only, larger range)

Use the runner to expand the corpus, weak-label, and build LayoutLMv3 datasets:

```
python scripts/run_option2_pipeline.py --config configs/option2_pipeline.yaml
```

Defaults (configurable): pages 120–420, dpi 300, candidate_threshold 0.30, min_avg_line_conf 0.65, max_pages 400, val_ratio 0.15, seed 42, min_tokens 40.

Outputs under `data/_generated/`:
- pages: `data/_generated/pages/boston_body/`
- ocr: `data/_generated/ocr/boston_body.jsonl` + `.parquet`
- candidates: `data/_generated/ocr/boston_recipe_candidates.csv` (+ top50 markdown)
- weak labels: `data/_generated/labels/boston_body_weak.jsonl` and structural highconf `_highconf_structural.jsonl`
- datasets: `data/_generated/datasets/boston_layoutlmv3_full/` and `_highconf/`
- sanity overlays: `data/reports/sanity_overlays/`

Prereqs: PyMuPDF, Tesseract binary (`sudo apt-get install tesseract-ocr`), `.venv` with `training/requirements.txt`.

### Phase 3.75 highconf
- Recipe detection is used for ranking; weak labeling can run on all pages (`--use_all_pages`).
- Structural filter keeps pages with reasonable structure (titles, ingredient/instruction counts), balanced O ratio, and token confidence. Tune thresholds in `configs/option2_pipeline.yaml`.
