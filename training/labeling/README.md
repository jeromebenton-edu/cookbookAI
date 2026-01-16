# Weak Labeling Pipeline (Phase 3)

This pipeline turns OCR outputs into weak labels for LayoutLMv3 training and evaluates against a small gold set.

## Modules
- `line_grouper.py`: groups OCR tokens into lines using bbox proximity (y clustering via median height; x-sorted within line).
- `spacy_labeler.py`: applies spaCy + heuristic rules to assign line labels and token labels.
- `rules/`: label-specific scoring (ingredients, instructions, metadata/time/temp/servings, title, notes). Signals and scores combine into a final label with confidence via `confidence.py`.
- `labels.py`: canonical label list.
- `run_weak_labeling.py`: CLI to generate weak labels JSONL (and high-confidence subset). Supports `--use_all_pages` to ignore recipe gating.
- `evaluate_against_gold.py`: compares weak labels to a gold JSONL and emits metrics.
- `filter_highconf_structural.py`: Phase 3.75 structural highconf filter (counts titles/ingredients/instructions, O ratio, token conf).
- `select_easy_pages.py`: surfaces easy pages for manual labeling (balanced structure, reasonable confidence).

## Running Weak Labeling

```bash
python -m training.labeling.run_weak_labeling \
  --parquet data/ocr/boston_pages.parquet \
  --recipe_candidates data/ocr/boston_recipe_candidates.csv \
  --out data/labels/boston_weak_labeled.jsonl \
  --out_highconf data/labels/boston_weak_labeled_highconf.jsonl \
  --candidate_threshold 0.65 \
  --min_avg_line_conf 0.80 \
  --max_pages 300 \
  --use_all_pages   # optional: ignore recipe gating
```

Outputs are JSONL records per page with line-level labels, token labels, confidences, and a page quality summary. A debug sample is written to `data/labels/debug_samples/`.

## Phase 3.75 Structural Highconf
Confidence-only gating starved the highconf set. Use the structural filter instead:
```bash
python -m training.labeling.filter_highconf_structural \
  --in_jsonl data/_generated/labels/boston_body_weak.jsonl \
  --out_jsonl data/_generated/labels/boston_body_weak_highconf_structural.jsonl \
  --out_report_md data/_generated/labels/highconf_filter_report.md \
  --out_stats_json data/_generated/labels/highconf_filter_stats.json \
  --min_avg_token_conf 0.60 --max_o_ratio 0.92 --min_ingredient_lines 2 --min_instruction_lines 2 --min_tokens 80
```
This keeps pages with reasonable structure (titles, ingredient/instruction counts), balanced O ratio, and non-trivial labeled token ratio.

## Easy pages for humans
```bash
python -m training.labeling.select_easy_pages \
  --in_jsonl data/_generated/labels/boston_body_weak.jsonl \
  --out_jsonl data/_generated/labels/boston_body_easy_pages.jsonl \
  --out_report_md data/_generated/labels/easy_pages_report.md \
  --max_pages 50
```
Selects pages with 1 title, 2–12 ingredients, 1–8 instructions, decent confidence/O-ratio—ideal for labeling first.

## Gold Evaluation

Create a gold JSONL (copy a weak-labeled page, correct labels) at `data/gold/boston_gold.jsonl`. Then:

```bash
python training/labeling/evaluate_against_gold.py \
  --gold data/gold/boston_gold.jsonl \
  --pred data/labels/boston_weak_labeled.jsonl \
  --out_md data/reports/gold_eval_report.md \
  --out_json data/reports/gold_eval_metrics.json
```

Reports include token/line classification metrics and a list of worst pages by token accuracy. Heuristics (units, verbs, etc.) live in `data/scripts/recipe_signals.py` for easy tuning.

## Tuning Workflow
1) Run weak labeling on a small page set (e.g., 50 pages).
2) Build a gold set for ~10 pages by correcting labels.
3) Iterate rule lists and thresholds; rerun evaluation until F1 improves.
4) Generate a high-confidence subset for model training.
5) TODO: add active learning loop to surface low-confidence pages for manual review.
