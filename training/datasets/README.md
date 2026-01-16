# LayoutLMv3 Dataset Builder (Phase 3.5)

This tool turns weak-labeled pages into Hugging Face `DatasetDict` objects ready for LayoutLMv3 token classification. BBoxes are normalized to 0–1000, NOTE labels are collapsed to O, and deterministic train/val splits are saved alongside label maps and stats.

## Build a dataset
```bash
python -m training.datasets.build_layoutlm_dataset \
  --in_jsonl data/labels/boston_weak_labeled.jsonl \
  --out_dir data/datasets/boston_layoutlmv3_full \
  --val_ratio 0.15 \
  --test_ratio 0.0 \
  --seed 42 \
  --max_pages 0 \
  --min_tokens 40
```

Outputs inside `--out_dir`:
- `dataset_dict/` — HF DatasetDict saved to disk
- `label_map.json` — `label2id`, `id2label` (NOTE removed)
- `splits.json` — page_num lists + seed/ratios
- `stats.json` — pages, token stats, label distribution, drops, %O
- `README.md` — build summary

Label schema lives in `training/datasets/labels.py`:
- TITLE, INGREDIENT_LINE, INSTRUCTION_STEP, TIME, TEMP, SERVINGS, O (NOTE collapsed to O)

## Sanity check overlays
Render sample pages with bboxes to verify normalization and label alignment:
```bash
python -m training.datasets.sanity_check \
  --dataset_dir data/datasets/boston_layoutlmv3_full \
  --num_samples 3 \
  --split train \
  --out_dir data/reports/sanity_overlays
```
This prints the first 25 tokens with labels and saves PNG overlays for non-O regions.

## Why 0–1000 bboxes?
LayoutLM-style models expect normalized coordinates (0–1000) instead of absolute pixels, keeping examples resolution-agnostic and compatible with standard processors.

## Phase 4 TODOs
- Align word labels to subwords via `LayoutLMv3Processor`
- Cache `pixel_values` and attention masks
- Optionally downsample pages dominated by O tokens
