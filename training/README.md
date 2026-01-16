# Training Scaffolding

This directory is a placeholder for the labeling, dataset, and fine-tuning workflow. The scripts are intentionally minimal to keep the portfolio focused while providing a clear path to expand.

## Structure

- `labeling/` - Build line-level labels from OCR and rules.
- `datasets/` - Convert labeled data into LayoutLMv3 training format (Phase 3.5 builder, sanity overlays).
- `train.py` / `eval.py` - Entry points for fine-tuning and evaluation.

## TODO

- Finalize labeling heuristics.
- Add gold dataset and metrics.
- Implement training configs and checkpoints.

## Option 2 pipeline (Phase 3.5 datasets)
End-to-end runner builds expanded weak labels and LayoutLMv3 DatasetDicts (NOTE -> O, 0–1000 bboxes):
```
python scripts/run_option2_pipeline.py --config configs/option2_pipeline.yaml
```
Outputs land in `data/_generated/datasets/` (full + highconf) with label maps, splits, stats, and sanity overlays.
