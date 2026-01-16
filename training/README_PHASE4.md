# Phase 4: LayoutLMv3 Fine-tuning

Train LayoutLMv3 for token classification in two stages: high-confidence first, then continue on the full dataset. Inference scripts produce token overlays for the frontend.

## Datasets
- Highconf: `data/datasets/boston_layoutlmv3_highconf/dataset_dict` (auto-regenerated if empty)
- Full: `data/datasets/boston_layoutlmv3_full/dataset_dict`
Each includes `image_path`, `words`, normalized `bboxes` (0–1000), `labels` (ids), `width`, `height`, `page_num`.
Label map: `label_map.json` in each dataset dir. Labels: TITLE, INGREDIENT_LINE, INSTRUCTION_STEP, TIME, TEMP, SERVINGS, O.

## Train
```bash
python -m training.train_layoutlmv3 \
  --highconf_dataset_dir data/_generated/datasets/boston_layoutlmv3_highconf \
  --full_dataset_dir data/_generated/datasets/boston_layoutlmv3_full \
  --output_dir models \
  --stage both \
  --batch_size 2 --eval_batch_size 2 \
  --learning_rate 5e-5 \
  --num_train_epochs_stageA 5 \
  --num_train_epochs_stageB 3 \
  --fp16   # if GPU supports
```
- Stage A trains on highconf; saves to `models/layoutlmv3_boston_stageA_highconf/`
- Stage B continues from Stage A on full; saves to `models/layoutlmv3_boston_stageB_full/`
- Final model saved to `models/layoutlmv3_boston_final/`
- Reports written to `data/reports/training/` (metrics JSON/MD, confusion matrix).
- Use `--stage highconf` or `--stage full` to run a single stage. `--debug` trains on the small dataset.

## Inference
Predict for a page (using final model):
```bash
python -m training.infer.predict_page \
  --model_dir models/layoutlmv3_boston_final \
  --dataset_dir data/datasets/boston_layoutlmv3_full \
  --page_num 38 \
  --out_json data/reports/inference/page_0150_predictions.json
```
Or provide `--image` and `--words_json`.
Render overlay:
```bash
python -m training.infer.render_predictions \
  --pred_json data/reports/inference/page_0150_predictions.json \
  --out_png data/reports/inference/overlays/page_0150_pred_overlay.png
```

## One-command experiment runner
Run smoke test + Stage A + Stage B + inference + weak-label eval, with reports under `data/reports/phase4_runs/<run_id>/`:
```bash
python scripts/run_phase4_experiments.py --config configs/phase4_experiment.yaml
```
Flags: `--run_pass1_only`, `--skip_pass1/2/3`, `--skip_inference`, `--overwrite`, `--run_id`, `--fp16 auto/true/false`, `--eval_pages`, `--infer_page_num auto|N`, `--infer_split validation|train`, `--auto_regen_highconf/--no_auto_regen_highconf`, `--min_highconf_pages N`.
Behaviour:
- If highconf train < 10 and `--auto_regen_highconf`, the runner calls `scripts/regenerate_highconf.py` to relax structural thresholds until a non-empty set is built.
- Inference page auto-selects the first page in the requested split if `--infer_page_num auto`, avoiding missing-page errors.
- REPORT.md summarizes datasets, hyperparams, checkpoints, inference outputs, and weak-label eval (sanity, not gold).

Regenerate highconf only:
```bash
make regen-highconf
# or
python scripts/regenerate_highconf.py --overwrite --min_pages 25
```

## Tips
- Bboxes are already normalized; labels are word-level (NOTE collapsed to O).
- Alignment assigns labels to the first subword only; others get -100; specials get -100.
- Reduce `max_length` or `batch_size` if you hit OOM; use `--gradient_accumulation_steps` to compensate.
- If GPU is absent, training will be slow but functional (disable `--fp16`).  
- Logs go to `data/reports/training/logs/` (HF Trainer output).  
- Frontend overlay JSON includes grouped tokens for AI Parse View.  
