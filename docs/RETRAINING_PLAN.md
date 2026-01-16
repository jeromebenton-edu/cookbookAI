# Phase 2: Retrain Model with Class Balancing

## Goal
Improve RECIPE_TITLE detection accuracy from 0% to 40-60% by using weighted loss during training.

## Current Problem
- RECIPE_TITLE: Only 1.8% of training tokens
- Model ignores this rare class in favor of dominant classes (O, INGREDIENT, INSTRUCTION)

## Solution: Weighted Cross-Entropy Loss

### Step 1: Calculate Class Weights

Based on inverse frequency:
```python
Label Distribution:
  O:                67.59% → weight = 1.0   (baseline)
  INGREDIENT_LINE:  15.81% → weight = 4.3
  INSTRUCTION_STEP: 12.60% → weight = 5.4
  PAGE_HEADER:       2.13% → weight = 31.7
  RECIPE_TITLE:      1.80% → weight = 37.5  ⚠️ KEY FIX
  SECTION_HEADER:    0.03% → weight = 2253
  SERVINGS:          0.03% → weight = 2253
  TEMP:              0.02% → weight = 3380
```

### Step 2: Modify Training Script

Find your training script (likely in `tools/` or `scripts/`) and add:

```python
import torch
import torch.nn as nn

# Define class weights
class_weights = torch.tensor([
    1.0,    # O
    4.3,    # INGREDIENT_LINE
    5.4,    # INSTRUCTION_STEP
    31.7,   # PAGE_HEADER
    37.5,   # RECIPE_TITLE (30-40x boost!)
    2253,   # SECTION_HEADER
    2253,   # SERVINGS
    3380,   # TEMP
], dtype=torch.float32).to(device)

# Use weighted loss in training loop
criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

# In training loop:
for batch in train_dataloader:
    outputs = model(**batch)
    loss = criterion(
        outputs.logits.view(-1, num_labels),
        batch['labels'].view(-1)
    )
    loss.backward()
    optimizer.step()
```

### Step 3: Training Parameters

```python
training_args = TrainingArguments(
    output_dir="models/layoutlmv3_v3_manual_balanced_v2",
    num_train_epochs=15,  # Increase epochs
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=5e-5,  # Standard for LayoutLMv3
    warmup_ratio=0.1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_RECIPE_TITLE",  # Focus on title F1
    logging_steps=50,
)
```

### Step 4: Custom Metrics

Track per-class F1 scores to verify RECIPE_TITLE improves:

```python
from sklearn.metrics import classification_report

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Flatten and filter padding
    labels_flat = labels.flatten()
    preds_flat = preds.flatten()
    mask = labels_flat != -100

    labels_flat = labels_flat[mask]
    preds_flat = preds_flat[mask]

    # Per-class metrics
    report = classification_report(
        labels_flat,
        preds_flat,
        target_names=id2label.values(),
        output_dict=True,
        zero_division=0
    )

    return {
        "eval_f1_overall": report["weighted avg"]["f1-score"],
        "eval_f1_RECIPE_TITLE": report.get("RECIPE_TITLE", {}).get("f1-score", 0),
        "eval_f1_INGREDIENT": report.get("INGREDIENT_LINE", {}).get("f1-score", 0),
        "eval_f1_INSTRUCTION": report.get("INSTRUCTION_STEP", {}).get("f1-score", 0),
    }
```

## Expected Results

### Before (Current Model)
```
RECIPE_TITLE:
  Precision: ~10%
  Recall: ~5%
  F1: ~7%
```

### After (With Class Weights)
```
RECIPE_TITLE:
  Precision: 50-70%
  Recall: 40-60%
  F1: 45-65%
```

This would increase perfect match rate from 0% to 40-60% on held-out test set.

## Execution Plan

### Tomorrow (2-3 hours):

1. **Find training script** (10 min)
   ```bash
   find tools/ scripts/ -name "*train*.py" -o -name "*finetune*.py"
   ```

2. **Add class weights** (30 min)
   - Modify loss function
   - Add custom metrics
   - Update training args

3. **Start training** (2-3 hours GPU time)
   ```bash
   python tools/train_layoutlmv3.py \
     --model_name microsoft/layoutlmv3-base \
     --dataset_path data/datasets/boston_layoutlmv3_v3_manual_expanded2/dataset_dict \
     --output_dir models/layoutlmv3_v3_balanced_titles \
     --num_epochs 15 \
     --use_class_weights
   ```

4. **Evaluate results** (30 min)
   ```bash
   python tools/evaluate_title_accuracy.py 50
   ```

5. **If improved, re-parse** (5 min)
   ```bash
   python tools/parse_full_cookbook.py
   ```

## Validation

Compare before/after on 20 test pages:
```bash
# Before (current model)
python tools/evaluate_title_accuracy.py 20
# Expected: 0% perfect, 5% partial, 95% wrong

# After (retrained model)
python tools/evaluate_title_accuracy.py 20
# Target: 40-60% perfect, 20-30% partial, 10-20% wrong
```

## Rollback Plan

If retraining makes things worse:
1. Keep using current heuristic fallback
2. Original model still available at: `models/layoutlmv3_v3_manual_59pages_balanced/`
3. Can always revert in `tools/parse_full_cookbook.py`

## Next Steps After Retraining

1. **If successful (>40% accuracy)**:
   - Deploy new model
   - Re-parse all recipes
   - Document improvement

2. **If mediocre (20-40% accuracy)**:
   - Keep heuristic + model hybrid
   - Add more training examples
   - Consider architecture changes

3. **If failed (<20% accuracy)**:
   - Stick with heuristic only
   - Investigate data quality issues
   - Consider different model architecture (e.g., Donut, TrOCR)

## Files Needed

- Training script: `tools/train_*.py` or `scripts/train_*.py`
- Dataset: `data/datasets/boston_layoutlmv3_v3_manual_expanded2/dataset_dict`
- Evaluation: `tools/evaluate_title_accuracy.py` (already created)
- Config: Check for `config.yaml` or `training_config.json`
