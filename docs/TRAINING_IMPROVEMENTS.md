# Training Pipeline Improvements

## Summary

Improved the training pipeline to add validation splits, per-epoch metrics, collapse detection, and weighted loss to prevent silent training failures.

## Changes Made

### A) Validation Split Creation

**File: `training/train_layoutlmv3.py`**

Added automatic validation split creation:

```python
def create_validation_split(ds_dict: DatasetDict, val_ratio: float = 0.1, seed: int = 42) -> DatasetDict:
    """
    Create a validation split from train if one doesn't exist.
    Returns a new DatasetDict with train + validation splits.
    """
    if "validation" in ds_dict or "val" in ds_dict or "eval" in ds_dict:
        LOG.info("Validation split already exists, skipping creation")
        return ds_dict

    if "train" not in ds_dict:
        LOG.warning("No train split found, cannot create validation split")
        return ds_dict

    LOG.info(f"Creating {val_ratio:.0%} validation split from train (seed={seed})")
    train_ds = ds_dict["train"]

    # Split the train dataset
    split_ds = train_ds.train_test_split(test_size=val_ratio, seed=seed)

    # Create new DatasetDict
    new_dict = DatasetDict({
        "train": split_ds["train"],
        "validation": split_ds["test"]
    })

    LOG.info(f"Split complete: train={len(new_dict['train'])}, validation={len(new_dict['validation'])}")
    return new_dict
```

**Features:**
- Automatically creates 90/10 train/validation split (configurable via `--val_ratio`)
- Uses deterministic seed (default: 42)
- Saves updated dataset with validation split to disk
- Skips if validation split already exists

### B) Collapse Detection Callback

Added `CollapseDetectionCallback` to monitor training:

```python
class CollapseDetectionCallback(TrainerCallback):
    """
    Callback to detect label collapse during training.
    Logs predicted label distribution and warns if any label dominates.
    """

    def __init__(self, label_list: List[str], collapse_threshold: float = 0.9,
                 patience: int = 2, stage_name: str = "train"):
        self.label_list = label_list
        self.collapse_threshold = collapse_threshold
        self.patience = patience
        self.stage_name = stage_name
        self.collapse_epochs = 0
        self.epoch_metrics = []

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        # Computes:
        # 1. Predicted label distribution on validation set
        # 2. Per-label precision/recall/F1
        # 3. Top confusion pairs
        # 4. Warns if any label >90% of predictions
        # 5. Stores metrics for summary report
```

**Per-Epoch Logging:**
```
================================================================================
Epoch 1 Complete - Computing Validation Metrics
================================================================================

Predicted Label Distribution (validation set):
------------------------------------------------------------
  O                        :   5432 (54.32%)
  INGREDIENT_LINE          :   2014 (20.14%)
  INSTRUCTION_STEP         :   1523 (15.23%)
  TITLE                    :    789 ( 7.89%)
  ...

✓ Label distribution healthy (max: O at 54.3%)

Per-Label Metrics:
--------------------------------------------------------------------------------
Label                       Precision     Recall         F1    Support
--------------------------------------------------------------------------------
INGREDIENT_LINE                 0.892      0.845      0.868       2100
INSTRUCTION_STEP                0.823      0.791      0.807       1600
TITLE                           0.945      0.912      0.928        850
...

Top Confusions:
------------------------------------------------------------
  INGREDIENT_LINE → O: 142 times
  INSTRUCTION_STEP → O: 89 times
  O → INGREDIENT_LINE: 67 times
================================================================================
```

**Collapse Warning:**
```
⚠️  COLLAPSE WARNING: Label 'O' dominates with 92.3% of predictions
   Collapse detected for 2 consecutive epoch(s)

❌ TRAINING FAILURE: Label collapse persisted for 2 epochs
   Training may have collapsed. Consider:
   - Reducing learning rate
   - Using weighted loss
   - Checking label distribution in training data
```

### C) Weighted Loss

Added class-weighted CrossEntropyLoss:

```python
def compute_class_weights(ds_dict: DatasetDict, label2id: dict, ignore_index: int = -100) -> torch.Tensor:
    """
    Compute class weights from training label distribution.
    Returns tensor of weights for use in CrossEntropyLoss.
    """
    LOG.info("Computing class weights from training labels")
    train_ds = ds_dict["train"]

    # Count all labels
    label_counts = Counter()
    if "ner_tags" in train_ds.column_names:
        for example in train_ds:
            tags = example["ner_tags"]
            for tag in tags:
                if tag != ignore_index:
                    label_counts[tag] += 1

    # Convert to weights (inverse frequency)
    num_labels = len(label2id)
    weights = torch.ones(num_labels)
    total_count = sum(label_counts.values())

    for label_id, count in label_counts.items():
        if 0 <= label_id < num_labels:
            # Inverse frequency with smoothing
            weights[label_id] = total_count / (num_labels * count)

    # Normalize weights
    weights = weights / weights.sum() * num_labels

    LOG.info(f"Class weights: min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}")
    return weights
```

**Integration:**
```python
# Apply class weights to model
if class_weights is not None and args.use_weighted_loss:
    LOG.info("Using weighted loss with class weights")
    class_weights = class_weights.to(model.device)

    # Wrap forward to use weighted loss
    def forward_with_weighted_loss(*args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        if outputs.loss is not None and "labels" in kwargs:
            logits = outputs.logits
            labels = kwargs["labels"]
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            outputs.loss = loss
        return outputs

    model.forward = forward_with_weighted_loss
```

### D) Training Summary Export

Training metrics are now saved to `docs/debug/train_metrics_{stage}.json`:

```json
{
  "stage": "stageB",
  "timestamp": "2026-01-10T12:34:56.789",
  "model_dir": "models/layoutlmv3_boston_stageB_full",
  "dataset_splits": {
    "train": 270,
    "validation": 30
  },
  "final_metrics": {
    "precision": 0.892,
    "recall": 0.845,
    "f1": 0.868,
    "report": {...}
  },
  "epoch_metrics": [
    {
      "epoch": 1,
      "pred_distribution": {
        "O": {"count": 5432, "percentage": 54.32},
        "INGREDIENT_LINE": {"count": 2014, "percentage": 20.14},
        ...
      },
      "max_label": "O",
      "max_percentage": 0.5432,
      "metrics": {...},
      "collapse_warning": false
    },
    ...
  ],
  "training_args": {
    "learning_rate": 5e-05,
    "batch_size": 2,
    "num_epochs": 3,
    "seed": 42,
    "use_weighted_loss": true
  }
}
```

### E) Health Endpoint Update

**File: `backend/app/services/layoutlm_service.py`**

Updated health endpoint to report validation split:

```python
def health(self) -> dict:
    # ... existing code ...
    has_validation_split = False
    validation_size = 0
    if isinstance(self._dataset, DatasetDict):
        dataset_sizes = {k: len(v) for k, v in self._dataset.items()}
        has_validation_split = "validation" in self._dataset or "val" in self._dataset or "eval" in self._dataset
        if "validation" in self._dataset:
            validation_size = len(self._dataset["validation"])
        elif "val" in self._dataset:
            validation_size = len(self._dataset["val"])
        elif "eval" in self._dataset:
            validation_size = len(self._dataset["eval"])

    return {
        # ... existing fields ...
        "dataset_sizes": dataset_sizes,
        "has_validation_split": has_validation_split,
        "validation_size": validation_size,
        # ...
    }
```

**Health Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "dataset_loaded": true,
  "dataset_splits": ["train", "validation"],
  "dataset_sizes": {
    "train": 270,
    "validation": 30
  },
  "has_validation_split": true,
  "validation_size": 30,
  ...
}
```

## New Command-Line Arguments

```bash
# Validation split ratio (default: 0.1 = 10%)
--val_ratio 0.1

# Enable/disable weighted loss (default: enabled)
--use_weighted_loss         # Enable (default)
--no_weighted_loss          # Disable
```

## Usage

### Basic Training (with all improvements)
```bash
python training/train_layoutlmv3.py \
  --stage both \
  --batch_size 2 \
  --learning_rate 5e-5 \
  --num_train_epochs_stageA 5 \
  --num_train_epochs_stageB 3
```

### Training with Custom Validation Split
```bash
python training/train_layoutlmv3.py \
  --stage full \
  --val_ratio 0.15  # 15% validation
```

### Training without Weighted Loss
```bash
python training/train_layoutlmv3.py \
  --stage both \
  --no_weighted_loss
```

## Acceptance Criteria Status

✅ **A) Validation Split**
- [x] Creates 90/10 split from train if none exists
- [x] Deterministic seed (default: 42)
- [x] Saves dataset_dict with train + validation
- [x] Health endpoint reports validation size

✅ **B) Training Instrumentation**
- [x] Per-epoch label distribution on validation
- [x] Per-label precision/recall/F1
- [x] Confusion matrix summary (top confusions)
- [x] Warns if any label >90% of predictions
- [x] Tracks collapse across epochs with patience

✅ **C) Weighted Loss**
- [x] Computes class weights from training label counts
- [x] Uses weighted CrossEntropyLoss(ignore_index=-100)
- [x] Enabled by default, can be disabled with --no_weighted_loss

✅ **D) Training Logs**
- [x] Writes training summary to docs/debug/train_metrics_{stage}.json
- [x] Includes epoch-by-epoch metrics
- [x] Includes predicted distributions
- [x] Includes collapse warnings

✅ **E) Label Sanity Checks** (Added 2026-01-10)
- [x] Standalone diagnostic script: scripts/sanity_check_encoded_labels.py
- [x] Analyzes encoded examples for usable supervision signal
- [x] Checks: masked ratio, non-O ratio, examples with labels
- [x] Integrated into training with --run_sanity_checks flag
- [x] Makefile target: make sanity-labels
- [x] Exit code 1 if checks fail (prevents wasted GPU time)

## Files Modified

1. **training/train_layoutlmv3.py** (~990 lines, +377 added)
   - Added validation split creation
   - Added CollapseDetectionCallback
   - Added weighted loss computation
   - Added training summary export
   - Added run_label_sanity_check() function (lines 34-156)
   - Added --run_sanity_checks CLI argument
   - Integrated sanity checks into training flow (lines 804-828)

2. **scripts/sanity_check_encoded_labels.py** (NEW file, ~352 lines)
   - Standalone diagnostic script for label validation
   - Comprehensive analysis and reporting
   - Fail-fast thresholds for training data quality

3. **backend/app/services/layoutlm_service.py** (+11 lines)
   - Enhanced health endpoint to report validation split info

4. **Makefile** (+7 lines)
   - Added sanity-labels target for running diagnostic script

## Testing

The training pipeline can now be run end-to-end with:

```bash
# Test with existing dataset
python training/train_layoutlmv3.py --stage full --debug

# Check health endpoint
curl http://localhost:8000/api/parse/health | jq '.has_validation_split, .validation_size'

# Review training metrics
cat docs/debug/train_metrics_stageB.json | jq '.epoch_metrics[] | {epoch, max_label, max_percentage, collapse_warning}'
```

## Benefits

1. **Prevents Silent Collapse**: Per-epoch monitoring catches training failures immediately
2. **Better Generalization**: Validation split ensures models are evaluated on held-out data
3. **Class Imbalance Handling**: Weighted loss helps with minority classes (INGREDIENT_LINE, INSTRUCTION_STEP, etc.)
4. **Transparency**: Complete training logs with predicted distributions and per-label metrics
5. **Debuggability**: Confusion matrix shows where model struggles most
6. **Reproducibility**: Deterministic splits with fixed seed

## E) Label Sanity Checks (Added 2026-01-10)

Added diagnostic tooling to validate encoded labels before training:

**Standalone Script: `scripts/sanity_check_encoded_labels.py`**
```bash
python scripts/sanity_check_encoded_labels.py \
  --dataset-dir data/datasets/boston_layoutlmv3_full/dataset_dict \
  --num-samples 20 \
  --verbose
```

**Features:**
- Analyzes encoded examples to verify usable supervision signal
- Checks: masked ratio (<95%), non-O ratio (>1%), examples with labels (>10%)
- Per-example and aggregate statistics
- Exit code 1 if checks fail

**Training Integration:**
Added `--run_sanity_checks` flag to training script:
```bash
python training/train_layoutlmv3.py \
  --stage full \
  --run_sanity_checks  # Runs checks before training starts
```

When enabled:
1. After dataset encoding, runs sanity check on 20 samples
2. Validates: masked tokens <95%, non-O labels >1%, examples with non-O >10%
3. If checks fail, prints detailed error and exits with code 1
4. Prevents wasted GPU time on broken training data

**Makefile Target:**
```bash
make sanity-labels  # Run standalone diagnostic script
```

**Example Output:**
```
================================================================================
AGGREGATE STATISTICS
================================================================================
Total examples analyzed:  20
Total tokens:             10240
Masked tokens:            3072 (30.0%)
Unmasked tokens:          7168
Non-O tokens:             1843 (25.7% of unmasked)

Global label distribution (unmasked tokens):
  O                   :   5325 ( 74.3%)
  INGREDIENT_LINE     :    892 ( 12.4%)
  INSTRUCTION_STEP    :    651 (  9.1%)
  TITLE               :    300 (  4.2%)

Examples with specific labels:
  INGREDIENT_LINE:   85.0% (17/20)
  INSTRUCTION_STEP:  75.0% (15/20)
  TITLE:             45.0% ( 9/20)
  Any non-O:         95.0% (19/20)

================================================================================
✓ SANITY CHECK PASSED
================================================================================
```

## Next Steps

After restarting servers:
1. Run label sanity checks: `make sanity-labels`
2. Run training with `make train` or equivalent (optionally with `--run_sanity_checks`)
3. Monitor console output for per-epoch metrics
4. Check `docs/debug/train_metrics_*.json` for detailed analysis
5. Verify health endpoint shows validation split
6. Inspect label distributions for collapse warnings
