# Validation Split Improvements

## Problem Statement

The training pipeline had a critical issue where validation splits existed but were empty (0 examples), causing:
- Validation metrics to fail silently
- Collapse detection callbacks to crash
- Training summaries to be incomplete
- Health endpoint reporting misleading information

**Observed Issue:**
```json
{
  "has_validation_split": true,
  "validation_size": 0,
  "dataset_sizes": {"train": 614, "val": 0, "test": 0}
}
```

## Solution Overview

Implemented a comprehensive fix with:
1. **Reliable validation split creation** with guardrails
2. **Persistent storage** with atomic saves and backups
3. **Hard training guardrails** that fail fast if validation is empty
4. **Enhanced health monitoring** with degraded status on empty validation
5. **Standalone script** for easy fixing: `make ensure-validation-split`

## Implementation Details

### A) Enhanced Validation Split Creation

**File: `training/train_layoutlmv3.py:41-129`**

**Key Improvements:**
- Checks all common validation split names (`validation`, `val`, `eval`)
- Detects and recreates empty validation splits
- Verifies dataset has ≥10 examples before splitting
- Ensures splits are non-empty and disjoint
- Raises `RuntimeError` if validation is empty after creation

```python
def create_validation_split(ds_dict: DatasetDict, val_ratio: float = 0.1, seed: int = 42) -> DatasetDict:
    """
    Create a validation split from train if one doesn't exist.

    Raises:
        RuntimeError: If validation split cannot be created or is empty
    """
    # Check for existing validation split
    existing_val_split = None
    if "validation" in ds_dict:
        existing_val_split = "validation"
    elif "val" in ds_dict:
        existing_val_split = "val"
    elif "eval" in ds_dict:
        existing_val_split = "eval"

    # If validation exists and is non-empty, we're good
    if existing_val_split:
        val_size = len(ds_dict[existing_val_split])
        if val_size > 0:
            LOG.info(f"Validation split '{existing_val_split}' already exists with {val_size} examples")
            return ds_dict
        else:
            LOG.warning(f"Validation split '{existing_val_split}' exists but is EMPTY, will recreate")
            # Remove empty split and recreate
            ...

    # Verify dataset size
    if train_size < 10:
        raise RuntimeError(f"Training set too small ({train_size} examples)")

    # Create split
    split_ds = train_ds.train_test_split(test_size=val_ratio, seed=seed)

    # Verify non-empty
    if val_new == 0:
        raise RuntimeError("Validation split is empty after creation!")

    return new_dict
```

**Logs:**
```
Creating 10% validation split from train (seed=42)
  Train size before split: 614
✓ Split complete: train=552, validation=62
  Validation ratio: 10.1% of original train set
```

### B) Persistent Storage with Verification

**File: `training/train_layoutlmv3.py:132-186`**

**Key Improvements:**
- Verifies validation exists and is non-empty before saving
- Creates timestamped backup of original dataset
- Verifies save succeeded by reloading
- Raises `RuntimeError` if validation becomes empty after save

```python
def save_dataset_with_validation(ds_dict: DatasetDict, dataset_dir: Path) -> None:
    """
    Save the dataset with validation split to disk.

    Raises:
        RuntimeError: If validation split is missing or empty
    """
    # Verify validation split
    if "validation" not in ds_dict:
        raise RuntimeError("Cannot save dataset: no 'validation' split found")

    if len(ds_dict["validation"]) == 0:
        raise RuntimeError("Cannot save dataset: validation split is EMPTY")

    # Create backup
    if save_path.exists():
        backup_path = save_path.parent / f"dataset_dict.backup.{int(time.time())}"
        LOG.info(f"  Creating backup: {backup_path}")
        shutil.move(str(save_path), str(backup_path))

    # Save dataset
    ds_dict.save_to_disk(str(save_path))

    # Verify reload
    loaded = load_from_disk(str(save_path))
    if len(loaded["validation"]) == 0:
        raise RuntimeError("Validation split is empty after reload!")

    LOG.info(f"✓ Dataset saved and verified: validation_size={loaded_val_size}")
```

### C) Hard Training Guardrails

**File: `training/train_layoutlmv3.py:496-532`**

**Key Improvements:**
- Checks validation split before training starts
- Fails immediately with actionable error message
- Normalizes split names (`val`/`eval` → `validation`)
- Logs validation status

```python
# HARD GUARDRAIL: Ensure validation split is non-empty before training
if isinstance(ds, DatasetDict):
    val_key = "validation" if "validation" in ds else ("val" if "val" in ds else ("eval" if "eval" in ds else None))

    if val_key is None:
        raise RuntimeError(
            f"❌ TRAINING FAILED: No validation split found in dataset.\n"
            f"   Available splits: {list(ds.keys())}\n"
            f"   Validation split is REQUIRED for collapse detection and metrics.\n"
            f"   → Run: make ensure-validation-split\n"
            f"   → Or rebuild dataset: make rebuild-dataset"
        )

    val_size = len(ds[val_key])
    if val_size == 0:
        raise RuntimeError(
            f"❌ TRAINING FAILED: Validation split '{val_key}' is EMPTY (0 examples).\n"
            f"   This will break validation metrics and collapse detection.\n"
            f"   → Run: make ensure-validation-split\n"
            f"   → Or rebuild dataset: make rebuild-dataset"
        )

    LOG.info(f"✓ Validation guardrail passed: train={train_size}, {val_key}={val_size}")
```

**Error Message Example:**
```
❌ TRAINING FAILED: Validation split 'val' is EMPTY (0 examples).
   This will break validation metrics and collapse detection.
   → Run: make ensure-validation-split
   → Or rebuild dataset: make rebuild-dataset
```

### D) Enhanced Health Endpoint

**File: `backend/app/services/layoutlm_service.py:113-172`**

**New Fields:**
- `validation_nonempty`: Boolean flag indicating non-empty validation
- `validation_key`: Name of the validation split (`validation`/`val`/`eval`)
- **Status degradation**: Sets `status="degraded"` if validation exists but is empty

```python
def health(self) -> dict:
    # ...
    validation_nonempty = False
    validation_key = None

    if isinstance(self._dataset, DatasetDict):
        # Check for validation split
        if "validation" in self._dataset:
            validation_key = "validation"
            validation_size = len(self._dataset["validation"])
        elif "val" in self._dataset:
            validation_key = "val"
            validation_size = len(self._dataset["val"])
        elif "eval" in self._dataset:
            validation_key = "eval"
            validation_size = len(self._dataset["eval"])

        has_validation_split = validation_key is not None
        validation_nonempty = validation_size > 0

        # Degrade status if validation exists but is empty
        if has_validation_split and not validation_nonempty:
            status = "degraded"
            self.errors["validation_empty"] = (
                f"Validation split '{validation_key}' exists but is EMPTY (0 examples). "
                f"This will break training metrics. Run: make ensure-validation-split"
            )

    return {
        # ... existing fields ...
        "validation_nonempty": validation_nonempty,
        "validation_key": validation_key,
        # ...
    }
```

**Health Response (Healthy):**
```json
{
  "status": "ok",
  "validation_nonempty": true,
  "validation_size": 62,
  "validation_key": "validation",
  "dataset_sizes": {
    "train": 552,
    "validation": 62,
    "test": 0
  }
}
```

**Health Response (Degraded - Empty Validation):**
```json
{
  "status": "degraded",
  "validation_nonempty": false,
  "validation_size": 0,
  "validation_key": "val",
  "errors": {
    "validation_empty": "Validation split 'val' exists but is EMPTY (0 examples). This will break training metrics. Run: make ensure-validation-split"
  }
}
```

### E) Standalone Validation Script

**File: `scripts/ensure_validation_split.py` (242 lines)**

**Purpose:** Standalone script to create/fix validation splits without requiring training dependencies.

**Features:**
- Checks dataset for validation split
- Creates 90/10 split if missing or empty
- Atomic save with backup using temp files
- Comprehensive verification
- No dependencies on training modules (standalone)

**Usage:**
```bash
python scripts/ensure_validation_split.py \
  --dataset-dir data/datasets/boston_layoutlmv3_full \
  --val-ratio 0.1 \
  --seed 42 \
  [--force]
```

**Output:**
```
Loading dataset from data/datasets/boston_layoutlmv3_full/dataset_dict
Current splits: train=614, val=0, test=0
❌ Validation split 'val' exists but is EMPTY (0 examples)

Creating 10% validation split (seed=42)...
✓ Split complete: train=552, validation=62

Saving dataset with validation split...
  Saving to temporary location: ...dataset_dict.new.1768067339
  Temp save verified: validation_size=62
  Creating backup: ...dataset_dict.backup.1768067339
  Moving to final location: ...dataset_dict
✓ Dataset saved and verified: validation_size=62

✓ Success! Dataset now has validation split:
  train=552, validation=62, test=0

Verify with: curl http://localhost:8000/api/parse/health | jq '.validation_size'
```

**Key Implementation Details:**

1. **Atomic Save with Temp Files:**
```python
# Save to temporary location first
temp_path = temp_parent / f"dataset_dict.new.{int(time.time())}"
ds_dict.save_to_disk(str(temp_path))

# Verify temp save
loaded_temp = load_from_disk(str(temp_path))
if len(loaded_temp["validation"]) == 0:
    raise RuntimeError("Validation split is empty in temp save!")

# Create backup
backup_path = temp_parent / f"dataset_dict.backup.{int(time.time())}"
shutil.move(str(save_path), str(backup_path))

# Move temp to final location (atomic)
shutil.move(str(temp_path), str(save_path))
```

2. **Comprehensive Verification:**
```python
# Final verification
loaded = load_from_disk(str(save_path))
loaded_val_size = len(loaded["validation"]) if "validation" in loaded else 0
if loaded_val_size == 0:
    raise RuntimeError("Validation split is empty after final save!")
```

### F) Makefile Integration

**File: `Makefile`**

**New Target:**
```makefile
ensure-validation-split:
> set -euo pipefail
> echo "Ensuring dataset has non-empty validation split..."
> python scripts/ensure_validation_split.py --dataset-dir data/datasets/boston_layoutlmv3_full --val-ratio 0.1 --seed 42
```

**Usage:**
```bash
# Fix validation split
make ensure-validation-split

# Verify it worked
curl http://localhost:8000/api/parse/health | jq '.validation_size'
```

### G) New Training Argument

**File: `training/train_layoutlmv3.py:577`**

```python
parser.add_argument("--force_resplit", action="store_true",
                   help="Force recreation of validation split even if one exists")
```

**Usage:**
```bash
# Force recreate validation split during training
python training/train_layoutlmv3.py --force_resplit
```

## Workflow

### Before (Broken State)

1. Dataset has empty validation split (`val=0`)
2. Health endpoint reports `validation_size: 0` but `status: "ok"`
3. Training starts normally
4. Collapse detection callback crashes when trying to evaluate on empty validation
5. Training metrics are incomplete/misleading

### After (Fixed State)

1. Dataset has empty validation split (`val=0`)
2. Health endpoint reports `validation_size: 0` and **`status: "degraded"`** with error message
3. User runs `make ensure-validation-split`
4. Validation split is created and saved (train=552, validation=62)
5. Backend reloads dataset and reports `validation_nonempty: true`
6. Training starts and validation metrics work correctly
7. Collapse detection runs successfully each epoch

## Testing

### Test 1: Standalone Script

```bash
$ python scripts/ensure_validation_split.py --dataset-dir data/datasets/boston_layoutlmv3_full

Loading dataset from data/datasets/boston_layoutlmv3_full/dataset_dict
Current splits: train=614, val=0, test=0
❌ Validation split 'val' exists but is EMPTY (0 examples)

Creating 10% validation split (seed=42)...
✓ Split complete: train=552, validation=62

Saving dataset with validation split...
✓ Dataset saved and verified: validation_size=62

✓ Success! Dataset now has validation split:
  train=552, validation=62, test=0
```

### Test 2: Health Endpoint Before

```bash
$ curl -s http://localhost:8000/api/parse/health | jq '{status, validation_nonempty, validation_size}'
{
  "status": "degraded",
  "validation_nonempty": false,
  "validation_size": 0
}
```

### Test 3: Health Endpoint After

```bash
$ make ensure-validation-split
$ # Restart backend to reload dataset
$ curl -s http://localhost:8000/api/parse/health | jq '{status, validation_nonempty, validation_size}'
{
  "status": "ok",
  "validation_nonempty": true,
  "validation_size": 62
}
```

### Test 4: Training Guardrail

```bash
# With empty validation
$ python training/train_layoutlmv3.py --stage full

❌ TRAINING FAILED: Validation split 'val' is EMPTY (0 examples).
   This will break validation metrics and collapse detection.
   → Run: make ensure-validation-split
   → Or rebuild dataset: make rebuild-dataset

# After fixing
$ make ensure-validation-split
$ python training/train_layoutlmv3.py --stage full

✓ Validation guardrail passed: train=552, validation=62
[Training proceeds normally...]
```

## Files Modified

1. **`training/train_layoutlmv3.py`** (+87 lines)
   - Enhanced `create_validation_split()` with RuntimeError guards
   - Enhanced `save_dataset_with_validation()` with verification
   - Added hard guardrails in `stage_train()`
   - Added `--force_resplit` argument

2. **`backend/app/services/layoutlm_service.py`** (+20 lines)
   - Added `validation_nonempty` and `validation_key` to health response
   - Degrades status to "degraded" if validation is empty
   - Adds error message with fix instructions

3. **`scripts/ensure_validation_split.py`** (NEW, 242 lines)
   - Standalone validation split creation
   - Atomic save with temp files and backup
   - Comprehensive verification
   - No training module dependencies

4. **`Makefile`** (+4 lines)
   - Added `ensure-validation-split` target
   - Updated `.PHONY` declarations

## Benefits

1. **Prevents Silent Failures**: Training fails immediately with clear error message
2. **Easy to Fix**: Single command (`make ensure-validation-split`) resolves the issue
3. **Proactive Monitoring**: Health endpoint detects and warns about empty validation
4. **Data Integrity**: Atomic saves with backups prevent corruption
5. **Comprehensive Logging**: Detailed logs at every step for debugging
6. **Deterministic Splits**: Seed-based splitting ensures reproducibility

## Recommendations

### For Development

```bash
# Always ensure validation split before training
make ensure-validation-split
python training/train_layoutlmv3.py --stage both
```

### For Production

```bash
# Check health endpoint regularly
curl http://localhost:8000/api/parse/health | jq '.validation_nonempty'

# If false, fix it
make ensure-validation-split

# Restart backend to reload
make stop && make dev
```

### For CI/CD

```yaml
# Add validation check to CI pipeline
- name: Ensure validation split
  run: python scripts/ensure_validation_split.py

- name: Verify validation size
  run: |
    validation_size=$(curl -s http://localhost:8000/api/parse/health | jq '.validation_size')
    if [ "$validation_size" -eq 0 ]; then
      echo "Validation split is empty!"
      exit 1
    fi
```

## Future Enhancements

1. **Automatic Healing**: Backend could automatically create validation split on startup if missing
2. **Split Ratio Configuration**: Make validation ratio configurable via environment variable
3. **Stratified Splitting**: Ensure label distribution is similar in train/validation
4. **Split Metadata**: Store split creation timestamp and parameters in dataset metadata
5. **Health Check Endpoint**: Add dedicated `/api/dataset/validate` endpoint for detailed checks

## Troubleshooting

### Issue: Validation still shows 0 after running script

**Cause**: Backend hasn't reloaded the dataset

**Fix:**
```bash
make stop
make dev
```

### Issue: RuntimeError during save

**Cause**: Disk space or permissions issue

**Fix:**
```bash
# Check disk space
df -h data/datasets/

# Check permissions
ls -la data/datasets/boston_layoutlmv3_full/

# Restore from backup if needed
mv data/datasets/boston_layoutlmv3_full/dataset_dict.backup.* \
   data/datasets/boston_layoutlmv3_full/dataset_dict
```

### Issue: Training still fails after creating split

**Cause**: Need to force resplit within training

**Fix:**
```bash
python training/train_layoutlmv3.py --force_resplit --stage full
```

## Summary

The validation split improvements ensure that:
- ✅ Validation splits are always non-empty
- ✅ Training fails fast with actionable errors
- ✅ Health monitoring detects issues proactively
- ✅ Dataset saves are atomic and verified
- ✅ Single command fixes the issue

This prevents hours of wasted training time due to silent validation failures!
