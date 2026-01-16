"""Fine-tune LayoutLMv3 for token classification in two stages (highconf -> full)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import datasets
import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from training.modeling.alignment import encode_example
from training.eval.metrics import compute_seqeval, compute_token_metrics, save_confusion_csv, save_report

LOG = logging.getLogger(__name__)


def check_dependencies():
    """Check that all required dependencies are installed."""
    missing = []

    try:
        import accelerate
    except ImportError:
        missing.append("accelerate")

    try:
        import seqeval
    except ImportError:
        missing.append("seqeval")

    if missing:
        print("=" * 80, file=sys.stderr)
        print("ERROR: Required dependencies are missing", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print("", file=sys.stderr)
        print("The following packages are required for training:", file=sys.stderr)
        for pkg in missing:
            print(f"  - {pkg}", file=sys.stderr)
        print("", file=sys.stderr)
        print("To install missing dependencies:", file=sys.stderr)
        print(f"  pip install {' '.join(missing)}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Or install all training dependencies:", file=sys.stderr)
        print("  pip install -r backend/requirements.txt", file=sys.stderr)
        print("  # or", file=sys.stderr)
        print("  make install-backend", file=sys.stderr)
        print("", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        sys.exit(1)
_collate_debug_printed = False


def run_label_sanity_check(ds: DatasetDict, processor: LayoutLMv3Processor, id2label: dict, num_samples: int = 20) -> bool:
    """
    Run sanity checks on encoded labels to ensure usable supervision signal.
    Returns True if checks pass, False otherwise.
    """
    LOG.info("\n" + "="*80)
    LOG.info("Running label sanity checks...")
    LOG.info("="*80)

    # Analyze samples from train split
    if "train" not in ds:
        LOG.error("❌ No train split found for sanity check")
        return False

    train_data = ds["train"]
    num_to_check = min(num_samples, len(train_data))

    LOG.info(f"Analyzing {num_to_check} training examples...")

    # Aggregate statistics
    total_tokens = 0
    total_masked = 0
    total_non_o = 0
    examples_with_ingredient = 0
    examples_with_instruction = 0
    examples_with_any_non_o = 0
    label_counts = Counter()

    for i in range(num_to_check):
        example = train_data[i]
        encoded = encode_example(example, processor, label_pad_token_id=-100, max_length=512)
        labels = encoded['labels']

        # Count statistics
        total_tokens += len(labels)
        masked_count = sum(1 for l in labels if l == -100)
        total_masked += masked_count

        # Count label distribution
        has_non_o = False
        has_ingredient = False
        has_instruction = False

        for label_id in labels:
            if label_id != -100:
                label_name = id2label.get(label_id, f"UNKNOWN_{label_id}")
                label_counts[label_name] += 1

                if label_name != 'O':
                    total_non_o += 1
                    has_non_o = True

                if 'INGREDIENT' in label_name:
                    has_ingredient = True
                if 'INSTRUCTION' in label_name:
                    has_instruction = True

        if has_non_o:
            examples_with_any_non_o += 1
        if has_ingredient:
            examples_with_ingredient += 1
        if has_instruction:
            examples_with_instruction += 1

    # Compute statistics
    unmasked_tokens = total_tokens - total_masked
    masked_ratio = total_masked / total_tokens if total_tokens > 0 else 0.0
    non_o_ratio = total_non_o / unmasked_tokens if unmasked_tokens > 0 else 0.0
    examples_with_any_non_o_pct = examples_with_any_non_o / num_to_check if num_to_check > 0 else 0.0

    # Print results
    LOG.info(f"\nSanity check results ({num_to_check} examples):")
    LOG.info(f"  Total tokens:     {total_tokens}")
    LOG.info(f"  Masked tokens:    {total_masked} ({masked_ratio:.1%})")
    LOG.info(f"  Unmasked tokens:  {unmasked_tokens}")
    LOG.info(f"  Non-O tokens:     {total_non_o} ({non_o_ratio:.1%} of unmasked)")

    LOG.info(f"\nLabel distribution (unmasked):")
    for label_name, count in sorted(label_counts.items(), key=lambda x: -x[1])[:10]:
        ratio = count / unmasked_tokens if unmasked_tokens > 0 else 0
        LOG.info(f"  {label_name:20s}: {count:5d} ({ratio:6.1%})")

    LOG.info(f"\nExamples with labels:")
    LOG.info(f"  Any non-O:        {examples_with_any_non_o}/{num_to_check} ({examples_with_any_non_o_pct:.1%})")
    LOG.info(f"  INGREDIENT_LINE:  {examples_with_ingredient}/{num_to_check}")
    LOG.info(f"  INSTRUCTION_STEP: {examples_with_instruction}/{num_to_check}")

    # Check failure conditions
    failures = []

    if masked_ratio > 0.95:
        failures.append(
            f"❌ FAIL: {masked_ratio:.1%} of tokens are masked (>95% threshold)"
        )

    if non_o_ratio < 0.01:
        failures.append(
            f"❌ FAIL: Only {non_o_ratio:.2%} of unmasked tokens have non-O labels (<1% threshold)"
        )

    if examples_with_any_non_o_pct < 0.10:
        failures.append(
            f"❌ FAIL: Only {examples_with_any_non_o_pct:.1%} of examples contain non-O labels (<10% threshold)"
        )

    if failures:
        LOG.error("\n" + "!"*80)
        LOG.error("SANITY CHECK FAILED")
        LOG.error("!"*80)
        for msg in failures:
            LOG.error(f"\n{msg}")
        LOG.error("\nLabels are not usable for training!")
        LOG.error("Please check:")
        LOG.error("  1. Dataset has correct 'labels' column with non-O annotations")
        LOG.error("  2. encode_example() is aligning labels correctly")
        LOG.error("  3. Tokenizer word boundaries match label alignment")
        return False

    LOG.info("\n" + "="*80)
    LOG.info("✓ SANITY CHECK PASSED")
    LOG.info("="*80)
    LOG.info("Labels look good! Encoding preserves usable supervision signal.\n")
    return True


def load_label_map(path: Path):
    data = json.loads(path.read_text())
    label2id = {k: int(v) if isinstance(v, str) and v.isdigit() else int(v) for k, v in data["label2id"].items()}
    id2label = {int(k): v for k, v in data["id2label"].items()}
    return label2id, id2label


def create_validation_split(ds_dict: DatasetDict, val_ratio: float = 0.1, seed: int = 42) -> DatasetDict:
    """
    Create a validation split from train if one doesn't exist.
    Returns a new DatasetDict with train + validation splits.

    Raises:
        RuntimeError: If validation split cannot be created or is empty
    """
    # Check for existing validation split (check all common names)
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
            # Remove empty split
            splits_to_keep = {k: v for k, v in ds_dict.items() if k != existing_val_split}
            ds_dict = DatasetDict(splits_to_keep)

    if "train" not in ds_dict:
        raise RuntimeError("No train split found, cannot create validation split")

    train_size = len(ds_dict["train"])

    # Verify dataset is large enough for split
    if train_size < 10:
        raise RuntimeError(
            f"Training set too small ({train_size} examples) to create validation split. "
            f"Need at least 10 examples."
        )

    LOG.info(f"Creating {val_ratio:.0%} validation split from train (seed={seed})")
    LOG.info(f"  Train size before split: {train_size}")

    train_ds = ds_dict["train"]

    # Split the train dataset
    split_ds = train_ds.train_test_split(test_size=val_ratio, seed=seed)

    # Create new DatasetDict with standardized "validation" name
    new_dict = DatasetDict({
        "train": split_ds["train"],
        "validation": split_ds["test"]
    })

    # Copy over any other splits (test, etc.)
    for split_name, split_data in ds_dict.items():
        if split_name not in new_dict and split_name != "train":
            new_dict[split_name] = split_data

    train_new = len(new_dict['train'])
    val_new = len(new_dict['validation'])

    # Verify splits are non-empty and disjoint
    if val_new == 0:
        raise RuntimeError(
            f"Validation split is empty after creation! "
            f"train={train_new}, validation={val_new}. "
            f"This should not happen. Check val_ratio={val_ratio}"
        )

    if train_new == 0:
        raise RuntimeError(
            f"Training split is empty after creation! "
            f"train={train_new}, validation={val_new}. "
            f"val_ratio={val_ratio} may be too large."
        )

    # Verify sizes add up
    if train_new + val_new != train_size:
        LOG.warning(
            f"Split size mismatch: original={train_size}, "
            f"train={train_new}, val={val_new}, sum={train_new + val_new}"
        )

    LOG.info(f"✓ Split complete: train={train_new}, validation={val_new}")
    LOG.info(f"  Validation ratio: {val_new / train_size:.1%} of original train set")

    return new_dict


def save_dataset_with_validation(ds_dict: DatasetDict, dataset_dir: Path) -> None:
    """
    Save the dataset with validation split to disk.

    Args:
        ds_dict: DatasetDict with train + validation (+ optionally other splits)
        dataset_dir: Directory containing dataset_dict folder

    Raises:
        RuntimeError: If validation split is missing or empty
    """
    # Verify validation split exists and is non-empty
    if "validation" not in ds_dict:
        raise RuntimeError(
            f"Cannot save dataset: no 'validation' split found. "
            f"Available splits: {list(ds_dict.keys())}"
        )

    val_size = len(ds_dict["validation"])
    if val_size == 0:
        raise RuntimeError(
            f"Cannot save dataset: validation split is EMPTY. "
            f"Split sizes: {{{', '.join(f'{k}={len(v)}' for k, v in ds_dict.items())}}}"
        )

    save_path = dataset_dir / "dataset_dict" if dataset_dir.name != "dataset_dict" else dataset_dir

    LOG.info(f"Saving dataset with validation split to {save_path}")
    LOG.info(f"  Splits: {', '.join(f'{k}={len(v)}' for k, v in ds_dict.items())}")

    # Save to temp location first, then move
    import time
    import shutil
    temp_path = save_path.parent / f"dataset_dict.temp.{int(time.time())}"

    LOG.info(f"  Saving to temporary location: {temp_path}")
    ds_dict.save_to_disk(str(temp_path))

    # Create backup if dataset already exists
    if save_path.exists():
        backup_path = save_path.parent / f"dataset_dict.backup.{int(time.time())}"
        LOG.info(f"  Creating backup: {backup_path}")
        shutil.move(str(save_path), str(backup_path))

    # Move temp to final location
    LOG.info(f"  Moving to final location: {save_path}")
    shutil.move(str(temp_path), str(save_path))

    # Verify save was successful
    if not save_path.exists():
        raise RuntimeError(f"Dataset save failed: {save_path} does not exist after save")

    # Verify we can load it back
    try:
        from datasets import load_from_disk
        loaded = load_from_disk(str(save_path))
        loaded_val_size = len(loaded["validation"]) if "validation" in loaded else 0
        if loaded_val_size == 0:
            raise RuntimeError(f"Validation split is empty after reload!")
        LOG.info(f"✓ Dataset saved and verified: validation_size={loaded_val_size}")
    except Exception as e:
        raise RuntimeError(f"Dataset save verification failed: {e}")


def compute_class_weights(ds_dict: DatasetDict, label2id: dict, ignore_index: int = -100) -> torch.Tensor:
    """
    Compute class weights from training label distribution.
    Returns tensor of weights for use in CrossEntropyLoss.
    """
    LOG.info("Computing class weights from training labels")
    train_ds = ds_dict["train"]

    # Count all labels (before encoding)
    label_counts = Counter()

    # Check for label column (ner_tags or labels)
    label_col = None
    if "ner_tags" in train_ds.column_names:
        label_col = "ner_tags"
    elif "labels" in train_ds.column_names:
        label_col = "labels"

    if label_col:
        for example in train_ds:
            tags = example[label_col]
            for tag in tags:
                if tag != ignore_index:
                    label_counts[tag] += 1
    else:
        # Fallback: all labels get equal weight
        LOG.warning("No ner_tags or labels column found, using uniform weights")
        return torch.ones(len(label2id))

    # Convert to weights (inverse frequency)
    num_labels = len(label2id)
    weights = torch.ones(num_labels)
    total_count = sum(label_counts.values())

    if total_count == 0:
        LOG.warning("No labels found in training data, using uniform weights")
        return weights

    for label_id, count in label_counts.items():
        if 0 <= label_id < num_labels:
            # Inverse frequency with smoothing
            weights[label_id] = total_count / (num_labels * count)

    # Normalize weights
    weights = weights / weights.sum() * num_labels

    LOG.info(f"Class weights: min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}")
    return weights


def encode_dataset(ds, processor: LayoutLMv3Processor, max_length: int):
    def _encode(ex):
        return encode_example(ex, processor, label_pad_token_id=-100, max_length=max_length)

    if isinstance(ds, datasets.DatasetDict):
        # Encode each split separately to avoid cache issues
        encoded_splits = {}
        for split_name, split_data in ds.items():
            cols = split_data.column_names
            LOG.info(f"Encoding {split_name} split ({len(split_data)} examples)...")
            encoded_splits[split_name] = split_data.map(_encode, remove_columns=cols, load_from_cache_file=False)
        return datasets.DatasetDict(encoded_splits)
    return ds.map(_encode, remove_columns=ds.column_names, load_from_cache_file=False)


def collate_fn(features: List[dict]):
    """
    Explicit collator to avoid shape surprises.
    Expects lists for ids/mask/labels/bbox and a tensor for pixel_values.
    Returns batched tensors with expected shapes for LayoutLMv3.
    """
    global _collate_debug_printed

    # Debug: print what keys we actually have
    if not _collate_debug_printed and len(features) > 0:
        LOG.info(f"[COLLATE_DEBUG] First feature keys: {list(features[0].keys())}")
        _collate_debug_printed = True

    input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
    labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
    bbox = torch.tensor([f["bbox"] for f in features], dtype=torch.long)
    pixel_values = torch.stack(
        [
            f["pixel_values"] if isinstance(f["pixel_values"], torch.Tensor) else torch.tensor(f["pixel_values"])
            for f in features
        ]
    ).to(torch.float32)

    # shape assertions
    assert input_ids.dim() == 2
    assert attention_mask.dim() == 2
    assert labels.dim() == 2
    assert bbox.dim() == 3 and bbox.size(-1) == 4
    assert pixel_values.dim() == 4
    assert input_ids.shape[1] == bbox.shape[1] == labels.shape[1], "sequence length mismatch in batch"

    if os.environ.get("COLLATE_DEBUG") == "1" and not _collate_debug_printed:
        _collate_debug_printed = True
        print(
            f"[COLLATE_DEBUG] input_ids {input_ids.shape}, attention_mask {attention_mask.shape}, "
            f"bbox {bbox.shape}, labels {labels.shape}, pixel_values {pixel_values.shape}"
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "bbox": bbox,
        "labels": labels,
        "pixel_values": pixel_values,
    }


def compute_metrics_fn(label_list: List[str]):
    """
    Compute metrics using token-level classification metrics (not seqeval).

    Uses sklearn-based metrics appropriate for non-BIO token classification.
    Focuses on non-O labels for precision/recall/F1 while providing full
    per-label breakdown and confusion matrix.
    """
    def _compute(p):
        preds = p.predictions
        labels = p.label_ids
        preds = preds.argmax(-1)

        # Flatten predictions and labels, filtering out masked tokens (-100)
        preds_flat = []
        labels_flat = []
        for pred, lab in zip(preds, labels):
            for p_i, l_i in zip(pred, lab):
                if l_i == -100:
                    continue
                preds_flat.append(p_i)
                labels_flat.append(l_i)

        # Convert to numpy arrays for sklearn
        preds_flat = np.array(preds_flat)
        labels_flat = np.array(labels_flat)

        # Use token-level metrics (appropriate for non-BIO labels)
        return compute_token_metrics(labels_flat, preds_flat, label_list)

    return _compute


class CollapseDetectionCallback(TrainerCallback):
    """
    Callback to detect label collapse during training.
    Logs predicted label distribution and warns if any label dominates.

    The callback should be initialized with eval_dataset passed explicitly
    to ensure reliable access during training.
    """

    def __init__(self, label_list: List[str], eval_dataset=None,
                 collapse_threshold: float = 0.9,
                 patience: int = 2, stage_name: str = "train"):
        self.label_list = label_list
        self.eval_dataset = eval_dataset  # Store eval_dataset at construction time
        self.collapse_threshold = collapse_threshold
        self.patience = patience
        self.stage_name = stage_name
        self.collapse_epochs = 0
        self.epoch_metrics = []

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Compute metrics on validation set after each epoch."""
        if model is None:
            return

        epoch = int(state.epoch)
        LOG.info(f"\n{'='*80}")
        LOG.info(f"Epoch {epoch} Complete - Computing Validation Metrics")
        LOG.info(f"{'='*80}")

        # Get validation dataset - use stored eval_dataset first (most reliable)
        eval_dataset = self.eval_dataset

        # Fallback 1: Try trainer.eval_dataset if available
        if eval_dataset is None:
            trainer = kwargs.get("trainer")
            if trainer is not None and hasattr(trainer, "eval_dataset"):
                eval_dataset = trainer.eval_dataset

        # Fallback 2: Try kwargs["eval_dataset"]
        if eval_dataset is None:
            eval_dataset = kwargs.get("eval_dataset")

        # Fallback 3: Last resort - try to get from trainer's eval dataloader
        if eval_dataset is None:
            trainer = kwargs.get("trainer")
            if trainer is not None:
                try:
                    eval_dataloader = trainer.get_eval_dataloader()
                    if eval_dataloader is not None and hasattr(eval_dataloader, "dataset"):
                        eval_dataset = eval_dataloader.dataset
                except Exception:
                    pass

        if eval_dataset is None:
            LOG.warning("No eval_dataset found for collapse detection")
            return

        # Run prediction on validation set
        try:
            from torch.utils.data import DataLoader
            device = model.device
            model.eval()

            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=args.per_device_eval_batch_size,
                collate_fn=collate_fn,
            )

            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in eval_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    logits = outputs.logits
                    preds = logits.argmax(dim=-1)
                    labels = batch["labels"]

                    # Flatten and filter out -100
                    preds_flat = preds.view(-1).cpu().numpy()
                    labels_flat = labels.view(-1).cpu().numpy()

                    mask = labels_flat != -100
                    all_preds.extend(preds_flat[mask].tolist())
                    all_labels.extend(labels_flat[mask].tolist())

            # Compute predicted label distribution
            pred_counts = Counter(all_preds)
            total_preds = len(all_preds)

            pred_dist = {}
            for label_id, count in pred_counts.items():
                if 0 <= label_id < len(self.label_list):
                    label_name = self.label_list[label_id]
                    pred_dist[label_name] = {
                        "count": count,
                        "percentage": count / total_preds * 100
                    }

            # Sort by percentage descending
            sorted_dist = sorted(pred_dist.items(), key=lambda x: x[1]["percentage"], reverse=True)

            LOG.info("\nPredicted Label Distribution (validation set):")
            LOG.info("-" * 60)
            for label_name, stats in sorted_dist[:10]:  # Top 10
                LOG.info(f"  {label_name:25s}: {stats['count']:6d} ({stats['percentage']:5.2f}%)")

            # Check for collapse
            max_percentage = sorted_dist[0][1]["percentage"] / 100 if sorted_dist else 0
            max_label = sorted_dist[0][0] if sorted_dist else "unknown"

            if max_percentage > self.collapse_threshold:
                self.collapse_epochs += 1
                LOG.warning(f"\n⚠️  COLLAPSE WARNING: Label '{max_label}' dominates with {max_percentage:.1%} of predictions")
                LOG.warning(f"   Collapse detected for {self.collapse_epochs} consecutive epoch(s)")

                if self.collapse_epochs >= self.patience:
                    LOG.error(f"\n❌ TRAINING FAILURE: Label collapse persisted for {self.collapse_epochs} epochs")
                    LOG.error(f"   Training may have collapsed. Consider:")
                    LOG.error(f"   - Reducing learning rate")
                    LOG.error(f"   - Using weighted loss")
                    LOG.error(f"   - Checking label distribution in training data")
                    # Don't stop training automatically, but log severe warning
            else:
                self.collapse_epochs = 0
                LOG.info(f"\n✓ Label distribution healthy (max: {max_label} at {max_percentage:.1%})")

            # Compute per-label metrics using token-level metrics
            if all_preds and all_labels:
                metrics = compute_token_metrics(
                    np.array(all_labels),
                    np.array(all_preds),
                    self.label_list
                )

                LOG.info("\nPer-Label Metrics:")
                LOG.info("-" * 80)
                LOG.info(f"{'Label':25s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
                LOG.info("-" * 80)

                report = metrics.get("report", {})
                for label_name in sorted(report.keys()):
                    if label_name in {"macro avg", "weighted avg", "micro avg"}:
                        continue
                    label_metrics = report[label_name]
                    LOG.info(
                        f"{label_name:25s} "
                        f"{label_metrics['precision']:>10.3f} "
                        f"{label_metrics['recall']:>10.3f} "
                        f"{label_metrics['f1-score']:>10.3f} "
                        f"{label_metrics['support']:>10d}"
                    )

                # Print confusion matrix top confusions
                if "confusion_matrix" in metrics and "labels" in metrics:
                    cm = np.array(metrics["confusion_matrix"])
                    labels = metrics["labels"]

                    # Find top confusions (off-diagonal)
                    confusions = []
                    for i in range(len(labels)):
                        for j in range(len(labels)):
                            if i != j and cm[i, j] > 0:
                                confusions.append((labels[i], labels[j], cm[i, j]))

                    confusions.sort(key=lambda x: x[2], reverse=True)

                    if confusions:
                        LOG.info("\nTop Confusions:")
                        LOG.info("-" * 60)
                        for true_lbl, pred_lbl, count in confusions[:5]:
                            LOG.info(f"  {true_lbl} → {pred_lbl}: {count} times")

                # Store metrics for summary
                self.epoch_metrics.append({
                    "epoch": epoch,
                    "pred_distribution": pred_dist,
                    "max_label": max_label,
                    "max_percentage": max_percentage,
                    "metrics": report,
                    "collapse_warning": max_percentage > self.collapse_threshold,
                })

            LOG.info(f"{'='*80}\n")

        except Exception as e:
            LOG.error(f"Error in collapse detection: {e}")
            import traceback
            traceback.print_exc()


def make_trainer(
    ds_dict,
    processor,
    model,
    args: argparse.Namespace,
    output_dir: Path,
    label_list: List[str],
    stage_tag: str,
    class_weights: Optional[torch.Tensor] = None,
):
    # Ensure validation split exists and is properly named
    eval_split = None
    if isinstance(ds_dict, datasets.DatasetDict):
        if "validation" in ds_dict:
            eval_split = ds_dict["validation"]
        elif "val" in ds_dict:
            # Legacy support for "val" split name (should be migrated to "validation")
            LOG.warning("Found 'val' split instead of 'validation'. Consider rebuilding dataset with standard split names.")
            eval_split = ds_dict["val"]
        elif "eval" in ds_dict:
            eval_split = ds_dict["eval"]
        else:
            raise RuntimeError(
                "No validation split found in dataset. "
                "Expected 'validation' split for proper evaluation. "
                "Available splits: " + str(list(ds_dict.keys()))
            )
    else:
        eval_split = ds_dict

    # Validate that eval_dataset is not None and not empty
    if eval_split is None:
        raise RuntimeError("eval_dataset is None - cannot initialize trainer without validation data")

    if len(eval_split) == 0:
        raise RuntimeError("eval_dataset is empty - validation split has 0 examples")

    LOG.info(f"✓ Using validation split with {len(eval_split)} examples for evaluation")
    LOG.info(f"  CollapseDetectionCallback will monitor validation metrics every epoch")

    # Apply class weights to model if provided
    if class_weights is not None and args.use_weighted_loss:
        LOG.info("Using weighted loss with class weights")

        # Store original forward
        original_forward = model.forward

        # Create weighted loss wrapper
        def forward_with_weighted_loss(*args, **kwargs):
            outputs = original_forward(*args, **kwargs)
            if outputs.loss is not None and "labels" in kwargs:
                # Recompute loss with class weights
                logits = outputs.logits
                labels = kwargs["labels"]

                # Move weights to same device as logits
                weights_device = class_weights.to(logits.device)

                # Flatten for loss computation
                loss_fct = torch.nn.CrossEntropyLoss(weight=weights_device, ignore_index=-100)
                loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
                outputs.loss = loss
            return outputs

        model.forward = forward_with_weighted_loss

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs_stageA if stage_tag == "stageA" else args.num_train_epochs_stageB,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,  # Keep all encoded columns for LayoutLMv3
    )

    # Add collapse detection callback with eval_dataset passed explicitly
    collapse_callback = CollapseDetectionCallback(
        label_list=label_list,
        eval_dataset=eval_split,  # Pass eval_dataset explicitly for reliable access
        collapse_threshold=0.9,
        patience=2,
        stage_name=stage_tag,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_dict["train"],
        eval_dataset=eval_split,
        tokenizer=processor.tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics_fn(label_list),
        callbacks=[collapse_callback],
    )

    # Store callback for later access to metrics
    trainer.collapse_callback = collapse_callback

    return trainer


def stage_train(stage_name: str, dataset_dir: Path, processor, label2id, id2label, args: argparse.Namespace, ckpt: Optional[Path]) -> Path:
    ds_path = dataset_dir
    if ds_path.name == "dataset_dict":
        ds = load_from_disk(str(ds_path))
    else:
        ds = load_from_disk(str(ds_path / "dataset_dict"))

    # Create validation split if needed
    if isinstance(ds, DatasetDict):
        original_size = {k: len(v) for k, v in ds.items()}

        # Check if we need to create/fix validation split
        needs_split_creation = False
        if "validation" not in ds and "val" not in ds and "eval" not in ds:
            LOG.info("No validation split found, will create one")
            needs_split_creation = True
        else:
            # Check if existing validation is empty
            val_key = "validation" if "validation" in ds else ("val" if "val" in ds else "eval")
            if len(ds[val_key]) == 0:
                LOG.warning(f"Validation split '{val_key}' exists but is EMPTY, will recreate")
                needs_split_creation = True

        if needs_split_creation or args.force_resplit:
            LOG.info(f"Creating validation split (force_resplit={getattr(args, 'force_resplit', False)})")
            ds_new = create_validation_split(ds, val_ratio=args.val_ratio, seed=args.seed)

            # Delete old dataset reference before saving
            del ds

            # Always save when we create a new split
            save_dataset_with_validation(ds_new, dataset_dir)

            # Reload the dataset from disk
            if dataset_dir.name == "dataset_dict":
                ds = load_from_disk(str(dataset_dir))
            else:
                ds = load_from_disk(str(dataset_dir / "dataset_dict"))
        else:
            # Validation exists and is non-empty
            val_key = "validation" if "validation" in ds else ("val" if "val" in ds else "eval")
            LOG.info(f"Using existing validation split '{val_key}' ({len(ds[val_key])} examples)")

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

        train_size = len(ds["train"]) if "train" in ds else 0
        LOG.info(f"✓ Validation guardrail passed: train={train_size}, {val_key}={val_size}")

        # Normalize to "validation" for consistency downstream
        if val_key != "validation":
            LOG.info(f"Normalizing validation split name: {val_key} → validation")
            ds_normalized = DatasetDict({
                "train": ds["train"],
                "validation": ds[val_key]
            })
            # Copy other splits
            for k, v in ds.items():
                if k not in ds_normalized:
                    ds_normalized[k] = v
            ds = ds_normalized

    label_list = [id2label[i] for i in sorted(id2label.keys())]

    # Run sanity checks BEFORE encoding (needs raw dataset with image_path)
    if args.run_sanity_checks:
        LOG.info("\n" + "="*80)
        LOG.info("Running label sanity checks on raw dataset...")
        LOG.info("="*80)

        sanity_passed = run_label_sanity_check(ds, processor, id2label, num_samples=20)

        if not sanity_passed:
            LOG.error("\n" + "!"*80)
            LOG.error("❌ SANITY CHECK FAILED - Aborting training")
            LOG.error("!"*80)
            LOG.error("\nLabels are not usable for training. Please investigate:")
            LOG.error("1. Check that dataset has correct 'labels' column with non-O annotations")
            LOG.error("2. Verify encode_example() is aligning labels correctly")
            LOG.error("3. Check for issues with tokenizer word boundaries")
            LOG.error("4. Review label masking logic (should only mask special tokens)")
            LOG.error("\nTo debug further, run:")
            LOG.error(f"  python scripts/sanity_check_encoded_labels.py --dataset-dir {dataset_dir} --verbose")
            import sys
            sys.exit(1)

        LOG.info("\n" + "="*80)
        LOG.info("✓ SANITY CHECK PASSED - Proceeding with training")
        LOG.info("="*80 + "\n")

    # Compute class weights before encoding
    class_weights = None
    if args.use_weighted_loss:
        class_weights = compute_class_weights(ds, label2id)

    # Encode dataset
    ds = encode_dataset(ds, processor, args.max_length)

    # Debug: verify encoded dataset has correct columns
    LOG.info(f"Encoded dataset train columns: {ds['train'].column_names}")
    LOG.info(f"Encoded dataset train[0] keys: {list(ds['train'][0].keys())}")

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )
    if ckpt:
        model = LayoutLMv3ForTokenClassification.from_pretrained(
            str(ckpt),
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id,
        )

    out_dir = Path(args.output_dir) / ("layoutlmv3_boston_stageA_highconf" if stage_name == "stageA" else "layoutlmv3_boston_stageB_full")
    trainer = make_trainer(ds, processor, model, args, out_dir, label_list,
                          "stageA" if stage_name == "stageA" else "stageB",
                          class_weights=class_weights)
    trainer.train(resume_from_checkpoint=ckpt if stage_name != "stageA" and ckpt else None)
    trainer.save_model(str(out_dir))
    trainer.state.save_to_json(str(out_dir / "trainer_state.json"))

    metrics_raw = trainer.evaluate()
    metrics = {k.replace("eval_", "") if k.startswith("eval_") else k: v for k, v in metrics_raw.items()}
    reports_dir = Path("data/reports/training")
    reports_dir.mkdir(parents=True, exist_ok=True)
    metric_file = reports_dir / ("stageA_metrics.json" if stage_name == "stageA" else "stageB_metrics.json")
    report_md = reports_dir / ("stageA_report.md" if stage_name == "stageA" else "stageB_report.md")
    save_report(metrics, metric_file, report_md)
    if "confusion_matrix" in metrics:
        save_confusion_csv(metrics, reports_dir / ("confusion_matrix_stageB.csv" if stage_name != "stageA" else "confusion_matrix_stageA.csv"))

    # Save training summary with epoch metrics
    if hasattr(trainer, "collapse_callback"):
        debug_dir = Path("docs/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "stage": stage_name,
            "timestamp": datetime.now().isoformat(),
            "model_dir": str(out_dir),
            "dataset_splits": {k: len(v) for k, v in ds.items()} if isinstance(ds, DatasetDict) else {},
            "final_metrics": metrics,
            "epoch_metrics": trainer.collapse_callback.epoch_metrics,
            "training_args": {
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "num_epochs": args.num_train_epochs_stageA if stage_name == "stageA" else args.num_train_epochs_stageB,
                "seed": args.seed,
                "use_weighted_loss": args.use_weighted_loss,
            }
        }

        summary_file = debug_dir / f"train_metrics_{stage_name}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        LOG.info(f"\n✓ Training summary saved to {summary_file}")

    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune LayoutLMv3 for token classification (two-stage)")
    parser.add_argument("--highconf_dataset_dir", required=False, default="data/_generated/datasets/boston_layoutlmv3_highconf")
    parser.add_argument("--full_dataset_dir", required=False, default="data/_generated/datasets/boston_layoutlmv3_full")
    parser.add_argument("--recipe_only_dataset_dir", required=False, default="data/datasets/boston_layoutlmv3_recipe_only", help="Recipe-only filtered dataset (dense supervision)")
    parser.add_argument("--output_dir", required=False, default="models")
    parser.add_argument("--model_name", default="microsoft/layoutlmv3-base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs_stageA", type=float, default=5)
    parser.add_argument("--num_train_epochs_stageB", type=float, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--stage", choices=["highconf", "full", "both"], default="both")
    parser.add_argument("--use_recipe_only", action="store_true", help="Use recipe-only dataset instead of full dataset for stage=full")
    parser.add_argument("--debug", action="store_true", help="Use small dataset for quick tests")
    parser.add_argument("--init_checkpoint", help="Checkpoint path to initialize stage B/full training")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio (default: 0.1 = 10%%)")
    parser.add_argument("--use_weighted_loss", action="store_true", default=True, help="Use weighted CrossEntropyLoss based on class frequency")
    parser.add_argument("--no_weighted_loss", dest="use_weighted_loss", action="store_false", help="Disable weighted loss")
    parser.add_argument("--force_resplit", action="store_true", help="Force recreation of validation split even if one exists")
    parser.add_argument("--run_sanity_checks", action="store_true", help="Run label sanity checks before training")
    return parser.parse_args()


def main() -> None:
    # Check dependencies before doing anything else
    check_dependencies()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    torch.manual_seed(args.seed)

    highconf_dir = Path(args.highconf_dataset_dir)
    full_dir = Path(args.full_dataset_dir)
    recipe_only_dir = Path(args.recipe_only_dataset_dir)

    # use small dataset for debug
    if args.debug:
        full_dir = Path("data/datasets/boston_layoutlmv3_small")
        highconf_dir = full_dir

    # Use recipe-only dataset if requested
    if args.use_recipe_only and args.stage in ("full", "both"):
        LOG.info("Using recipe-only dataset (dense supervision) instead of full dataset")
        full_dir = recipe_only_dir

    # load label map from full (authoritative)
    def _resolve_label_map(dataset_dir: Path) -> Path:
        dataset_root = dataset_dir.parent if dataset_dir.name == "dataset_dict" else dataset_dir
        candidate = dataset_root / "label_map.json"
        fallback = dataset_root.parent / "label_map.json"
        if candidate.exists():
            return candidate
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"label_map.json not found near {dataset_dir}")

    label_map_path = _resolve_label_map(full_dir)
    label2id, id2label = load_label_map(label_map_path)

    processor = LayoutLMv3Processor.from_pretrained(args.model_name, apply_ocr=False)

    stageA_ckpt: Optional[Path] = None
    if args.stage in ("highconf", "both"):
        if not (highconf_dir / "dataset_dict").exists():
            LOG.error("Highconf dataset not found at %s", highconf_dir)
            return
        stageA_ckpt = stage_train("stageA", highconf_dir, processor, label2id, id2label, args, ckpt=None)

    init_ckpt = Path(args.init_checkpoint) if args.init_checkpoint else None

    if args.stage in ("full", "both"):
        if not (full_dir / "dataset_dict").exists():
            LOG.error("Full dataset not found at %s", full_dir)
            return
        stageB_ckpt = stage_train(
            "stageB",
            full_dir,
            processor,
            label2id,
            id2label,
            args,
            ckpt=stageA_ckpt if stageA_ckpt else init_ckpt,
        )
        # copy final
        final_dir = Path(args.output_dir) / "layoutlmv3_boston_final"
        final_dir.mkdir(parents=True, exist_ok=True)
        # save final model weights
        model = LayoutLMv3ForTokenClassification.from_pretrained(str(stageB_ckpt))
        model.save_pretrained(str(final_dir))
        processor.save_pretrained(str(final_dir))


if __name__ == "__main__":
    main()
