#!/usr/bin/env python3
"""
Train LayoutLMv3 with v3 label taxonomy (header-aware).

This script:
1. Loads v3 dataset with PAGE_HEADER/SECTION_HEADER/RECIPE_TITLE labels
2. Trains with class imbalance handling (weighted loss or focal loss)
3. Evaluates with header-specific metrics
4. Includes anti-collapse checks to prevent "SERVINGS everywhere" failures
5. Reports on demo_eval set separately

Usage:
    python tools/train_v3_model.py \\
        --dataset data/datasets/boston_layoutlmv3_v3/dataset_dict \\
        --output models/layoutlmv3_v3_headers \\
        --epochs 15 \\
        --batch-size 4 \\
        --learning-rate 5e-5

Output:
    models/layoutlmv3_v3_headers/
        checkpoint-best/
            config.json (with id2label/label2id)
            model.safetensors
            processor/
        training_report.json
        demo_eval_report.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# Add ml config to path
ml_path = Path(__file__).parent.parent / "ml"
if str(ml_path) not in sys.path:
    sys.path.insert(0, str(ml_path))

from config.labels import get_label_config

# Add training modules
training_path = Path(__file__).parent.parent / "training"
if str(training_path) not in sys.path:
    sys.path.insert(0, str(training_path))

from modeling.alignment import encode_example
from eval.metrics import compute_seqeval, compute_token_metrics
from eval.header_metrics import (
    compute_header_title_metrics,
    compute_title_anchor_accuracy,
    print_demo_scorecard,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class WeightedLossTrainer(Trainer):
    """Custom Trainer with class-weighted cross entropy loss."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Use weighted cross entropy if weights provided
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device),
                ignore_index=-100
            )
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        else:
            # Default loss
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


class CollapseDetectionCallback(TrainerCallback):
    """
    Detect if model collapses to predicting one class everywhere.

    If >X% of predictions are the same class on validation, stop training.
    Only checks after a minimum number of epochs to allow initial learning.
    """

    def __init__(self, threshold=0.80, check_every_n_steps=100, min_epochs=3):
        self.threshold = threshold
        self.check_every_n_steps = check_every_n_steps
        self.min_epochs = min_epochs

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        # Don't check for collapse until after min_epochs
        current_epoch = state.epoch if state.epoch is not None else 0
        if current_epoch < self.min_epochs:
            return

        # Check if pred_label_distribution exists
        pred_dist = metrics.get("eval_pred_label_distribution", {})

        if not pred_dist:
            return

        # Find most common prediction
        total_preds = sum(pred_dist.values())
        if total_preds == 0:
            return

        max_class_count = max(pred_dist.values())
        max_class_ratio = max_class_count / total_preds

        if max_class_ratio > self.threshold:
            max_class = max(pred_dist, key=pred_dist.get)
            logger.error(
                f"\n{'!' * 80}\n"
                f"COLLAPSE DETECTED!\n"
                f"{'!' * 80}\n"
                f"  {max_class_ratio:.1%} of predictions are class '{max_class}'\n"
                f"  This exceeds collapse threshold of {self.threshold:.1%}\n"
                f"  Training will be stopped to prevent useless checkpoint.\n"
                f"{'!' * 80}\n"
            )
            control.should_training_stop = True


def compute_metrics_fn(eval_dataset, id2label):
    """
    Factory to create compute_metrics function for Trainer.

    Args:
        eval_dataset: Eval dataset for computing anchor accuracy
        id2label: Label mapping

    Returns:
        compute_metrics function
    """

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)

        # Flatten and filter out padding tokens (-100)
        mask = labels != -100
        predictions_flat = predictions[mask]
        labels_flat = labels[mask]

        # Standard token metrics (y_true, y_pred, label_names)
        token_metrics = compute_token_metrics(labels_flat, predictions_flat, id2label)

        # Header-specific metrics (now uses flattened data)
        header_metrics = compute_header_title_metrics(
            predictions_flat, labels_flat, id2label
        )

        # Title anchor accuracy (still needs batched predictions for bbox matching)
        anchor_metrics = compute_title_anchor_accuracy(
            eval_dataset, predictions, id2label
        )

        # Combine all metrics
        combined = {**token_metrics, **header_metrics, **anchor_metrics}

        return combined

    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Train LayoutLMv3 v3 (header-aware)")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to v3 dataset_dict")
    parser.add_argument("--output", type=Path, required=True, help="Output model directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")

    args = parser.parse_args()

    # Load label config
    label_config = get_label_config(version="v3")
    logger.info(f"Using v3 label taxonomy: {label_config['num_labels']} labels")

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}...")
    ds_dict = load_from_disk(args.dataset)
    logger.info(f"Dataset loaded:")
    for split, ds in ds_dict.items():
        logger.info(f"  {split}: {len(ds)} examples")

    # Initialize processor and model
    logger.info("Initializing LayoutLMv3 processor and model...")
    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base",
        apply_ocr=False,
    )

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=label_config["num_labels"],
        id2label=label_config["id2label"],
        label2id=label_config["label2id"],
    )

    logger.info(f"Model initialized with {label_config['num_labels']} labels")

    # Compute class weights from training data (inverse frequency)
    logger.info("Computing class weights from training data...")
    label_counts = np.zeros(label_config["num_labels"])
    for example in ds_dict["train"]:
        labels = example["labels"]
        for label_id in labels:
            if label_id >= 0:  # Ignore padding (-100)
                label_counts[label_id] += 1

    # Compute inverse frequency weights (with smoothing)
    total_samples = label_counts.sum()
    class_weights = total_samples / (label_config["num_labels"] * (label_counts + 1))  # +1 smoothing

    # Apply dampening factor to INSTRUCTION_STEP to reduce over-prediction
    # INSTRUCTION_STEP has been shown to be over-predicted by ~45% in validation/test
    # Reduce its weight by 50% to balance precision vs recall
    instruction_step_id = next(
        (lid for lid, lname in label_config["id2label"].items() if lname == "INSTRUCTION_STEP"),
        None
    )
    if instruction_step_id is not None:
        original_weight = class_weights[instruction_step_id]
        class_weights[instruction_step_id] *= 0.5  # Dampen by 50%
        logger.info(f"Applied dampening to INSTRUCTION_STEP: {original_weight:.2f} -> {class_weights[instruction_step_id]:.2f}")

    class_weights = torch.FloatTensor(class_weights)

    logger.info(f"Class weights computed:")
    for label_id, weight in enumerate(class_weights):
        label_name = label_config["id2label"][label_id]
        count = int(label_counts[label_id])
        logger.info(f"  {label_name:20s}: weight={weight:.2f}, count={count}")

    # Data collator
    def collate_fn(examples):
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "bbox": [],
            "pixel_values": [],
            "labels": [],
        }

        for ex in examples:
            encoded = encode_example(ex, processor, label_pad_token_id=-100, max_length=args.max_length)
            batch["input_ids"].append(encoded["input_ids"])
            batch["attention_mask"].append(encoded["attention_mask"])
            batch["bbox"].append(encoded["bbox"])
            batch["pixel_values"].append(encoded["pixel_values"])
            batch["labels"].append(encoded["labels"])

        # Convert to tensors
        # pixel_values is already a tensor from encode_example, others are lists
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "bbox": torch.tensor(batch["bbox"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "pixel_values": torch.stack(batch["pixel_values"]),
        }

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_title_anchor_accuracy",  # Optimize for demo metric!
        greater_is_better=True,
        logging_steps=50,
        seed=args.seed,
        dataloader_num_workers=0,  # Set to 0 to avoid multiprocessing issues
        remove_unused_columns=False,
        label_names=["labels"],
    )

    # Metrics function
    metrics_fn = compute_metrics_fn(ds_dict["validation"], label_config["id2label"])

    # Callbacks
    collapse_callback = CollapseDetectionCallback(threshold=0.75, check_every_n_steps=100, min_epochs=5)

    # Trainer with class weights
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_dict["train"],
        eval_dataset=ds_dict["validation"],
        data_collator=collate_fn,
        compute_metrics=metrics_fn,
        callbacks=[collapse_callback],
        class_weights=class_weights,
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save final model
    logger.info(f"Saving final model to {args.output}...")
    trainer.save_model()
    processor.save_pretrained(args.output)

    # Evaluate on all splits
    logger.info("\n" + "=" * 80)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 80)

    all_metrics = {}

    for split_name in ["validation", "test", "demo_eval"]:
        if split_name not in ds_dict:
            continue

        logger.info(f"\nEvaluating on {split_name}...")

        # Update compute_metrics to use the correct dataset for this split
        # This ensures anchor accuracy and header metrics use the right ground truth
        trainer.compute_metrics = compute_metrics_fn(ds_dict[split_name], label_config["id2label"])

        eval_metrics = trainer.evaluate(eval_dataset=ds_dict[split_name])

        # Print demo scorecard
        if split_name == "demo_eval":
            print_demo_scorecard(eval_metrics)

        all_metrics[split_name] = eval_metrics

    # Save reports
    report_path = args.output / "training_report.json"
    with open(report_path, "w") as f:
        # Convert Path objects to strings for JSON serialization
        training_args_serializable = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        json.dump({
            "label_config": label_config,
            "training_args": training_args_serializable,
            "train_result": {k: v for k, v in train_result.metrics.items()},
            "eval_metrics": all_metrics,
        }, f, indent=2)

    logger.info(f"✓ Training complete!")
    logger.info(f"✓ Model saved to: {args.output}")
    logger.info(f"✓ Report saved to: {report_path}")

    # Print summary
    demo_metrics = all_metrics.get("demo_eval", {})
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Best checkpoint: {args.output}")
    print(f"Demo eval title anchor accuracy: {demo_metrics.get('eval_title_anchor_accuracy', 0.0):.1%}")
    print(f"Demo eval header→title confusion: {demo_metrics.get('eval_header_title_confusion_count', 0)} tokens")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
