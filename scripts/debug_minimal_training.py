#!/usr/bin/env python3
"""Minimal training test to reproduce CUDA error."""

import os
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    Trainer,
    TrainingArguments,
)

# Set CUDA_LAUNCH_BLOCKING for better error messages
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Add ml config to path
ml_path = Path(__file__).parent / "ml"
if str(ml_path) not in sys.path:
    sys.path.insert(0, str(ml_path))

from config.labels import get_label_config

# Add training modules
training_path = Path(__file__).parent / "training"
if str(training_path) not in sys.path:
    sys.path.insert(0, str(training_path))

from modeling.alignment import encode_example

# Load label config
label_config = get_label_config(version="v3")
print(f"Label config: {label_config['num_labels']} labels")

# Load dataset
dataset_path = Path("data/datasets/boston_layoutlmv3_v3/dataset_dict")
print(f"Loading dataset from {dataset_path}...")
ds_dict = load_from_disk(dataset_path)

# Take only first 10 examples for quick testing
train_ds = ds_dict["train"].select(range(10))
val_ds = ds_dict["validation"].select(range(5))

print(f"Using {len(train_ds)} training examples and {len(val_ds)} validation examples")

# Load processor
print("Loading processor...")
processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base",
    apply_ocr=False,
)

# Initialize model
print("Initializing model...")
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=label_config["num_labels"],
    id2label=label_config["id2label"],
    label2id=label_config["label2id"],
)

print(f"Model config: num_labels={model.config.num_labels}")

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
        encoded = encode_example(ex, processor, label_pad_token_id=-100, max_length=512)
        batch["input_ids"].append(encoded["input_ids"])
        batch["attention_mask"].append(encoded["attention_mask"])
        batch["bbox"].append(encoded["bbox"])
        batch["pixel_values"].append(encoded["pixel_values"])
        batch["labels"].append(encoded["labels"])

    # Convert to tensors
    return {
        "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
        "bbox": torch.tensor(batch["bbox"], dtype=torch.long),
        "labels": torch.tensor(batch["labels"], dtype=torch.long),
        "pixel_values": torch.stack(batch["pixel_values"]),
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="models/test_debug",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=5e-5,
    evaluation_strategy="steps",
    eval_steps=5,
    save_strategy="no",
    logging_steps=1,
    dataloader_num_workers=0,  # Set to 0 for debugging
    remove_unused_columns=False,
    label_names=["labels"],
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
)

# Train
print("\n" + "=" * 80)
print("STARTING MINIMAL TRAINING TEST")
print("=" * 80)

try:
    trainer.train()
    print("\n✓ Training completed successfully!")
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)
