#!/usr/bin/env python3
"""Debug the collate function with multiple examples."""

import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import LayoutLMv3Processor

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

# Load dataset
dataset_path = Path("data/datasets/boston_layoutlmv3_v3/dataset_dict")
print(f"Loading dataset from {dataset_path}...")
ds_dict = load_from_disk(dataset_path)

# Load processor
print("Loading processor...")
processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base",
    apply_ocr=False,
)

# Define collate function (same as in train_v3_model.py)
def collate_fn(examples):
    batch = {
        "input_ids": [],
        "attention_mask": [],
        "bbox": [],
        "pixel_values": [],
        "labels": [],
    }

    print(f"  Collating {len(examples)} examples...")

    for i, ex in enumerate(examples):
        try:
            encoded = encode_example(ex, processor, label_pad_token_id=-100, max_length=512)
            batch["input_ids"].append(encoded["input_ids"])
            batch["attention_mask"].append(encoded["attention_mask"])
            batch["bbox"].append(encoded["bbox"])
            batch["pixel_values"].append(encoded["pixel_values"])
            batch["labels"].append(encoded["labels"])

            # Check for any invalid labels
            encoded_labels = [lbl for lbl in encoded["labels"] if lbl != -100]
            if encoded_labels:
                max_label = max(encoded_labels)
                min_label = min(encoded_labels)
                if max_label >= label_config["num_labels"] or min_label < 0:
                    print(f"    ❌ Example {i}: Invalid labels [{min_label}, {max_label}]")
        except Exception as e:
            print(f"    ❌ Example {i}: Encoding failed: {e}")
            raise

    # Convert to tensors
    try:
        result = {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "bbox": torch.tensor(batch["bbox"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "pixel_values": torch.stack(batch["pixel_values"]),
        }
        print(f"  ✓ Collation successful")
        print(f"    Batch shapes:")
        for k, v in result.items():
            print(f"      {k}: {v.shape}")

        # Check value ranges
        print(f"    Value ranges:")
        print(f"      input_ids: [{result['input_ids'].min()}, {result['input_ids'].max()}]")
        print(f"      bbox: [{result['bbox'].min()}, {result['bbox'].max()}]")
        labels_without_pad = result['labels'][result['labels'] != -100]
        if len(labels_without_pad) > 0:
            print(f"      labels (excluding -100): [{labels_without_pad.min()}, {labels_without_pad.max()}]")

        return result
    except Exception as e:
        print(f"  ❌ Tensor conversion failed: {e}")
        raise


# Test collate with different batch sizes
print("\n" + "=" * 80)
print("TESTING COLLATE FUNCTION")
print("=" * 80)

train_ds = ds_dict["train"]

for batch_size in [1, 2, 4]:
    print(f"\nBatch size: {batch_size}")
    examples = [train_ds[i] for i in range(batch_size)]
    try:
        batch = collate_fn(examples)
        print(f"  ✓ Batch creation successful for batch_size={batch_size}")
    except Exception as e:
        print(f"  ❌ Batch creation failed for batch_size={batch_size}: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)
