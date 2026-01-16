#!/usr/bin/env python3
"""Find which training example causes the CUDA error."""

import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

# Add paths
ml_path = Path(__file__).parent / "ml"
if str(ml_path) not in sys.path:
    sys.path.insert(0, str(ml_path))

from config.labels import get_label_config

training_path = Path(__file__).parent / "training"
if str(training_path) not in sys.path:
    sys.path.insert(0, str(training_path))

from modeling.alignment import encode_example

# Load everything
label_config = get_label_config(version="v3")
ds_dict = load_from_disk("data/datasets/boston_layoutlmv3_v3/dataset_dict")
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=label_config["num_labels"],
    id2label=label_config["id2label"],
    label2id=label_config["label2id"],
).cuda()

model.eval()

train_ds = ds_dict["train"]

print(f"Testing {len(train_ds)} training examples...")
print("=" * 80)

# The error happened at step 2 with batch_size=4, so it's in examples 4-7
# Let's test examples in batches to narrow it down
batch_size = 4

for batch_start in range(0, min(20, len(train_ds)), batch_size):
    batch_examples = [train_ds[i] for i in range(batch_start, min(batch_start + batch_size, len(train_ds)))]

    print(f"\nTesting batch [{batch_start}, {batch_start + len(batch_examples)})")
    for i, ex in enumerate(batch_examples):
        idx = batch_start + i
        print(f"  Example {idx}: {ex['image_path']}")

    # Encode batch
    batch = {
        "input_ids": [],
        "attention_mask": [],
        "bbox": [],
        "pixel_values": [],
        "labels": [],
    }

    for ex in batch_examples:
        encoded = encode_example(ex, processor, label_pad_token_id=-100, max_length=512)
        batch["input_ids"].append(encoded["input_ids"])
        batch["attention_mask"].append(encoded["attention_mask"])
        batch["bbox"].append(encoded["bbox"])
        batch["pixel_values"].append(encoded["pixel_values"])
        batch["labels"].append(encoded["labels"])

    batch = {
        "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long).cuda(),
        "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long).cuda(),
        "bbox": torch.tensor(batch["bbox"], dtype=torch.long).cuda(),
        "labels": torch.tensor(batch["labels"], dtype=torch.long).cuda(),
        "pixel_values": torch.stack(batch["pixel_values"]).cuda(),
    }

    # Try forward pass
    try:
        with torch.no_grad():
            outputs = model(**batch)
        print(f"  ✓ Batch OK (loss: {outputs.loss:.4f})")
    except Exception as e:
        print(f"  ❌ BATCH FAILED: {e}")
        print(f"\nNarrowing down to individual examples...")

        # Test each example individually
        for i, ex in enumerate(batch_examples):
            idx = batch_start + i
            print(f"\n  Testing individual example {idx}: {ex['image_path']}")

            encoded = encode_example(ex, processor, label_pad_token_id=-100, max_length=512)
            single_batch = {
                "input_ids": torch.tensor([encoded["input_ids"]], dtype=torch.long).cuda(),
                "attention_mask": torch.tensor([encoded["attention_mask"]], dtype=torch.long).cuda(),
                "bbox": torch.tensor([encoded["bbox"]], dtype=torch.long).cuda(),
                "labels": torch.tensor([encoded["labels"]], dtype=torch.long).cuda(),
                "pixel_values": encoded["pixel_values"].unsqueeze(0).cuda(),
            }

            try:
                with torch.no_grad():
                    outputs = model(**single_batch)
                print(f"    ✓ Example {idx} OK")
            except Exception as e2:
                print(f"    ❌ Example {idx} FAILED: {e2}")
                print(f"\n{'!' * 80}")
                print(f"PROBLEMATIC EXAMPLE FOUND: Index {idx}")
                print(f"{'!' * 80}")
                print(f"Image: {ex['image_path']}")
                print(f"Words: {len(ex['words'])}")
                print(f"Labels: {set(ex['labels'])}")
                print(f"Bbox range: [{min(min(b) for b in ex['bboxes'])}, {max(max(b) for b in ex['bboxes'])}]")

                # Check input_ids
                print(f"\nEncoded input_ids range: [{single_batch['input_ids'].min()}, {single_batch['input_ids'].max()}]")
                print(f"Encoded bbox range: [{single_batch['bbox'].min()}, {single_batch['bbox'].max()}]")

                sys.exit(1)

        sys.exit(1)

print("\n" + "=" * 80)
print("All tested examples passed!")
print("=" * 80)
