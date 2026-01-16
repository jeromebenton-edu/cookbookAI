#!/usr/bin/env python3
"""Debug training data to find out-of-bounds values."""

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

# Load label config
label_config = get_label_config(version="v3")
print(f"Label config: {label_config['num_labels']} labels")
print(f"Labels: {label_config['labels']}")

# Check first 5 training examples
print("\n" + "=" * 80)
print("CHECKING FIRST 5 TRAINING EXAMPLES")
print("=" * 80)

train_ds = ds_dict["train"]

for i in range(min(5, len(train_ds))):
    example = train_ds[i]
    print(f"\nExample {i}:")
    print(f"  Image: {example['image_path']}")
    print(f"  Words: {len(example['words'])} words")
    print(f"  Bboxes: {len(example['bboxes'])} bboxes")
    print(f"  Labels: {len(example['labels'])} labels")

    # Check raw labels
    raw_labels = example["labels"]
    unique_labels = set(raw_labels)
    print(f"  Unique label IDs: {sorted(unique_labels)}")
    print(f"  Label range: [{min(raw_labels)}, {max(raw_labels)}]")

    # Check for invalid labels
    invalid_labels = [lbl for lbl in raw_labels if lbl < 0 or lbl >= label_config["num_labels"]]
    if invalid_labels:
        print(f"  ❌ INVALID LABELS FOUND: {invalid_labels[:10]}")
    else:
        print(f"  ✓ All raw labels valid")

    # Check bboxes
    bboxes = example["bboxes"]
    flat_coords = [coord for bbox in bboxes for coord in bbox]
    if flat_coords:
        min_coord = min(flat_coords)
        max_coord = max(flat_coords)
        print(f"  Bbox coordinate range: [{min_coord}, {max_coord}]")

        # LayoutLMv3 expects coords in [0, 1000]
        if min_coord < 0 or max_coord > 1000:
            print(f"  ⚠️  WARNING: Bbox coords outside [0, 1000] range!")

    # Encode example
    try:
        encoded = encode_example(example, processor, label_pad_token_id=-100, max_length=512)

        print(f"\n  After encoding:")
        print(f"    input_ids: len={len(encoded['input_ids'])}, dtype=list")
        print(f"    attention_mask: len={len(encoded['attention_mask'])}, dtype=list")
        print(f"    bbox: len={len(encoded['bbox'])}, shape=[{len(encoded['bbox'])}, 4]")
        print(f"    labels: len={len(encoded['labels'])}, dtype=list")
        print(f"    pixel_values: shape={encoded['pixel_values'].shape}")

        # Check encoded labels
        encoded_labels = [lbl for lbl in encoded["labels"] if lbl != -100]
        if encoded_labels:
            unique_encoded = set(encoded_labels)
            print(f"    Unique encoded label IDs (excluding -100): {sorted(unique_encoded)}")
            print(f"    Encoded label range: [{min(encoded_labels)}, {max(encoded_labels)}]")

            invalid_encoded = [lbl for lbl in encoded_labels if lbl < 0 or lbl >= label_config["num_labels"]]
            if invalid_encoded:
                print(f"    ❌ INVALID ENCODED LABELS: {invalid_encoded[:10]}")
            else:
                print(f"    ✓ All encoded labels valid")

        # Check encoded bboxes
        encoded_bboxes = encoded["bbox"]
        flat_encoded_coords = [coord for bbox in encoded_bboxes for coord in bbox]
        if flat_encoded_coords:
            min_enc_coord = min(flat_encoded_coords)
            max_enc_coord = max(flat_encoded_coords)
            print(f"    Encoded bbox range: [{min_enc_coord}, {max_enc_coord}]")

            if min_enc_coord < 0 or max_enc_coord > 1000:
                print(f"    ⚠️  WARNING: Encoded bbox coords outside [0, 1000]!")

        # Create tensors to simulate what happens in collate_fn
        print(f"\n  Creating tensors (simulating collate_fn):")
        input_ids_tensor = torch.tensor(encoded["input_ids"], dtype=torch.long)
        attention_mask_tensor = torch.tensor(encoded["attention_mask"], dtype=torch.long)
        bbox_tensor = torch.tensor(encoded["bbox"], dtype=torch.long)
        labels_tensor = torch.tensor(encoded["labels"], dtype=torch.long)
        pixel_values_tensor = encoded["pixel_values"]

        print(f"    input_ids: {input_ids_tensor.shape}, dtype={input_ids_tensor.dtype}")
        print(f"    attention_mask: {attention_mask_tensor.shape}, dtype={attention_mask_tensor.dtype}")
        print(f"    bbox: {bbox_tensor.shape}, dtype={bbox_tensor.dtype}")
        print(f"    labels: {labels_tensor.shape}, dtype={labels_tensor.dtype}")
        print(f"    pixel_values: {pixel_values_tensor.shape}, dtype={pixel_values_tensor.dtype}")

        # Check for any NaN or inf values
        if torch.isnan(pixel_values_tensor).any():
            print(f"    ⚠️  WARNING: NaN values in pixel_values!")
        if torch.isinf(pixel_values_tensor).any():
            print(f"    ⚠️  WARNING: Inf values in pixel_values!")

        print(f"    ✓ Encoding successful")

    except Exception as e:
        print(f"    ❌ ENCODING FAILED: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)
