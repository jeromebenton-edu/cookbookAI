#!/usr/bin/env python3
"""Test model predictions on new demo pages."""

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
model = LayoutLMv3ForTokenClassification.from_pretrained("models/layoutlmv3_v3_production")

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

demo_ds = ds_dict["demo_eval"]

print(f"Testing {len(demo_ds)} demo pages")
print("=" * 80)

id2label = label_config["id2label"]

for idx, example in enumerate(demo_ds):
    print(f"\nPage {idx + 1}: {example['image_path']}")
    print(f"  Ground truth labels: {set(example['labels'])}")

    # Encode
    encoded = encode_example(example, processor, label_pad_token_id=-100, max_length=512)

    # Create batch (add batch dimension)
    batch = {
        "input_ids": torch.tensor([encoded["input_ids"]], dtype=torch.long).to(device),
        "attention_mask": torch.tensor([encoded["attention_mask"]], dtype=torch.long).to(device),
        "bbox": torch.tensor([encoded["bbox"]], dtype=torch.long).to(device),
        "pixel_values": encoded["pixel_values"].unsqueeze(0).to(device),
    }

    # Predict
    with torch.no_grad():
        outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)

    # Get predictions (remove batch dimension and convert to list)
    pred_ids = predictions[0].cpu().tolist()

    # Filter out padding tokens
    attention_mask = encoded["attention_mask"]
    valid_preds = [pred_ids[i] for i in range(len(pred_ids)) if attention_mask[i] == 1]

    # Count predictions
    pred_counts = {}
    for pred_id in valid_preds:
        label_name = id2label[pred_id]
        pred_counts[label_name] = pred_counts.get(label_name, 0) + 1

    print(f"  Predicted labels: {pred_counts}")

    # Check for RECIPE_TITLE predictions
    recipe_title_count = pred_counts.get("RECIPE_TITLE", 0)
    if recipe_title_count > 0:
        print(f"  ✓ Found {recipe_title_count} RECIPE_TITLE predictions!")

        # Find which words were predicted as RECIPE_TITLE
        words = example["words"]
        title_words = []
        word_idx = 0
        for i, (pred_id, mask) in enumerate(zip(pred_ids, attention_mask)):
            if mask == 0:
                continue
            if word_idx >= len(words):
                break

            label_name = id2label[pred_id]
            if label_name == "RECIPE_TITLE":
                title_words.append(words[word_idx])
            word_idx += 1

        print(f"  Title words: {' '.join(title_words[:20])}{'...' if len(title_words) > 20 else ''}")

print("\n" + "=" * 80)
print("DEMO EVALUATION COMPLETE")
print("=" * 80)
