#!/usr/bin/env python3
"""
Inspect ML model predictions for a specific page to understand token labeling.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import load_from_disk
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification


def inspect_page(page_num: int):
    """Inspect all tokens and their predictions for a specific page."""

    # Load model and dataset
    model_path = PROJECT_ROOT / "models/layoutlmv3_v3_manual_59pages_balanced"
    dataset_path = PROJECT_ROOT / "data/datasets/boston_layoutlmv3_v3_manual_expanded2/dataset_dict"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    processor = LayoutLMv3Processor.from_pretrained(model_path)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_path).to(device)
    model.eval()

    dataset = load_from_disk(str(dataset_path))

    # Find the page in dataset
    page_data = None
    split_name = None
    for split in dataset.keys():
        for idx, num in enumerate(dataset[split]["page_num"]):
            if int(num) == page_num:
                page_data = dataset[split][idx]
                split_name = split
                break
        if page_data:
            break

    if not page_data:
        print(f"Page {page_num} not found in dataset")
        return

    print(f"Found page {page_num} in split: {split_name}")
    print(f"Image path: {page_data['image_path']}")
    print(f"Number of words: {len(page_data['words'])}\n")

    # Load image
    from PIL import Image
    image = Image.open(page_data['image_path'])

    # Prepare input
    encoding = processor(
        image,
        page_data["words"],
        boxes=page_data["bboxes"],
        return_tensors="pt",
        truncation=True,
        padding="max_length",
    )

    # Run inference
    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"].to(device),
            bbox=encoding["bbox"].to(device),
            pixel_values=encoding["pixel_values"].to(device),
        )

    # Get predictions
    predictions = outputs.logits.argmax(-1).squeeze().tolist()

    # Map label IDs to names
    id2label = model.config.id2label

    # Extract predicted labels for each word
    word_labels = []
    for word_idx, word in enumerate(page_data["words"]):
        # Find corresponding token in encoding
        # Note: processor may split words into multiple tokens
        token_idx = word_idx + 1  # +1 because of [CLS] token
        if token_idx < len(predictions):
            label_id = predictions[token_idx]
            label = id2label[label_id]
            bbox = page_data["bboxes"][word_idx]

            word_labels.append({
                "word": word,
                "label": label,
                "bbox": bbox,
                "position": (bbox[1], bbox[0])  # (y, x) for sorting
            })

    # Group by label
    print("=" * 80)
    print("TOKENS BY LABEL")
    print("=" * 80)

    label_groups = {}
    for item in word_labels:
        label = item["label"]
        if label != "O":  # Skip non-entity tokens
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(item)

    # Display each label group
    for label in sorted(label_groups.keys()):
        items = label_groups[label]
        print(f"\n{label} ({len(items)} tokens):")
        print("-" * 80)

        # Sort by position (top-to-bottom, left-to-right)
        sorted_items = sorted(items, key=lambda x: x["position"])

        for idx, item in enumerate(sorted_items):
            y, x = item["position"]
            bbox = item["bbox"]
            print(f"  {idx+1:2d}. '{item['word']:30s}' @ y={y:4d} x={x:4d}  bbox={bbox}")

        # Show combined text for RECIPE_TITLE
        if label == "RECIPE_TITLE":
            combined = " ".join(item["word"] for item in sorted_items)
            print(f"\n  COMBINED: {combined}")

            # Analyze stopping points
            print(f"\n  ANALYSIS:")
            for idx, item in enumerate(sorted_items):
                if item["word"].rstrip().endswith('.'):
                    print(f"    - Token {idx+1} ends with period: '{item['word']}'")
                    if idx + 1 < len(sorted_items):
                        next_item = sorted_items[idx + 1]
                        y_gap = next_item["bbox"][1] - item["bbox"][3]
                        print(f"      Y-gap to next token: {y_gap}px")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        page_num = int(sys.argv[1])
    else:
        page_num = 519  # Default to Gherkins page

    inspect_page(page_num)
