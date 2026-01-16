#!/usr/bin/env python3
"""
Evaluate RECIPE_TITLE detection accuracy across the dataset.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import load_from_disk
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image


def evaluate_title_accuracy(sample_size=50):
    """Check how often the model correctly predicts RECIPE_TITLE tokens."""

    model_path = PROJECT_ROOT / "models/layoutlmv3_v3_manual_59pages_balanced"
    dataset_path = PROJECT_ROOT / "data/datasets/boston_layoutlmv3_v3_manual_expanded2/dataset_dict"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = LayoutLMv3Processor.from_pretrained(model_path)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_path).to(device)
    model.eval()

    dataset = load_from_disk(str(dataset_path))

    # Get pages with recipe titles in ground truth
    pages_with_titles = []
    for split in dataset.keys():
        for idx in range(len(dataset[split])):
            data = dataset[split][idx]
            if 'RECIPE_TITLE' in data['label_names']:
                page_num = int(data['page_num'])
                if 69 <= page_num <= 535:  # Only recipe pages
                    pages_with_titles.append((split, idx, page_num))

    print(f"Found {len(pages_with_titles)} pages with RECIPE_TITLE labels")
    print(f"Evaluating sample of {min(sample_size, len(pages_with_titles))} pages...\n")

    correct_pages = 0
    partial_pages = 0
    wrong_pages = 0

    import random
    random.seed(42)
    sample = random.sample(pages_with_titles, min(sample_size, len(pages_with_titles)))

    for split, idx, page_num in sample:
        data = dataset[split][idx]

        # Get ground truth titles
        gt_titles = set()
        for word, label in zip(data['words'], data['label_names']):
            if label == 'RECIPE_TITLE':
                gt_titles.add(word)

        # Run inference
        image = Image.open(data['image_path'])
        encoding = processor(
            image,
            data["words"],
            boxes=data["bboxes"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        )

        with torch.no_grad():
            outputs = model(
                input_ids=encoding["input_ids"].to(device),
                bbox=encoding["bbox"].to(device),
                pixel_values=encoding["pixel_values"].to(device),
            )

        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        id2label = model.config.id2label

        # Get predicted titles
        pred_titles = set()
        for word_idx, word in enumerate(data['words']):
            token_idx = word_idx + 1
            if token_idx < len(predictions):
                label = id2label[predictions[token_idx]]
                if label == 'RECIPE_TITLE':
                    pred_titles.add(word)

        # Calculate overlap
        if pred_titles == gt_titles:
            correct_pages += 1
            status = "✓ CORRECT"
        elif len(pred_titles & gt_titles) > 0:
            partial_pages += 1
            status = "~ PARTIAL"
        else:
            wrong_pages += 1
            status = "✗ WRONG"

        overlap = len(pred_titles & gt_titles)
        print(f"Page {page_num:3d}: {status:10s}  GT: {len(gt_titles):2d} words, Pred: {len(pred_titles):2d} words, Overlap: {overlap:2d}")

        if status == "✗ WRONG" and len(gt_titles) > 0:
            gt_text = " ".join(sorted(gt_titles)[:5])
            pred_text = " ".join(sorted(pred_titles)[:5]) if pred_titles else "(none)"
            print(f"         GT:   {gt_text[:60]}")
            print(f"         Pred: {pred_text[:60]}")

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Correct (100% match): {correct_pages:3d} / {len(sample)} ({100*correct_pages/len(sample):.1f}%)")
    print(f"Partial (some match): {partial_pages:3d} / {len(sample)} ({100*partial_pages/len(sample):.1f}%)")
    print(f"Wrong (no match):     {wrong_pages:3d} / {len(sample)} ({100*wrong_pages/len(sample):.1f}%)")


if __name__ == "__main__":
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    evaluate_title_accuracy(sample_size)
