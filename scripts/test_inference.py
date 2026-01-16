#!/usr/bin/env python3
"""Quick inference test for the balanced LayoutLMv3 model."""

from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from datasets import load_from_disk
import torch

# Load model
model_path = "models/layoutlmv3_v3_manual_59pages_balanced"
print(f"Loading model from {model_path}...")
processor = LayoutLMv3Processor.from_pretrained(model_path)
model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
model.eval()

# Get label mapping from model config
id2label = model.config.id2label

print(f"Model loaded with {len(id2label)} labels")

# Load dataset to get a test example
print("\nLoading test dataset...")
ds = load_from_disk('data/datasets/boston_layoutlmv3_v3_manual_expanded2/dataset_dict')

# Get a demo eval example
example = ds['demo_eval'][0]
print(f"Testing on page {example.get('page_num', 'unknown')}...")

# Load image
image = Image.open(example['image_path']).convert("RGB")
words = [w for w in example['words'] if w]  # Filter empty strings
boxes = example['bboxes'][:len(words)]  # Match length

# Prepare inputs
encoding = processor(
    image,
    words,
    boxes=boxes,
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_length=512
)

# Run inference
print("Running inference...")
with torch.no_grad():
    outputs = model(**encoding)
    predictions = outputs.logits.argmax(-1).squeeze().tolist()

# Decode predictions
print(f"\n{'Word':<30} {'True Label':<20} {'Predicted Label':<20}")
print("-" * 70)

true_labels = example['labels']
correct = 0
total = 0

for i, (word, true_id, pred_id) in enumerate(zip(words, true_labels, predictions)):
    if i >= len(words) or true_id == -100:  # Stop at actual words or skip padding
        break

    true_label = id2label.get(true_id, f"ID_{true_id}")
    pred_label = id2label.get(pred_id, f"ID_{pred_id}")

    # Only print non-O predictions or mismatches for clarity
    if pred_label != "O" or true_label != "O":
        match = "✓" if true_label == pred_label else "✗"
        print(f"{word:<30} {true_label:<20} {pred_label:<20} {match}")

    if true_label == pred_label:
        correct += 1
    total += 1

# Print accuracy
print(f"\nToken accuracy: {correct}/{total} ({100*correct/total:.1f}%)")

# Count label-specific stats
from collections import Counter
true_counter = Counter()
pred_counter = Counter()

for true_id, pred_id in zip(true_labels[:len(words)], predictions[:len(words)]):
    if true_id != -100:
        true_counter[id2label[true_id]] += 1
        pred_counter[id2label[pred_id]] += 1

print("\nLabel distribution:")
print(f"{'Label':<20} {'True':<10} {'Predicted':<10}")
print("-" * 40)
for label in sorted(set(true_counter.keys()) | set(pred_counter.keys())):
    if label != "O":  # Skip O for brevity
        print(f"{label:<20} {true_counter[label]:<10} {pred_counter[label]:<10}")

print("\nInference complete!")
