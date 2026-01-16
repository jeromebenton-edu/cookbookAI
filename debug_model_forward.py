#!/usr/bin/env python3
"""Debug model forward pass with actual batch data."""

import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

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

print(f"\nModel config:")
print(f"  num_labels: {model.config.num_labels}")
print(f"  id2label: {model.config.id2label}")

# Check model architecture
print(f"\nModel architecture:")
print(f"  Classifier layer: {model.classifier}")

# Get classifier weight shape
if hasattr(model, 'classifier'):
    classifier = model.classifier
    print(f"  Classifier type: {type(classifier)}")
    if hasattr(classifier, 'weight'):
        print(f"  Classifier weight shape: {classifier.weight.shape}")
    if hasattr(classifier, 'out_proj'):
        print(f"  Classifier out_proj weight shape: {classifier.out_proj.weight.shape}")

# Try a forward pass with one example
print("\n" + "=" * 80)
print("TESTING FORWARD PASS")
print("=" * 80)

example = ds_dict["train"][0]
print(f"\nEncoding example...")
encoded = encode_example(example, processor, label_pad_token_id=-100, max_length=512)

# Create batch (batch_size=1)
batch = {
    "input_ids": torch.tensor([encoded["input_ids"]], dtype=torch.long),
    "attention_mask": torch.tensor([encoded["attention_mask"]], dtype=torch.long),
    "bbox": torch.tensor([encoded["bbox"]], dtype=torch.long),
    "labels": torch.tensor([encoded["labels"]], dtype=torch.long),
    "pixel_values": encoded["pixel_values"].unsqueeze(0),
}

print(f"\nBatch shapes:")
print(f"  input_ids: {batch['input_ids'].shape}")
print(f"  attention_mask: {batch['attention_mask'].shape}")
print(f"  bbox: {batch['bbox'].shape}")
print(f"  labels: {batch['labels'].shape}")
print(f"  pixel_values: {batch['pixel_values'].shape}")

print(f"\nBatch value ranges:")
print(f"  input_ids: [{batch['input_ids'].min()}, {batch['input_ids'].max()}]")
print(f"  attention_mask: [{batch['attention_mask'].min()}, {batch['attention_mask'].max()}]")
print(f"  bbox: [{batch['bbox'].min()}, {batch['bbox'].max()}]")
print(f"  labels (excluding -100): [{batch['labels'][batch['labels'] != -100].min()}, {batch['labels'][batch['labels'] != -100].max()}]")

# Move to CPU for debugging
model = model.cpu()
batch = {k: v.cpu() for k, v in batch.items()}

print(f"\nRunning forward pass on CPU...")
try:
    with torch.no_grad():
        outputs = model(**batch)
    print(f"✓ Forward pass successful!")
    print(f"  Loss: {outputs.loss}")
    print(f"  Logits shape: {outputs.logits.shape}")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()

# Now try with CUDA if available
if torch.cuda.is_available():
    print(f"\n" + "=" * 80)
    print("TESTING FORWARD PASS ON CUDA")
    print("=" * 80)

    model = model.cuda()
    batch = {k: v.cuda() for k, v in batch.items()}

    print(f"\nRunning forward pass on CUDA...")
    try:
        with torch.no_grad():
            outputs = model(**batch)
        print(f"✓ CUDA forward pass successful!")
        print(f"  Loss: {outputs.loss}")
        print(f"  Logits shape: {outputs.logits.shape}")
    except Exception as e:
        print(f"❌ CUDA forward pass failed: {e}")
        import traceback
        traceback.print_exc()

        # Try to get more info about the error
        print(f"\nDEBUGGING CUDA ERROR...")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Batch device: {batch['input_ids'].device}")

        # Check embeddings
        print(f"\nModel embeddings:")
        if hasattr(model.layoutlmv3, 'embeddings'):
            embeddings = model.layoutlmv3.embeddings
            if hasattr(embeddings, 'word_embeddings'):
                print(f"  word_embeddings.num_embeddings: {embeddings.word_embeddings.num_embeddings}")
                print(f"  word_embeddings.embedding_dim: {embeddings.word_embeddings.embedding_dim}")
                print(f"  Max input_id: {batch['input_ids'].max()}")
                if batch['input_ids'].max() >= embeddings.word_embeddings.num_embeddings:
                    print(f"  ❌ input_ids contains values >= num_embeddings!")

print("\n" + "=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)
