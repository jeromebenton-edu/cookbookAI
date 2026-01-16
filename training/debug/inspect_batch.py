"""Smoke-test batch shapes and a forward pass for LayoutLMv3."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

from training.train_layoutlmv3 import collate_fn, encode_dataset, load_label_map

LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a batch and run a single forward pass.")
    parser.add_argument("--dataset_dir", required=True, help="Path to dataset or dataset_dict folder.")
    parser.add_argument("--model", default="microsoft/layoutlmv3-base")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    return parser.parse_args()


def resolve_paths(dataset_dir: Path) -> tuple[Path, Path]:
    dataset_path = dataset_dir if dataset_dir.name == "dataset_dict" else dataset_dir / "dataset_dict"
    dataset_root = dataset_dir.parent if dataset_dir.name == "dataset_dict" else dataset_dir
    label_map = dataset_root / "label_map.json"
    fallback = dataset_root.parent / "label_map.json"
    if not label_map.exists() and fallback.exists():
        label_map = fallback
    if not label_map.exists():
        raise FileNotFoundError(f"Could not find label_map.json near {dataset_dir}")
    return dataset_path, label_map


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    dataset_path, label_map_path = resolve_paths(dataset_dir)

    ds = load_from_disk(str(dataset_path))
    label2id, id2label = load_label_map(label_map_path)
    label_list = [id2label[i] for i in sorted(id2label.keys())]

    processor = LayoutLMv3Processor.from_pretrained(args.model, apply_ocr=False)
    ds_enc = encode_dataset(ds, processor, args.max_length)

    if hasattr(ds_enc, "keys"):
        split = "train" if "train" in ds_enc else next(iter(ds_enc.keys()))
        base_ds = ds_enc[split]
    else:
        split = "train"
        base_ds = ds_enc

    dataloader = DataLoader(base_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    batch = next(iter(dataloader))
    print("Batch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {tuple(v.shape)}  dtype={v.dtype}")

    model.eval()
    with torch.no_grad():
        outputs = model(**batch)
    print("Forward pass ok. Logits shape:", tuple(outputs.logits.shape))


if __name__ == "__main__":
    main()
