"""Quick sanity evaluation against weak labels (not gold)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, List

import torch
from datasets import load_from_disk
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

from training.eval.metrics import compute_seqeval, save_confusion_csv, save_report


def to_py(obj: Any):
    """Recursively convert numpy/torch scalars and arrays to plain Python for JSON safety."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_py(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_py(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def infer_page(model, processor, example: dict, max_length: int = 512):
    image = Image.open(example["image_path"]).convert("RGB")
    encoding = processor(
        image,
        example["words"],
        boxes=example["bboxes"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    word_ids = encoding.word_ids()
    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            bbox=encoding["bbox"],
            pixel_values=encoding["pixel_values"],
        )
        logits = outputs.logits
        preds = logits.argmax(-1).squeeze(0).tolist()

    pred_labels = []
    true_labels = []
    seen = set()
    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen:
            continue
        seen.add(word_id)
        pred_labels.append(model.config.id2label[preds[idx]])
        true_labels.append(model.config.id2label[example["labels"][word_id]])
    return true_labels, pred_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions against weak labels (sanity only)")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--dataset_dir", required=True, help="Dataset dir containing dataset_dict")
    parser.add_argument("--split", default="val")
    parser.add_argument("--num_pages", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--selftest", action="store_true", help="Run a JSON serialization self-test and exit")
    args = parser.parse_args()

    if args.selftest:
        dummy = {"a": 1, "b": 2.5, "c": None}
        import numpy as np

        dummy["np_int"] = np.int64(5)
        dummy["np_arr"] = np.array([1, 2, 3])
        dummy["nested"] = {"x": np.float32(1.2)}
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(to_py(dummy)))
        print("Self-test JSON serialization OK ->", args.out_json)
        return

    rng = random.Random(args.seed)
    ds_path = Path(args.dataset_dir)
    if ds_path.name == "dataset_dict":
        ds = load_from_disk(str(ds_path))
    else:
        ds = load_from_disk(str(ds_path / "dataset_dict"))
    split = ds[args.split] if args.split in ds else ds["train"]
    indices = list(range(len(split)))
    rng.shuffle(indices)
    indices = indices[: args.num_pages]

    model = LayoutLMv3ForTokenClassification.from_pretrained(args.model_dir)
    processor = LayoutLMv3Processor.from_pretrained(args.model_dir, apply_ocr=False)

    y_true: List[List[str]] = []
    y_pred: List[List[str]] = []
    for idx in indices:
        ex = split[int(idx)]
        t, p = infer_page(model, processor, ex, max_length=args.max_length)
        y_true.append(t)
        y_pred.append(p)

    metrics = compute_seqeval(y_true, y_pred)
    metrics = to_py(metrics)
    save_report(metrics, Path(args.out_json), Path(args.out_md))
    save_confusion_csv(metrics, Path(args.out_csv))
    print(f"Eval pages: {len(indices)} | F1: {metrics['f1']:.3f}")


if __name__ == "__main__":
    main()
