"""Run inference on a single page with a fine-tuned LayoutLMv3 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch
from datasets import load_from_disk
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

from training.modeling.alignment import align_labels_with_word_ids


def load_words_bboxes_from_json(path: Path):
    data = json.loads(path.read_text())
    return data["words"], data["bboxes"]


def load_from_dataset(dataset_dir: Path, page_num: int):
    ds = load_from_disk(str(dataset_dir / "dataset_dict"))
    for split in ["train", "val", "test"]:
        if split in ds:
            rows = ds[split].filter(lambda ex: ex["page_num"] == page_num)
            if len(rows) > 0:
                ex = rows[0]
                return ex["words"], ex["bboxes"], ex["image_path"]
    raise ValueError(f"page_num {page_num} not found in dataset {dataset_dir}")


def predict(model, processor, image_path: Path, words: List[str], bboxes: List[List[int]]) -> dict:
    image = Image.open(image_path).convert("RGB")
    encoding = processor(
        image,
        words,
        boxes=bboxes,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        max_length=512,
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
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_ids = probs.argmax(-1).squeeze(0).tolist()
        confidences = probs.max(-1).values.squeeze(0).tolist()

    tokens = []
    seen_words = set()
    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen_words:
            continue
        seen_words.add(word_id)
        tokens.append(
            {
                "text": words[word_id],
                "bbox": bboxes[word_id],
                "pred_label": model.config.id2label[pred_ids[idx]],
                "pred_id": pred_ids[idx],
                "confidence": float(confidences[idx]),
            }
        )
    grouped = {}
    for tok in tokens:
        grouped.setdefault(tok["pred_label"], []).append(tok)
    return {"tokens": tokens, "grouped": grouped}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict labels for a page with LayoutLMv3")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--image", help="Path to page image")
    parser.add_argument("--words_json", help="JSON with words and bboxes")
    parser.add_argument("--dataset_dir", help="Dataset dir (with dataset_dict) to load words/bboxes")
    parser.add_argument("--page_num", type=int, help="Page number when loading from dataset")
    parser.add_argument("--out_json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = LayoutLMv3ForTokenClassification.from_pretrained(args.model_dir)
    processor = LayoutLMv3Processor.from_pretrained(args.model_dir, apply_ocr=False)

    if args.words_json:
        words, bboxes = load_words_bboxes_from_json(Path(args.words_json))
        image_path = Path(args.image)
    elif args.dataset_dir and args.page_num is not None:
        words, bboxes, image_path = load_from_dataset(Path(args.dataset_dir), args.page_num)
        image_path = Path(image_path)
    else:
        raise SystemExit("Provide either --words_json and --image, or --dataset_dir and --page_num")

    pred = predict(model, processor, image_path, words, bboxes)
    output = {
        "page_num": args.page_num,
        "image_path": str(image_path),
        "tokens": pred["tokens"],
        "grouped": pred["grouped"],
        "label_map": model.config.id2label,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(output, indent=2))
    print(f"Wrote predictions -> {args.out_json}")


if __name__ == "__main__":
    main()
