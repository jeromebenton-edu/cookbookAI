"""Helpers for encoding examples and aligning word labels to subwords."""

from __future__ import annotations

import os
from typing import List, Optional

import torch
from PIL import Image
from transformers import LayoutLMv3Processor


def align_labels_with_word_ids(word_ids: List[Optional[int]], word_labels: List[int], pad_token_id: int = -100) -> List[int]:
    aligned_labels: List[int] = []
    prev_word_id: Optional[int] = None
    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(pad_token_id)
        elif word_id != prev_word_id:
            aligned_labels.append(word_labels[word_id])
        else:
            aligned_labels.append(pad_token_id)
        prev_word_id = word_id
    return aligned_labels


_encode_debug_printed = False


def encode_example(
    example: dict,
    processor: LayoutLMv3Processor,
    label_pad_token_id: int = -100,
    max_length: int = 512,
) -> dict:
    """
    Returns:
    - input_ids, attention_mask: list[int] len=max_length
    - bbox: list[list[int]] shape [max_length, 4]
    - labels: list[int] len=max_length
    - pixel_values: torch.FloatTensor [3, H, W]
    """
    global _encode_debug_printed

    image = Image.open(example["image_path"]).convert("RGB")
    words = example["words"]
    boxes = example["bboxes"]
    word_labels = example["labels"]

    encoding = processor(
        image,
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        return_offsets_mapping=False,
        return_attention_mask=True,
    )

    # squeeze batch dim; processor always returns a batch
    input_ids = encoding["input_ids"].squeeze(0)  # [L]
    attention_mask = encoding["attention_mask"].squeeze(0)  # [L]
    bbox = encoding["bbox"].squeeze(0)  # [L, 4]
    pixel_values = encoding["pixel_values"].squeeze(0).to(torch.float32)  # [3, H, W]

    seq_len = input_ids.shape[0]
    word_ids = encoding.word_ids(batch_index=0)
    labels = align_labels_with_word_ids(word_ids, word_labels, pad_token_id=label_pad_token_id)

    # pad/truncate labels and bbox to exactly seq_len/max_length
    if len(labels) > seq_len:
        labels = labels[:seq_len]
    elif len(labels) < seq_len:
        labels = labels + [label_pad_token_id] * (seq_len - len(labels))

    if bbox.shape[0] > seq_len:
        bbox = bbox[:seq_len]
    elif bbox.shape[0] < seq_len:
        pad_rows = torch.zeros((seq_len - bbox.shape[0], 4), dtype=bbox.dtype)
        bbox = torch.cat([bbox, pad_rows], dim=0)

    assert seq_len == max_length or seq_len == len(input_ids), "unexpected sequence length from processor"
    assert len(input_ids) == len(attention_mask) == len(labels) == bbox.shape[0], "sequence length mismatch after encoding"

    if os.environ.get("ENCODE_DEBUG") == "1" and not _encode_debug_printed:
        _encode_debug_printed = True
        print(
            f"[ENCODE_DEBUG] input_ids {input_ids.shape}, attention_mask {attention_mask.shape}, "
            f"bbox {bbox.shape}, labels {len(labels)}, pixel_values {pixel_values.shape}"
        )

    return {
        "input_ids": input_ids.to(torch.int64).tolist(),
        "attention_mask": attention_mask.to(torch.int64).tolist(),
        "bbox": bbox.to(dtype=torch.int64).tolist(),
        "labels": labels,
        "pixel_values": pixel_values,
    }
