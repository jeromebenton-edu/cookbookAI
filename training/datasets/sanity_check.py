"""Sample pages from a saved DatasetDict and render bbox overlays."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from datasets import load_from_disk
from PIL import Image, ImageDraw


def load_label_map(dataset_dir: Path) -> Dict[int, str]:
    label_map_path = dataset_dir / "label_map.json"
    data = json.loads(label_map_path.read_text())
    id2label = {int(k): v for k, v in data["id2label"].items()}
    return id2label


def to_pixels(bbox_norm: List[int], width: int, height: int) -> List[int]:
    x0 = int(bbox_norm[0] / 1000 * width)
    y0 = int(bbox_norm[1] / 1000 * height)
    x1 = int(bbox_norm[2] / 1000 * width)
    y1 = int(bbox_norm[3] / 1000 * height)
    return [x0, y0, x1, y1]


def render_overlay(sample: dict, id2label: Dict[int, str], out_path: Path) -> None:
    img = Image.open(sample["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(img)
    # Pillow default font is fine for tiny labels
    for bbox_norm, label_id in zip(sample["bboxes"], sample["labels"]):
        label_name = id2label.get(int(label_id), "UNK")
        if label_name == "O":
            continue
        x0, y0, x1, y1 = to_pixels(bbox_norm, sample["width"], sample["height"])
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        draw.text((x0 + 1, y0 + 1), label_name, fill="red")
    img.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check overlays for LayoutLM datasets")
    parser.add_argument("--dataset_dir", required=True, help="Directory containing dataset_dict and label_map.json")
    parser.add_argument("--split", default="train", help="Split to sample from")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of pages to sample")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed")
    parser.add_argument("--out_dir", default="data/reports/sanity_overlays", help="Output dir for overlay PNGs")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dataset_path = Path(args.dataset_dir) / "dataset_dict"
    ds = load_from_disk(str(dataset_path))
    id2label = load_label_map(Path(args.dataset_dir))
    split = ds.get(args.split)
    if split is None or len(split) == 0:
        raise SystemExit(f"Split '{args.split}' is empty or missing")

    indices = list(range(len(split)))
    rng.shuffle(indices)
    indices = indices[: args.num_samples]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        sample = split[int(idx)]
        tokens = sample["words"][:25]
        labels = [id2label.get(int(l), "UNK") for l in sample["labels"][:25]]
        token_info = ", ".join(f"{t}({l})" for t, l in zip(tokens, labels))
        print(f"page {sample['page_num']} :: {token_info}")
        out_path = out_dir / f"page_{int(sample['page_num']):04d}_overlay.png"
        render_overlay(sample, id2label, out_path)
        print(f"wrote overlay -> {out_path}")


if __name__ == "__main__":
    main()
