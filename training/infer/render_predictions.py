"""Render prediction overlays for a page."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def draw_overlay(pred_json: Path, out_path: Path) -> None:
    data = json.loads(pred_json.read_text())
    image = Image.open(data["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = ["red", "blue", "green", "purple", "orange", "brown"]
    color_map = {}
    for tok in data.get("tokens", []):
        lbl = tok["pred_label"]
        if lbl == "O":
            continue
        if lbl not in color_map:
            color_map[lbl] = colors[len(color_map) % len(colors)]
        c = color_map[lbl]
        x0, y0, x1, y1 = tok["bbox"]
        draw.rectangle([x0, y0, x1, y1], outline=c, width=2)
        draw.text((x0 + 1, y0 + 1), lbl, fill=c)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render overlay from prediction JSON")
    parser.add_argument("--pred_json", required=True)
    parser.add_argument("--out_png", required=True)
    args = parser.parse_args()
    draw_overlay(Path(args.pred_json), Path(args.out_png))
    print(f"Wrote overlay -> {args.out_png}")


if __name__ == "__main__":
    main()
