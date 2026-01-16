#!/usr/bin/env python3
"""Convert weak labels JSONL into Label Studio tasks with pre-annotations.

The weak-label JSONL produced by training/labeling/run_weak_labeling.py is not
directly importable into Label Studio. This helper turns each page into a Label
Studio task that points at the image file and includes rectangle predictions
for every line box.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def load_records(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def line_to_result(line: dict, page_width: int, page_height: int) -> dict:
    """Convert a single weak-labeled line into a Label Studio rectangle result."""
    x0, y0, x1, y1 = line["line_bbox"]
    return {
        "id": f"{line.get('line_id')}",
        "from_name": "line_labels",
        "to_name": "image",
        "type": "rectanglelabels",
        "score": line.get("confidence"),
        "value": {
            # Label Studio expects percentages, not absolute pixels.
            "x": x0 / page_width * 100,
            "y": y0 / page_height * 100,
            "width": (x1 - x0) / page_width * 100,
            "height": (y1 - y0) / page_height * 100,
            "rotation": 0,
            "rectanglelabels": [line["label"]],
        },
    }


def convert(
    weak_labels: Path,
    image_root: Path,
    out_path: Path,
    use_local_files_url: bool = False,
    url_prefix: str | None = None,
    embed_images: bool = False,
) -> int:
    tasks = []
    for rec in load_records(weak_labels):
        img_path = Path(rec["image_path"])
        if not img_path.is_absolute():
            img_path = (image_root / img_path).resolve()
        if embed_images:
            import base64
            mime = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
            data = img_path.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            img_value = f"data:{mime};base64,{b64}"
        elif url_prefix:
            try:
                rel = img_path.relative_to(image_root.resolve())
            except ValueError:
                rel = img_path.name
            img_value = f"{url_prefix.rstrip('/')}/{rel.as_posix()}"
        elif use_local_files_url:
            img_value = f"/data/local-files/?d={img_path}"
        else:
            img_value = str(img_path)

        lines = rec.get("lines") or []
        results = [
            line_to_result(line, rec["width"], rec["height"])
            for line in lines
            if "line_bbox" in line
        ]

        tasks.append(
            {
                "data": {
                    "image": img_value,
                    "page_num": rec.get("page_num"),
                },
                "predictions": [
                    {
                        "result": results,
                        "model_version": "weak_labels",
                        "score": rec.get("page_label_quality", {}).get(
                            "avg_line_confidence"
                        ),
                    }
                ],
            }
        )

    with out_path.open("w") as out_f:
        if out_path.suffix.lower() == ".jsonl":
            for task in tasks:
                out_f.write(json.dumps(task) + "\n")
        else:
            json.dump(tasks, out_f)
    return len(tasks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert weak labels JSONL into Label Studio tasks."
    )
    parser.add_argument(
        "--weak-labels",
        type=Path,
        default=Path("data/labels/boston_weak_labeled.jsonl"),
        help="Path to weak labels JSONL.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("."),
        help="Base dir to resolve relative image paths.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/labels/labelstudio_tasks.json"),
        help="Output path. Use .json for a list or .jsonl for one-task-per-line.",
    )
    parser.add_argument(
        "--use-local-files-url",
        action="store_true",
        help="Emit /data/local-files/?d=... URLs so Label Studio can serve local files without extra storage setup.",
    )
    parser.add_argument(
        "--url-prefix",
        type=str,
        help="Optional base URL (e.g., http://localhost:9000) to prefix image paths. Paths will be relative to --image-root if possible.",
    )
    parser.add_argument(
        "--embed-images",
        action="store_true",
        help="Embed images as data URLs to avoid any serving/CORS issues (increases JSON size).",
    )
    args = parser.parse_args()

    count = convert(
        args.weak_labels,
        args.image_root,
        args.out,
        use_local_files_url=args.use_local_files_url,
        url_prefix=args.url_prefix,
        embed_images=args.embed_images,
    )
    print(f"Wrote {count} Label Studio tasks -> {args.out}")


if __name__ == "__main__":
    main()
