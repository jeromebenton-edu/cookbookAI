"""Convert OCR JSONL into a Parquet dataset with derived features.

Example:
python data/scripts/jsonl_to_parquet.py \
  --in data/ocr/boston_pages.jsonl \
  --out data/ocr/boston_pages.parquet
"""

import argparse
import json
import os
import pathlib
import sys
from typing import Any, Dict, Iterator, List

import pandas as pd

REQUIRED_FIELDS = [
    "page_num",
    "image_path",
    "width",
    "height",
    "words",
    "bboxes",
]


def stream_jsonl(path: pathlib.Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def derive_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    words: List[str] = record.get("words", []) or []
    confidences: List[float] = record.get("confidences") or []
    token_count = len(words)
    unique_token_count = len(set(words)) if words else 0
    avg_confidence = sum(confidences) / len(confidences) if confidences else None

    # crude line_count estimate from punctuation/newlines in text if present
    text = record.get("text") or ""
    if text:
        line_count = len([ln for ln in text.splitlines() if ln.strip()])
    else:
        # fallback: approximate via token count / 8
        line_count = max(1, token_count // 8) if token_count else 0

    record["token_count"] = token_count
    record["unique_token_count"] = unique_token_count
    record["line_count"] = line_count
    record["avg_confidence"] = avg_confidence
    return record


def validate(record: Dict[str, Any]) -> bool:
    for field in REQUIRED_FIELDS:
        if field not in record:
            return False
    return True


def convert_jsonl_to_parquet(in_path: pathlib.Path, out_path: pathlib.Path) -> None:
    rows: List[Dict[str, Any]] = []
    for record in stream_jsonl(in_path):
        if not validate(record):
            continue
        rows.append(derive_fields(record))

    if not rows:
        raise ValueError("No valid records found to convert.")

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {len(df)} rows to {out_path} ({file_size_mb:.2f} MB)")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert OCR JSONL to Parquet")
    parser.add_argument("--in", dest="input_path", required=True, help="Input JSONL path")
    parser.add_argument("--out", dest="output_path", required=True, help="Output Parquet path")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    convert_jsonl_to_parquet(pathlib.Path(args.input_path), pathlib.Path(args.output_path))


if __name__ == "__main__":
    main(sys.argv[1:])
