#!/usr/bin/env python3
import os
from pathlib import Path

import requests

from .smoke_common import wait_for_health

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")


def main():
    wait_for_health()
    sample = Path("docs/samples/sample_clean_printed.png")
    if not sample.exists():
        raise SystemExit("Sample image missing; run scripts/generate_sample_assets.py")

    with sample.open("rb") as f:
        res = requests.post(f"{API_BASE}/api/upload/page", files={"file": f}, timeout=60)
    if not res.ok:
        raise SystemExit(f"Upload failed: {res.status_code} {res.text}")
    data = res.json()
    session = data["session_id"]
    print("Session", session)

    ocr = requests.get(f"{API_BASE}/api/upload/{session}/ocr", timeout=30).json()
    if not (ocr.get("words") and ocr.get("bboxes")):
        raise SystemExit("OCR schema invalid")
    if len(ocr["words"]) < 30:
        raise SystemExit("Too few OCR words")

    pred = requests.get(f"{API_BASE}/api/upload/{session}/pred", timeout=30).json()
    if not pred.get("tokens"):
        raise SystemExit("Pred tokens missing")
    if len(pred["tokens"]) < 30:
        raise SystemExit("Too few pred tokens")

    recipe = requests.get(f"{API_BASE}/api/upload/{session}/recipe", timeout=30).json()
    if not recipe.get("ingredients_lines") or not recipe.get("instruction_lines"):
        raise SystemExit("Recipe lines missing")

    # cleanup
    requests.delete(f"{API_BASE}/api/upload/{session}", timeout=10)
    print("Upload smoke passed.")


if __name__ == "__main__":
    main()
