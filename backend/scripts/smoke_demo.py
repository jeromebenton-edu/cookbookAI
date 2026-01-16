"""Lightweight end-to-end smoke test for the CookbookAI demo.

Run with:
    python backend/scripts/smoke_demo.py

Or via Makefile target `make smoke-demo` (backend must be running).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import requests


BASE_URL = os.getenv("COOKBOOKAI_API_BASE_URL", "http://localhost:8000").rstrip("/")


def _fail(msg: str, resp: requests.Response | None = None) -> None:
    print(f"❌ {msg}")
    if resp is not None:
        body = resp.text
        if len(body) > 500:
            body = body[:500] + "..."
        print(f"Status: {resp.status_code}\nBody: {body}")
    sys.exit(1)


def _get(path: str, timeout: float) -> requests.Response:
    url = f"{BASE_URL}{path}"
    try:
        resp = requests.get(url, timeout=timeout)
    except Exception as exc:  # pragma: no cover - network error
        _fail(f"Request to {url} failed: {exc}")
    return resp


def _validate_inference(payload: Dict[str, Any]) -> bool:
    # Flexible: accept either tokens list at top-level or under overlay
    candidates = []
    if isinstance(payload, dict):
        if isinstance(payload.get("tokens"), list):
            candidates.append(payload.get("tokens"))
        if isinstance(payload.get("overlay"), dict) and isinstance(payload["overlay"].get("tokens"), list):
            candidates.append(payload["overlay"].get("tokens"))
    for toks in candidates:
        if not toks:
            continue
        sample = toks[0]
        if isinstance(sample, dict) and {"text", "bbox"}.issubset(sample.keys()):
            return True
    return False


def main() -> None:
    print(f"Base URL: {BASE_URL}")

    # Health
    resp = _get("/api/parse/health", timeout=10)
    if resp.status_code != 200:
        _fail("Health endpoint returned non-200", resp)
    try:
        health = resp.json()
    except Exception:
        _fail("Failed to parse health JSON", resp)
    if not (health.get("status") == "ok" and health.get("model_loaded") and health.get("dataset_loaded") and health.get("pages_available")):
        _fail("Health checks failed (status/model/dataset/pages)", resp)
    print("✅ health ok")

    # Demo bundle
    resp = _get("/api/parse/boston/demo", timeout=10)
    if resp.status_code != 200:
        _fail("Demo endpoint returned non-200", resp)
    try:
        demo = resp.json()
    except Exception:
        _fail("Failed to parse demo JSON", resp)
    if not isinstance(demo, dict) or not demo.get("featured"):
        _fail("Demo endpoint missing featured bundle", resp)
    pages = demo.get("featured", {}).get("pages") or []
    default_page = demo.get("default_page") or (pages[0]["page_num"] if pages else 4)
    print("✅ demo endpoint ok")

    # Inference
    resp = _get(f"/api/parse/boston/{int(default_page)}", timeout=30)
    if resp.status_code != 200:
        _fail(f"Inference failed for page {default_page}", resp)
    try:
        inf = resp.json()
    except Exception:
        _fail("Failed to parse inference JSON", resp)
    if not _validate_inference(inf):
        _fail(f"Inference response missing expected token fields. Keys: {list(inf.keys())}")
    print(f"✅ inference ok for page {default_page}")

    print("🎉 Smoke demo PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()

