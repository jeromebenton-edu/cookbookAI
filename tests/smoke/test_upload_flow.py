import os
from pathlib import Path

import pytest
import requests

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")


def backend_running():
    try:
        res = requests.get(f"{API_BASE}/api/parse/health", timeout=5)
        return res.ok
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not backend_running(), reason="Backend not running")


def test_upload_flow():
    sample = Path("docs/samples/sample_clean_printed.png")
    if not sample.exists():
        pytest.skip("Sample image missing")
    with sample.open("rb") as f:
        res = requests.post(f"{API_BASE}/api/upload/page", files={"file": f}, timeout=60)
    assert res.ok, res.text
    session = res.json()["session_id"]
    ocr = requests.get(f"{API_BASE}/api/upload/{session}/ocr", timeout=30).json()
    assert len(ocr.get("words", [])) > 10
    pred = requests.get(f"{API_BASE}/api/upload/{session}/pred", timeout=30).json()
    assert len(pred.get("tokens", [])) > 10
    recipe = requests.get(f"{API_BASE}/api/upload/{session}/recipe", timeout=30).json()
    assert recipe.get("ingredients_lines")
    requests.delete(f"{API_BASE}/api/upload/{session}", timeout=10)
