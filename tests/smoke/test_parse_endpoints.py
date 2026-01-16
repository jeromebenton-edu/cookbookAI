import os
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


def test_health_and_pages():
    res = requests.get(f"{API_BASE}/api/parse/health", timeout=10).json()
    assert "model_dir" in res
    pages = requests.get(f"{API_BASE}/api/parse/boston/pages", timeout=10).json()
    assert pages.get("pages")


def test_recipe_schema():
    pages = requests.get(f"{API_BASE}/api/parse/boston/pages", timeout=10).json()["pages"]
    page = pages[0]
    recipe = requests.get(f"{API_BASE}/api/parse/boston/{page}/recipe", timeout=10).json()
    assert "ingredients_lines" in recipe
    assert "instruction_lines" in recipe
