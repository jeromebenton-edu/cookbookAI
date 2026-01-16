import os
import time
import requests

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")


def wait_for_health(timeout=20):
    start = time.time()
    while time.time() - start < timeout:
        try:
            res = requests.get(f"{API_BASE}/api/parse/health", timeout=5)
            if res.ok:
                return res.json()
        except Exception:
            time.sleep(1)
    raise SystemExit("Backend not responding at /api/parse/health")


def get_json(path: str):
    url = f"{API_BASE}{path}"
    res = requests.get(url, timeout=20)
    if not res.ok:
        raise SystemExit(f"Request failed {res.status_code} for {url}: {res.text}")
    return res.json()
