Troubleshooting
===============

- OCR/Tesseract missing:
  - Error: `OCR unavailable: tesseract binary not found`
  - Install Tesseract (`sudo apt-get install tesseract-ocr` on Ubuntu/WSL2, `brew install tesseract` on macOS)
  - Verify: `tesseract --version`

- API base URL mismatch:
  - Ensure `NEXT_PUBLIC_API_BASE_URL` points to your backend (default http://localhost:8000)
  - Check `/api/parse/health` in browser/curl

- Model path not found:
  - Health shows wrong `model_dir`; set `MODEL_DIR` env var or place model under `models/layoutlmv3_boston_final`

- Slow CPU inference:
  - Use smaller images or run fewer epochs; GPU optional but not required

- Upload parsing disabled:
  - Health shows `ocr_available=false`; install Tesseract
  - Backend 503 on upload: check Tesseract installation

- WSL2 file permissions:
  - Keep repo inside Linux FS (`/home/...`)
  - Ensure `backend/cache/` is writable

- Regenerate caches:
  - Featured/recipe caches: delete `backend/cache/*` or run `scripts/precompute_demo_cache.py`

- Logs:
  - Backend tmux session `cookbookai-backend`
  - Frontend tmux session `cookbookai-frontend`

If blocked, rerun `make smoke` to see minimal reproducible errors.
