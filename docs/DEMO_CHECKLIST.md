Demo Checklist
==============

Prereqs
- Install Python deps (`pip install -r backend/requirements.txt`)
- Install Tesseract (`sudo apt-get install tesseract-ocr` or `brew install tesseract`)
- Generate/verify samples: `python scripts/generate_sample_assets.py --overwrite` and `python scripts/verify_sample_assets.py`

Run demo
- `make demo` (starts backend/frontend, opens http://localhost:3000/demo)
- Try featured pages, toggles, tooltips, raw JSON
- Click “Upload your own” to test Phase 6 upload flow

Smoke tests
- `make smoke` (runs sample verification + backend/upload smoke scripts)
- Expected: no errors, upload smoke passes, cache files under data/reports/smoke/ (if emitted)

Artifacts/logs
- Backend logs (tmux session `cookbookai-backend`)
- Frontend logs (tmux session `cookbookai-frontend`)

If anything fails, see docs/TROUBLESHOOTING.md
