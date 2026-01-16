# CookbookAI

CookbookAI is a teaching portfolio project that turns the 1896 *Boston Cooking-School Cook Book* into a polished cookbook website with an AI Parse View powered by LayoutLMv3. The repo includes a Next.js frontend, a FastAPI inference service, and scaffolding for labeling and training.

## Repository Structure

```
cookbookAI/
|- frontend/      # Next.js app (deploy to Vercel)
|- backend/       # FastAPI inference service
|- training/      # labeling + training scaffolding
`- README.md
```

## Quick Start

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
MOCK_INFERENCE=true uvicorn app:app --reload
```

Set `NEXT_PUBLIC_API_BASE_URL` in the frontend to connect the backend.

### Data / Labeling Prereqs

The OCR + weak-label pipelines require the system Tesseract binary in addition to the Python deps:

```
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS (Homebrew)
brew install tesseract
```

Then install Python deps: `pip install -r data/requirements.txt`.

### Label Studio (optional gold labeling)

To avoid dependency clashes with the data pipeline (which pins numpy 1.x), use a separate virtualenv for Label Studio:

```
# one-liner helper
./scripts/label_studio_venv.sh start

# or manually:
python -m venv ~/.venvs/labelstudio
source ~/.venvs/labelstudio/bin/activate
pip install label-studio
label-studio start
```

Keep your main `.venv` for OCR/weak labeling. If you install Label Studio into the main env by mistake, re-pin with:

```
pip install --force-reinstall "numpy==1.26.4" "thinc==8.2.4" "spacy==3.7.5"
```

## Recipe Pipeline (recipes -> LayoutLMv3 datasets)
Run the expanded end-to-end pipeline (render → OCR → detect recipes → weak labels → HF datasets → sanity overlays):
```
python scripts/run_option2_pipeline.py --config configs/option2_pipeline.yaml
```
Defaults target pages 120–420 with relaxed thresholds (candidate 0.30, min_avg_line_conf 0.65) and build both full and highconf LayoutLMv3 DatasetDicts under `data/_generated/`.

Phase 3.75: high-confidence is derived by structural filtering (titles + ingredient/instruction counts, O ratio, token conf), not just confidence gates. Recipe detection ranks; you can label all pages with `--use_all_pages`.

## Make-based dev workflow

- `make dev` (alias: `make run-all`) installs backend/frontend deps (auto-detects poetry/pip and pnpm/npm), builds both apps, starts uvicorn (binds to `BACKEND_BIND_HOST/BACKEND_PORT`) and Next.js (binds to `FRONTEND_BIND_HOST/FRONTEND_PORT`), waits for health, and blocks with a clean Ctrl-C trap. PID files and logs live in `.pids/`.
- `make status` shows PID presence and listening ports (uses `lsof`/`ss` when available).
- `make health` polls both services for up to 60 seconds and reports ✅/❌ with log tails on failure.
- `make stop` stops both servers using the PID files written by `make dev` and also kills any stray listeners on the expected ports (handles uvicorn reloaders/Next.js dev instances).

Defaults: Backend `http://localhost:8000`, Frontend `http://localhost:3001`. Override with environment variables:
`BACKEND_HOST`, `BACKEND_PORT`, `FRONTEND_HOST`, `FRONTEND_PORT` (binding hosts default to `BACKEND_BIND_HOST`/`FRONTEND_BIND_HOST`, which default to `0.0.0.0` for uvicorn/Next.js).

## Rebuilding the full dataset from PDF
- `make render-pages` — render the full Boston cookbook PDF to `data/pages/boston/*.png` (set `COOKBOOKAI_MAX_PAGES` to limit during dev).
- `make build-dataset` — OCR rendered pages and build a HuggingFace dataset at `data/datasets/boston_layoutlmv3_full/dataset_dict` (unlabeled pages get `O` labels).
- `make rebuild-data` — run both steps in order. Use this when the backend health shows an incomplete page count.

## Phase 4 training (LayoutLMv3)
- One-command runner: `python scripts/run_phase4_experiments.py --config configs/phase4_experiment.yaml`
- Auto highconf regeneration: if the highconf dataset is empty, the runner calls `scripts/regenerate_highconf.py` (or run `make regen-highconf`) to relax structural thresholds until at least the target number of pages pass.
- Auto inference page selection: `--infer_page_num auto` picks the first page from the validation split (fallback to train) to avoid missing-page errors.
- Weak-label eval writes JSON/MD/CSV under `data/reports/phase4_runs/<run_id>/evaluation/` without JSON serialization crashes.

## Phase 5.75 (Compare Mode & Corrections)
- Backend recipe endpoint returns per-line confidences (`title_obj`, `ingredients_lines`, `instruction_lines`) with stable IDs and bboxes.
- Frontend Compare Mode highlights differences (Curated vs AI) with confidence chips and inline edits; export corrected JSON for future training.
- Demo page (`/demo`) defaults to Compare, pulling featured pages from the API.
- See `docs/PHASE_5_75_COMPARE_MODE.md` for diff logic and export format.

## Demo Quickstart (Phase 6.25)
- Install Tesseract (`sudo apt-get install tesseract-ocr` or `brew install tesseract`)
- Activate the venv and install backend deps:
  ```bash
  source .venv/bin/activate
  pip install -r backend/requirements.txt
  ```
- Install frontend deps:
  ```bash
  cd frontend
  npm install
  cd ..
  ```
- `make demo` to start backend+frontend and open `/demo`
- `make smoke` to run minimal health/upload smoke tests
- `make smoke-demo` to run an end-to-end API smoke (backend must be running)
- Samples live under `docs/samples/`; generate/verify with scripts in `scripts/`

## Deployment

- The frontend is ready for Vercel. Import the `frontend/` directory and set env vars.
- The backend can be deployed separately (Render, Fly, etc.).

## TODO

- Add more recipe scans and labels.
- Fine-tune LayoutLMv3 weights and update inference.
- Improve OCR and reading order logic.
