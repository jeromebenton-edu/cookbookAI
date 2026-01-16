# Deployment Guide (CookbookAI)

Goal: shareable demo where frontend (Vercel) talks to backend (FastAPI) serving LayoutLMv3 predictions.

## Backend (FastAPI)
Options: Fly.io, Railway, Render, Hugging Face Spaces (Docker). CPU works for demo; GPU optional.

Required files available to the container:
- Model dir (default: `models/layoutlmv3_boston_final` or latest stageB full)
- Dataset dir (default: `data/datasets/boston_layoutlmv3_full/dataset_dict`)
- Cache dir (optional): `backend/cache`

Env vars:
- `MODEL_DIR` (path inside container)
- `DATASET_DIR` (path inside container)
- `FRONTEND_ORIGIN` (e.g., `https://your-vercel-app.vercel.app` or `http://localhost:3000`)
- `WARMUP=true` (run one inference on startup)
- `CORS_ALLOW_ALL=true` (for dev only)

Run:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Precompute demo caches (optional, speeds up first load):
```bash
python scripts/precompute_demo_cache.py --limit 10
```
Persist `backend/cache` via a volume if possible.

Health check:
```
GET /api/parse/health
```

## Frontend (Vercel)
Set env var in Vercel project:
- `NEXT_PUBLIC_API_BASE_URL=https://<your-backend-host>`

Build command: `npm run build` (or `npm run lint && npm run build`)
Framework preset: Next.js (app router).

Local dev:
```bash
cd frontend
npm install
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 npm run dev
```

## Demo UX checklist
- /demo page loads featured pages from `/api/parse/boston/featured`
- Overlay toggles and tooltips show predictions
- Extracted recipe panel uses `/api/parse/boston/{page}/recipe`
- Raw JSON modal works

## Cold-start mitigation
- Enable WARMUP
- Precompute caches (`scripts/precompute_demo_cache.py`)
- Keep container warm (provider-specific keepalive)
