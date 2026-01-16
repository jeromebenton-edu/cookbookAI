# CookbookAI Backend (Inference)

FastAPI service that serves LayoutLMv3 token predictions for scanned cookbook pages and exposes them to the frontend AI Parse View.

## Quickstart

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# run API
uvicorn app.main:app --reload --port 8000
```

### Key env vars

- `MODEL_DIR` (default tries `models/layoutlmv3_boston_stageB_full_latest/layoutlmv3_boston_final` then `models/layoutlmv3_boston_final`)
- `DATASET_DIR` (default tries `_generated` then `data/datasets/boston_layoutlmv3_full/dataset_dict`)
- `FRONTEND_ORIGIN` (e.g. `http://localhost:3000` or your Vercel URL)
- `WARMUP=true|false` (load model + run one page on startup; default true)
- `CORS_ALLOW_ALL=true` to allow all origins in dev
- OCR uploads: install Tesseract (`sudo apt-get install tesseract-ocr` or `brew install tesseract`); Python dep `pytesseract` is in requirements.

### Parse endpoints

- `GET /api/parse/health` – model/dataset status
- `GET /api/parse/boston/pages` – list page numbers
- `GET /api/parse/boston/{page_num}?refresh=&grouped=&min_conf=` – overlay JSON (tokens + grouped + label_map)
- `GET /api/parse/boston/{page_num}/image` – serve the scanned page image

### Example

```bash
curl http://localhost:8000/api/parse/health
curl http://localhost:8000/api/parse/boston/pages | head
curl http://localhost:8000/api/parse/boston/16 | jq '.tokens[0]'
```

Responses include:
```json
{
  "page_num": 16,
  "image_path": "data/pages/boston/0016.png",
  "tokens": [{"text":"Scald","bbox":[...],"pred_label":"INSTRUCTION_STEP","confidence":0.92}],
  "grouped": {...},
  "label_map": {"id2label": {...}, "label2id": {...}},
  "meta": {"device":"cpu","model_dir":"...","dataset_dir":"..."}
}
```

### Upload endpoints
- `POST /api/upload/page` (PNG/JPG) runs OCR + inference + recipe extraction, returns session URLs (image/ocr/pred/recipe).
- `GET /api/upload/{session}` and `/image` `/ocr` `/pred` `/recipe`.
- `DELETE /api/upload/{session}` to clean up.
