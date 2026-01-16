from __future__ import annotations

import os
from typing import List
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import parse_router
from app.routes.upload import router as upload_router
from app.services import get_service
from app.settings import get_settings

settings = get_settings()
app = FastAPI(title="CookbookAI Inference", version="0.3.0")


def _allowed_origins() -> List[str]:
    origins = settings.frontend_origins[:]
    if os.getenv("CORS_ALLOW_ALL", "false").lower() == "true":
        origins = ["*"]
    return origins


cors_origins = _allowed_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    warm = settings.warmup or settings.prewarm or os.getenv("COOKBOOKAI_PREWARM") == "1"
    svc = get_service()
    svc.load()
    if warm:
        def _prewarm():
            pages = svc.list_pages()
            if pages:
                try:
                    svc.predict_page(pages[0], grouped=True, min_conf=0.0, refresh=True)
                    print(f"Prewarm inference succeeded on page {pages[0]}")
                except Exception as exc:
                    print(f"Prewarm inference failed: {exc}")
        threading.Thread(target=_prewarm, daemon=True).start()
    print(f"Started with CORS origins: {cors_origins}")
    print(f"Model dir: {svc.model_dir}")
    print(f"Dataset dir: {svc.dataset_dir}")
    print(f"OCR check: {get_service().health().get('ocr_message')}")


# Routers
app.include_router(parse_router)
app.include_router(upload_router)


@app.get("/health")
def root_health():
    return {"status": "ok", "parse": get_service().health()}
