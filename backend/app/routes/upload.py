from __future__ import annotations

import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
import json

from app.models.upload_session import UploadSession
from app.services import get_service
from app.utils.ocr_tesseract import image_to_ocr_tokens
from app.utils.recipe_extraction import recipe_from_prediction, cache_recipe
from app.utils.prereqs import check_tesseract_available

router = APIRouter(prefix="/api/upload", tags=["upload"])

UPLOAD_ROOT = Path("backend/cache/uploads")
MAX_SIZE_BYTES = 10 * 1024 * 1024


def _save_upload(file: UploadFile, dest: Path) -> Path:
    raw_dir = dest / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(file.filename or "upload").name
    raw_path = raw_dir / filename
    content = file.file.read()
    if len(content) > MAX_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large (10MB max)")
    raw_path.write_bytes(content)
    return raw_path


def _ensure_image(raw_path: Path, dest_dir: Path) -> Path:
    if raw_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise HTTPException(status_code=415, detail="Only PNG/JPG supported for now.")
    image = Image.open(raw_path).convert("RGB")
    out_path = dest_dir / "page.png"
    image.save(out_path)
    return out_path


def _session_dir(session_id: str) -> Path:
    return UPLOAD_ROOT / session_id


def _load_session(session_id: str) -> UploadSession:
    root = _session_dir(session_id)
    if not (root / "session.json").exists():
        raise HTTPException(status_code=404, detail="Session not found")
    return UploadSession.load(root)


@router.post("/page")
async def upload_page(
    file: UploadFile = File(...),
    run_ocr: bool = Query(True),
    run_infer: bool = Query(True),
    run_recipe: bool = Query(True),
):
    ok, msg = check_tesseract_available()
    if not ok and run_ocr:
        raise HTTPException(status_code=503, detail=f"OCR unavailable: {msg}")
    session_id = uuid.uuid4().hex
    root = _session_dir(session_id)
    root.mkdir(parents=True, exist_ok=True)

    session = UploadSession(
        session_id=session_id,
        filename=file.filename or "upload",
        created_at=datetime.utcnow(),
        status="uploaded",
        image_path="",
        ocr_json_path=None,
        pred_json_path=None,
        recipe_json_path=None,
        error=None,
    )
    try:
        raw_path = _save_upload(file, root)
        image_path = _ensure_image(raw_path, root)
        session.image_path = str(image_path)
        session.status = "uploaded"
        session.save(root)

        ocr_json_path = None
        ocr_data = None
        if run_ocr:
            ocr_data = image_to_ocr_tokens(image_path)
            ocr_json_path = root / "ocr.json"
            ocr_json_path.write_text(json.dumps(ocr_data, indent=2))
            session.ocr_json_path = str(ocr_json_path)
            session.status = "ocr_done"
            session.save(root)
        elif (root / "ocr.json").exists():
            ocr_data = json.loads((root / "ocr.json").read_text())

        pred_json_path = None
        if run_infer:
            if ocr_data is None:
                raise HTTPException(status_code=400, detail="OCR data missing; enable run_ocr.")
            svc = get_service()
            pred = svc.predict_from_ocr(
                ocr_data["words"],
                ocr_data["bboxes"],
                image_path=str(image_path),
                width=ocr_data.get("width"),
                height=ocr_data.get("height"),
                grouped=True,
            )
            pred["image_url"] = f"/api/upload/{session_id}/image"
            pred_json_path = root / "pred.json"
            pred_json_path.write_text(json.dumps(pred, indent=2))
            session.pred_json_path = str(pred_json_path)
            session.status = "pred_done"
            session.save(root)

        if run_recipe and pred_json_path:
            pred = json.loads(pred_json_path.read_text())
            recipe = recipe_from_prediction(pred, include_raw=False, include_lines=True)
            recipe["meta"]["source_session"] = session_id
            recipe_json_path = root / "recipe.json"
            cache_recipe(recipe_json_path, recipe)
            session.recipe_json_path = str(recipe_json_path)
            session.save(root)

        return {
            "session_id": session_id,
            "status": session.status,
            "image_url": f"/api/upload/{session_id}/image",
            "ocr_url": f"/api/upload/{session_id}/ocr" if session.ocr_json_path else None,
            "pred_url": f"/api/upload/{session_id}/pred" if session.pred_json_path else None,
            "recipe_url": f"/api/upload/{session_id}/recipe" if session.recipe_json_path else None,
        }
    except HTTPException:
        session.status = "failed"
        session.error = "upload_failed"
        session.save(root)
        raise
    except Exception as exc:
        session.status = "failed"
        session.error = str(exc)
        session.save(root)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc


@router.get("/{session_id}")
def get_session(session_id: str):
    return _load_session(session_id)


@router.get("/{session_id}/image")
def get_image(session_id: str):
    session = _load_session(session_id)
    return FileResponse(session.image_path)


@router.get("/{session_id}/ocr")
def get_ocr(session_id: str):
    session = _load_session(session_id)
    if not session.ocr_json_path:
        raise HTTPException(status_code=404, detail="OCR not available")
    return json.loads(Path(session.ocr_json_path).read_text())


@router.get("/{session_id}/pred")
def get_pred(session_id: str):
    session = _load_session(session_id)
    if not session.pred_json_path:
        raise HTTPException(status_code=404, detail="Prediction not available")
    return json.loads(Path(session.pred_json_path).read_text())


@router.get("/{session_id}/recipe")
def get_recipe(session_id: str):
    session = _load_session(session_id)
    if not session.recipe_json_path:
        raise HTTPException(status_code=404, detail="Recipe not available")
    return json.loads(Path(session.recipe_json_path).read_text())


@router.delete("/{session_id}")
def delete_session(session_id: str):
    root = _session_dir(session_id)
    if not root.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    shutil.rmtree(root)
    return {"status": "deleted", "session_id": session_id}
