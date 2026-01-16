from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

import json
from datetime import datetime

from pathlib import Path
from typing import Optional, Dict, List
import random
from collections import Counter

from app.services import get_service
from app.utils.recipe_confidence import RECIPE_CONF_VERSION
from app.utils.recipe_extraction import recipe_from_prediction, cache_recipe, load_cached_recipe
from app.utils.prereqs import check_tesseract_available
router = APIRouter(prefix="/api/parse", tags=["parse"])


@router.get("/health")
def health():
    svc = get_service()
    ok, msg = check_tesseract_available()
    h = svc.health()
    h["ocr_available"] = ok
    h["ocr_message"] = msg
    return h


@router.get("/debug/model_labels")
def debug_model_labels():
    svc = get_service()
    svc.load()
    if not svc._model:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    config = svc._model.config
    id2label = {int(k): v for k, v in config.id2label.items()}
    label2id = {k: int(v) for k, v in config.label2id.items()}
    return {
        "id2label": id2label,
        "label2id": label2id,
        "num_labels": len(id2label),
        "config_path": getattr(config, "_name_or_path", None),
        "model_dir": str(svc.model_dir),
        "label_source": "config",
    }


@router.get("/boston/pages")
def list_pages():
    svc = get_service()
    pages = svc.list_pages()
    return {"pages": pages, "count": len(pages)}


@router.get("/boston/available")
def available_pages():
    svc = get_service()
    pages = svc.list_pages()
    return {
        "num_pages": len(pages),
        "available_page_ids": pages,
        "available_png_ids": [f"{p:04d}" for p in pages],
    }


@router.get("/boston/featured")
def featured_pages(limit: int = Query(10, ge=1, le=100), refresh: bool = Query(False)):
    svc = get_service()
    if not (svc._model and svc._dataset):  # degraded
        raise HTTPException(status_code=503, detail="Model/dataset not loaded; see /api/parse/health")
    cache_path = Path("backend/cache/featured_pages.json")
    if cache_path.exists() and not refresh:
        try:
            data = json.loads(cache_path.read_text())
            if all(("recipe_confidence" in (p or {})) and ((p or {}).get("recipe_conf_version") == RECIPE_CONF_VERSION) for p in data.get("pages", [])):
                return data
        except Exception:
            pass
    scored = svc.recipe_index(refresh=refresh, force_rescore=True)
    recipe_pages = [p for p in scored if p.get("is_recipe_page")]
    ranked = sorted(recipe_pages, key=lambda x: x.get("recipe_confidence", 0.0), reverse=True)[:limit]
    if not ranked:  # fallback to highest confidence even if not recipe
        ranked = sorted(scored, key=lambda x: x.get("recipe_confidence", 0.0), reverse=True)[:limit]
    result = {"count": len(ranked), "pages": ranked}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, indent=2))
    return result


def _load_curated_featured(dataset: str, page_index: list[dict], valid_ids: list[int]) -> list[dict]:
    """
    Load curated featured pages from data/demo/featured_pages.json and filter to valid recipe pages.
    """
    path = Path("data/demo/featured_pages.json")
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    ids = data.get(dataset) or []
    if not isinstance(ids, list):
        return []
    curated: list[dict] = []
    page_lookup = {p.get("page_id"): p for p in page_index}
    for pid in ids:
        if not isinstance(pid, int):
            continue
        if pid not in valid_ids:
            continue
        entry = page_lookup.get(pid)
        if entry and entry.get("recipe_like"):
            curated.append(entry)
    return curated


def _count_labels(tokens: List[dict]) -> Dict[str, int]:
    c: Counter = Counter()
    for t in tokens or []:
        lbl = t.get("pred_label") or t.get("label") or "O"
        c[lbl] += 1
    return dict(c)


def _avg_conf_by_label(tokens: List[dict]) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for t in tokens or []:
        lbl = t.get("pred_label") or t.get("label") or "O"
        sums[lbl] = sums.get(lbl, 0.0) + float(t.get("confidence", t.get("score", 0.0)))
        counts[lbl] = counts.get(lbl, 0) + 1
    return {lbl: (sums[lbl] / counts[lbl]) for lbl in sums}


@router.get("/boston/demo")
def demo_bundle():
    svc = get_service()
    try:
        index = svc.recipe_index(force_rescore=True)
    except Exception:
        index = []

    recipe_candidates = [p for p in index if p.get("recipe_like")]
    pages_available = svc.list_pages()
    pages_with_images = [
        pid for pid in pages_available if (svc.pages_dir / f"{int(pid):04d}.png").exists()
    ]

    curated = _load_curated_featured("boston", recipe_candidates, pages_available)
    featured_mode = "curated_recipe"
    featured_sorted = curated if curated else sorted(
        recipe_candidates, key=lambda p: p.get("recipe_confidence", 0.0), reverse=True
    )
    if not featured_sorted and recipe_candidates:
        featured_mode = "heuristic_recipe"
        featured_sorted = sorted(
            recipe_candidates, key=lambda p: p.get("recipe_confidence", 0.0), reverse=True
        )

    status = "ok"
    message = None

    if not featured_sorted and pages_available:
        # fallback to first available pages even if not recipes
        featured_mode = "fallback_any"
        status = "degraded"
        message = "No recipe pages found in curated demo set. Falling back to first available pages."
        fallback_ids = pages_with_images if pages_with_images else pages_available
        fallback_ids = fallback_ids[:10]
        featured_sorted = [
            {
                "page_id": pid,
                "png_id": f"{int(pid):04d}",
                "page_num": pid,
                "is_recipe_page": False,
                "recipe_confidence": 0.0,
            }
            for pid in fallback_ids
        ]
    elif not featured_sorted and not pages_available:
        featured_mode = "fallback_any"
        status = "degraded"
        message = "No pages available. Run: make rebuild-data"

    default_page: Optional[int] = None
    if featured_sorted:
        default_page = featured_sorted[0].get("page_id")
    if default_page is None:
        default_page = svc.find_first_recipe_page(start_page=1, jump=20)
    if default_page is None and recipe_candidates:
        default_page = recipe_candidates[0].get("page_id")
    if default_page is None and index:
        ranked = sorted(index, key=lambda p: p.get("recipe_confidence", 0.0), reverse=True)
        default_page = ranked[0].get("page_id") if ranked else None
    if default_page is None and pages_available:
        default_page = pages_available[0]

    featured_payload = {
        "count": len(featured_sorted[:10]),
        "pages": featured_sorted[:10],
        "source": featured_mode,
    }

    return {
        "featured": featured_payload,
        "featured_pages": featured_payload.get("pages", []),
        "health": svc.health(),
        "default_page": default_page,
        "default_page_id": default_page,
        "status": status,
        "message": message,
        "pages_total": len(pages_available),
        "pages_with_images": len(pages_with_images),
        "featured_mode": featured_mode,
    }

def _overlay_not_available(page_num: int):
    svc = get_service()
    total = len(svc.list_pages())
    return HTTPException(
        status_code=404,
        detail={
            "error": "overlay_not_available",
            "message": "AI overlay not available for this page in the demo subset.",
            "requested_page": page_num,
            "num_pages": total,
            "hint": "Try one of the featured demo pages from /api/parse/boston/demo.",
        },
    )


@router.get("/boston/{page_num}")
def predict_page(
    page_num: int,
    refresh: bool = Query(False, description="Bypass cache if true"),
    grouped: bool = Query(True, description="Include grouped tokens by label"),
    min_conf: float = Query(0.0, ge=0.0, le=1.0, description="Filter out tokens below this confidence"),
):
    svc = get_service()
    try:
        result = svc.predict_page(page_num, grouped=grouped, min_conf=min_conf, refresh=refresh)
    except KeyError:
        raise _overlay_not_available(page_num)
    except Exception as exc:  # pragma: no cover - inference best effort
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    return result


@router.get("/debug/page_stats/{page_num}")
def debug_page_stats(page_num: int):
    svc = get_service()
    svc.load()
    record = svc._get_record(page_num)  # type: ignore[attr-defined]
    if record is None:
        raise _overlay_not_available(page_num)
    # raw inference without filtering
    try:
        raw_tokens, meta = svc._run_inference(
            record["words"],
            record["bboxes"],
            record["image_path"],
            width=record.get("width"),
            height=record.get("height"),
            min_conf=0.0,
            source="debug_raw",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    label_counts_raw = _count_labels(raw_tokens)
    avg_conf_raw = _avg_conf_by_label(raw_tokens)

    # post-processing (standard overlay)
    overlay = svc.predict_page(page_num, grouped=True, min_conf=0.0, refresh=True)
    label_counts_post = _count_labels(overlay.get("tokens", []))
    avg_conf_post = _avg_conf_by_label(overlay.get("tokens", []))

    # top tokens per label (for key labels)
    key_labels = ["TITLE", "INGREDIENT_LINE", "INSTRUCTION_STEP"]
    top_tokens = {}
    for lbl in key_labels:
        toks = [t for t in raw_tokens if (t.get("pred_label") or t.get("label")) == lbl]
        toks_sorted = sorted(toks, key=lambda t: float(t.get("confidence", 0.0)), reverse=True)[:20]
        top_tokens[lbl] = [{"text": t.get("text", ""), "confidence": t.get("confidence", 0.0)} for t in toks_sorted]

    extracted = recipe_from_prediction(overlay, include_raw=False, include_lines=True)

    return {
        "page_num": page_num,
        "token_count_total": len(raw_tokens),
        "label_counts_raw": label_counts_raw,
        "label_counts_post": label_counts_post,
        "avg_confidence_raw": avg_conf_raw,
        "avg_confidence_post": avg_conf_post,
        "top_20_tokens_per_label": top_tokens,
        "extracted_recipe": extracted,
        "meta": overlay.get("meta", {}),
        "label_map": overlay.get("label_map", {}),
    }


@router.get("/boston/{page_num}/image")
def page_image(page_num: int):
    svc = get_service()
    record = svc._get_record(page_num)  # type: ignore[attr-defined]
    if record is None:
        raise _overlay_not_available(page_num)
    img_path = record["image_path"]
    path = Path(img_path)
    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Image file not found at {path}. Check pages_dir configuration.",
        )
    return FileResponse(path)


@router.get("/boston/{page_num}/recipe")
def page_recipe(
    page_num: int,
    refresh: bool = Query(False),
    include_raw: bool = Query(False),
    include_lines: bool = Query(True),
):
    svc = get_service()
    cache_path = Path("backend/cache/recipes") / f"page_{page_num:04d}_recipe.json"
    try:
        if not refresh:
            existing = load_cached_recipe(cache_path)
            if existing and ("is_recipe_page" in existing or "recipe_confidence" in existing):
                return existing
    except Exception:
        pass
    try:
        pred = svc.predict_page(page_num, grouped=True, min_conf=0.0, refresh=refresh)
    except KeyError:
        raise _overlay_not_available(page_num)
    is_recipe = bool(pred.get("is_recipe_page"))
    if not is_recipe:
        recipe = {
            "page_num": pred.get("page_num", page_num),
            "title": "",
            "title_obj": None,
            "ingredients_lines": [] if include_lines else None,
            "instruction_lines": [] if include_lines else None,
            "ingredients": [],
            "instructions": [],
            "confidence": {"title": 0.0, "ingredients": 0.0, "instructions": 0.0, "overall": 0.0},
            "is_recipe_page": False,
            "recipe_confidence": pred.get("recipe_confidence", 0.0),
            "message": "No recipe detected on this page (front matter).",
            "meta": {
                "source": "layoutlmv3_predictions",
                "notes": ["No recipe fields detected"],
                "label_counts": pred.get("label_counts", {}),
            },
        }
        if include_raw:
            recipe["raw"] = {"grouped_tokens": pred.get("grouped")}
    else:
        recipe = recipe_from_prediction(pred, include_raw=include_raw, include_lines=include_lines)
        recipe["is_recipe_page"] = True
        recipe["recipe_confidence"] = pred.get("recipe_confidence", 0.0)
        recipe["meta"]["label_counts"] = pred.get("label_counts", {})
    recipe["meta"]["model"] = svc.model_dir if hasattr(svc, "model_dir") else ""
    recipe["meta"]["dataset"] = str(svc.dataset_dir)
    recipe["meta"]["created_at"] = datetime.utcnow().isoformat() + "Z"
    try:
        cache_recipe(cache_path, recipe)
    except Exception:
        pass
    return recipe


@router.get("/debug/sample_stats")
def debug_sample_stats(n: int = Query(10, ge=1, le=50)):
    svc = get_service()
    svc.load()
    pages = svc.list_pages()
    if not pages:
        return {"pages_tested": 0, "errors": ["No pages available"], "label_counts_raw": {}, "label_counts_post": {}}

    stats = []
    errors = []
    sampled = random.sample(pages, min(n, len(pages)))
    recipe_index = svc.recipe_index(force_rescore=True)
    recipe_candidates = [p.get("page_id") for p in recipe_index if p.get("recipe_like")]
    sampled_recipes = recipe_candidates[: n] if recipe_candidates else []
    candidate_pages = list(dict.fromkeys(sampled_recipes + sampled))

    agg_raw: Counter = Counter()
    agg_post: Counter = Counter()

    for pid in candidate_pages:
        try:
            record = svc._get_record(pid)  # type: ignore[attr-defined]
            if record is None:
                raise KeyError("page missing from dataset")
            raw_tokens, _ = svc._run_inference(
                record["words"],
                record["bboxes"],
                record["image_path"],
                width=record.get("width"),
                height=record.get("height"),
                min_conf=0.0,
                source="debug_sample",
            )
            overlay = svc.predict_page(pid, grouped=False, min_conf=0.0, refresh=True)
            lc_raw = Counter(_count_labels(raw_tokens))
            lc_post = Counter(_count_labels(overlay.get("tokens", [])))
            agg_raw.update(lc_raw)
            agg_post.update(lc_post)
            stats.append(
                {
                    "page_id": pid,
                    "label_counts_raw": dict(lc_raw),
                    "label_counts_post": dict(lc_post),
                }
            )
        except Exception as exc:
            errors.append(f"page {pid}: {exc}")

    best_by_ing = sorted(stats, key=lambda s: s["label_counts_post"].get("INGREDIENT_LINE", 0), reverse=True)[:5]
    best_by_instr = sorted(stats, key=lambda s: s["label_counts_post"].get("INSTRUCTION_STEP", 0), reverse=True)[:5]
    best_by_title = sorted(stats, key=lambda s: s["label_counts_post"].get("TITLE", 0), reverse=True)[:5]

    return {
        "pages_tested": len(stats),
        "label_counts_raw": dict(agg_raw),
        "label_counts_post": dict(agg_post),
        "best_ingredient_pages": best_by_ing,
        "best_instruction_pages": best_by_instr,
        "best_title_pages": best_by_title,
        "errors": errors,
    }
