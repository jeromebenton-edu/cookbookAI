from __future__ import annotations

import json
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import glob
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_from_disk
from PIL import Image
from torch.nn import functional as F
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

from app.core.paths import get_pages_dir, get_project_root
from app.utils.recipe_confidence import RECIPE_CONF_VERSION, is_recipe_like, score_prediction

DEFAULT_MODEL_CANDIDATES = [
    os.getenv("MODEL_DIR"),
    "models/layoutlmv3_v3_manual_59pages_balanced",
    "models/layoutlmv3_boston_stageB_full_latest/layoutlmv3_boston_final",
    "models/layoutlmv3_boston_stageB_full_*/layoutlmv3_boston_final",
    "models/layoutlmv3_boston_final",
]

DEFAULT_DATASET_CANDIDATES = [
    os.getenv("DATASET_DIR"),
    "data/datasets/boston_layoutlmv3_v3_manual_expanded2/dataset_dict",
    "data/datasets/boston_layoutlmv3_v3_manual_expanded2",
    "data/_generated/datasets/boston_layoutlmv3_full/dataset_dict",
    "data/_generated/datasets/boston_layoutlmv3_full",
    "data/datasets/boston_layoutlmv3_full/dataset_dict",
    "data/datasets/boston_layoutlmv3_full",
]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = Path("backend/cache/predictions")


def _first_existing(candidates: List[Optional[str]]) -> Path:
    for cand in candidates:
        if not cand:
            continue
        cand_path = Path(cand)
        if not cand_path.is_absolute():
            cand_path = PROJECT_ROOT / cand_path
        if "*" in cand_path.as_posix():
            matches = sorted(glob.glob(str(cand_path)), reverse=True)
            if matches:
                return Path(matches[0])
        else:
            if cand_path.exists():
                return cand_path
    raise FileNotFoundError(f"No valid path found from candidates: {candidates}")


class LayoutLMService:
    def __init__(self) -> None:
        self.model_dir = None
        self.dataset_dir = None
        self._model: Optional[LayoutLMv3ForTokenClassification] = None
        self._processor: Optional[LayoutLMv3Processor] = None
        self._dataset = None
        self._page_index: Dict[int, Tuple[str, int]] = {}
        self._recipe_page_meta: Dict[int, dict] = {}
        self._recipe_page_index: List[dict] = []
        self._first_recipe_page_id: Optional[int] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.project_root = get_project_root()
        self.pages_dir: Path = get_pages_dir()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.errors: Dict[str, str] = {}
        # resolve paths but don't crash
        try:
            self.model_dir = _first_existing(DEFAULT_MODEL_CANDIDATES)
        except Exception as exc:
            self.errors["model_dir"] = str(exc)
        try:
            self.dataset_dir = _first_existing(DEFAULT_DATASET_CANDIDATES)
        except Exception as exc:
            self.errors["dataset_dir"] = str(exc)

    # -- Loading -----------------------------------------------------------------
    def load(self) -> None:
        if self._model is not None and self._processor is not None and self._dataset is not None:
            return
        # load model
        if self.model_dir:
            try:
                self._processor = LayoutLMv3Processor.from_pretrained(self.model_dir, apply_ocr=False)
                self._model = LayoutLMv3ForTokenClassification.from_pretrained(
                    self.model_dir
                ).to(self.device)
                self._model.eval()
            except Exception as exc:
                self.errors["model_load"] = str(exc)
                self._model = None
                self._processor = None
        # load dataset
        if self.dataset_dir and self._processor:
            try:
                self._dataset = self._load_hf_dataset(self.dataset_dir)
                self._page_index.clear()
                for split in self._dataset.keys():
                    for idx, num in enumerate(self._dataset[split]["page_num"]):
                        self._page_index[int(num)] = (split, idx)
                if not self._page_index:
                    self.errors["dataset_pages"] = "Dataset loaded but no page_num entries were found."
            except Exception as exc:
                self.errors["dataset_load"] = str(exc)
                self._dataset = None

    # -- Public API --------------------------------------------------------------
    def health(self) -> dict:
        model_loaded = self._model is not None
        dataset_loaded = self._dataset is not None and bool(self._page_index)
        status = "ok" if model_loaded and dataset_loaded else "degraded"
        dataset_sizes = {}
        has_validation_split = False
        validation_size = 0
        validation_nonempty = False
        validation_key = None

        if isinstance(self._dataset, DatasetDict):
            dataset_sizes = {k: len(v) for k, v in self._dataset.items()}

            # Check for validation split (any common name)
            if "validation" in self._dataset:
                validation_key = "validation"
                validation_size = len(self._dataset["validation"])
            elif "val" in self._dataset:
                validation_key = "val"
                validation_size = len(self._dataset["val"])
            elif "eval" in self._dataset:
                validation_key = "eval"
                validation_size = len(self._dataset["eval"])

            has_validation_split = validation_key is not None
            validation_nonempty = validation_size > 0

            # Degrade status if validation exists but is empty
            if has_validation_split and not validation_nonempty:
                status = "degraded"
                self.errors["validation_empty"] = (
                    f"Validation split '{validation_key}' exists but is EMPTY (0 examples). "
                    f"This will break training metrics. Run: make ensure-validation-split"
                )

        sample_page = self.pages_dir / "0001.png"
        pages_available = sample_page.exists()
        png_files = sorted(self.pages_dir.glob("*.png"))
        png_count = len(png_files)
        dataset_count = len(self._page_index) if self._page_index else 0
        if dataset_count and png_count:
            mismatch = abs(dataset_count - png_count) / max(dataset_count, 1)
            if mismatch > 0.02:
                status = "degraded"
                self.errors["page_mismatch"] = (
                    f"Dataset pages ({dataset_count}) and PNGs ({png_count}) differ by >2%."
                )
        if not pages_available:
            self.errors.setdefault("pages", f"Sample page not found at {sample_page}")
            status = "degraded"

        return {
            "status": status,
            "model_loaded": model_loaded,
            "dataset_loaded": dataset_loaded,
            "model_dir": str(self.model_dir) if self.model_dir else None,
            "dataset_dir": str(self.dataset_dir) if self.dataset_dir else None,
            "project_root": str(self.project_root),
            "device": str(self.device),
            "num_pages": len(self._page_index) if self._page_index else 0,
            "png_count": png_count,
            "errors": self.errors,
            "dataset_type": self._dataset.__class__.__name__ if self._dataset is not None else None,
            "dataset_splits": list(self._dataset.keys()) if isinstance(self._dataset, DatasetDict) else [],
            "dataset_sizes": dataset_sizes,
            "has_validation_split": has_validation_split,
            "validation_size": validation_size,
            "validation_nonempty": validation_nonempty,
            "validation_key": validation_key,
            "pages_dir": str(self.pages_dir),
            "pages_available": pages_available,
            "pages_sample": str(sample_page),
        }

    def list_pages(self) -> List[int]:
        self.load()
        return sorted(self._page_index.keys())

    def predict_page(self, page_num: int, grouped: bool = True, min_conf: float = 0.0, refresh: bool = False) -> dict:
        self.load()
        if refresh:
            self._predict_cached.cache_clear()
            self._recipe_page_meta.pop(page_num, None)
            self._recipe_page_index = []
        # disk cache?
        disk_path = CACHE_DIR / f"page_{page_num:04d}.json"
        result = None
        needs_save = refresh
        if not refresh and disk_path.exists():
            try:
                result = json.loads(disk_path.read_text())
            except Exception:
                result = None
        if result is None:
            result = self._predict_cached(page_num, grouped, float(min_conf))
            needs_save = True
        has_recipe_meta = "recipe_confidence" in result and "is_recipe_page" in result
        result = self._attach_recipe_meta(result, refresh=refresh or not has_recipe_meta)
        if needs_save or refresh or not has_recipe_meta:
            disk_path.write_text(json.dumps(result, indent=2))
        return result

    def recipe_index(self, refresh: bool = False, force_rescore: bool = False) -> List[dict]:
        """
        Build (and cache) a lightweight index of pages scored by recipe-likeness.
        """
        if (self._recipe_page_index and not refresh and not force_rescore):
            return self._recipe_page_index
        if refresh or force_rescore:
            self._recipe_page_meta.clear()
            self._recipe_page_index = []
            self._first_recipe_page_id = None
        pages = self.list_pages()
        results: List[dict] = []
        for num in pages:
            try:
                overlay = self.predict_page(num, grouped=True, min_conf=0.0, refresh=refresh)
                meta = self._recipe_page_meta.get(num) or score_prediction(overlay)
                if meta.get("recipe_conf_version") != RECIPE_CONF_VERSION:
                    meta = score_prediction(overlay)
                entry = {
                    "page_id": int(num),
                    "png_id": f"{int(num):04d}",
                    "page_num": int(overlay.get("page_num", num)),
                    "image_url": overlay.get("image_url"),
                    **meta,
                }
                self._recipe_page_meta[num] = meta
                results.append(entry)
            except Exception:
                continue
        # keep the cache around for repeated use
        self._recipe_page_index = results
        return results

    # -- Internals ---------------------------------------------------------------
    @lru_cache(maxsize=64)
    def _predict_cached(self, page_num: int, grouped: bool, min_conf: float) -> dict:
        record = self._get_record(page_num)
        if record is None:
            raise KeyError(f"page_num {page_num} not found")

        tokens, meta = self._run_inference(
            record["words"],
            record["bboxes"],
            record["image_path"],
            width=record.get("width"),
            height=record.get("height"),
            min_conf=min_conf,
            source="dataset",
        )
        return self._build_overlay(
            page_num=record["page_num"],
            image_path=record["image_path"],
            tokens=tokens,
            grouped=grouped,
            width=record.get("width"),
            height=record.get("height"),
        )

    def _attach_recipe_meta(self, overlay: dict, refresh: bool = False) -> dict:
        """
        Compute and merge recipe scoring metadata into an overlay response.
        """
        page_num = int(overlay.get("page_num") or -1)
        cached = self._recipe_page_meta.get(page_num) if page_num != -1 else None
        needs_rescore = (
            refresh
            or cached is None
            or cached.get("recipe_conf_version") != RECIPE_CONF_VERSION
            or overlay.get("recipe_conf_version") != RECIPE_CONF_VERSION
        )
        meta = score_prediction(overlay) if needs_rescore else cached
        if page_num != -1:
            self._recipe_page_meta[page_num] = meta
        overlay.update(meta)
        return overlay

    def _get_record(self, page_num: int) -> Optional[dict]:
        if page_num not in self._page_index:
            return None
        split, idx = self._page_index[page_num]
        record = dict(self._dataset[split][idx])
        # normalize image path to project root
        record["image_path"] = str(self._resolve_image_path(record.get("image_path", "")))
        return record

    def _resolve_image_path(self, image_path: str) -> Path:
        """
        Make image paths absolute, preferring the configured pages_dir.
        Handles relative paths regardless of current working directory.
        """
        if not image_path:
            return self.pages_dir / "MISSING.png"
        p = Path(image_path)
        # absolute path that exists
        if p.is_absolute() and p.exists():
            return p
        # relative to project root
        pr = self.project_root / p
        if pr.exists():
            return pr
        # relative to pages dir using filename
        candidate = self.pages_dir / p.name
        if candidate.exists():
            return candidate
        # fallback: pages_dir / original relative
        return self.pages_dir / p

    def _get_page_text(self, page_num: int) -> str:
        """
        Return the best available text representation for a page to run heuristics.
        Prefers cached OCR words; falls back to overlay tokens if necessary.
        """
        record = self._get_record(page_num)
        if record:
            words = record.get("words") or []
            if words:
                try:
                    return " ".join(map(str, words))
                except Exception:
                    pass
        try:
            overlay = self.predict_page(page_num, grouped=False, min_conf=0.0, refresh=False)
            return " ".join(str(t.get("text", "")) for t in overlay.get("tokens", []))
        except Exception:
            return ""

    def find_first_recipe_page(self, start_page: int = 1, jump: int = 20, refresh: bool = False) -> Optional[int]:
        """
        Quickly locate the first recipe-like page by scanning in chunks.
        Uses text-based heuristics to avoid running full inference on every page.
        """
        self.load()
        if refresh:
            self._first_recipe_page_id = None
        if self._first_recipe_page_id is not None and self._first_recipe_page_id > 0:
            return self._first_recipe_page_id
        if self._first_recipe_page_id == -1:
            return None

        pages = self.list_pages()
        if not pages:
            return None

        start = max(start_page, min(pages))
        max_page = max(pages)
        current = start
        found: Optional[int] = None

        while current <= max_page and found is None:
            chunk = [p for p in pages if current <= p < current + jump]
            for pid in chunk:
                if is_recipe_like(self._get_page_text(pid)):
                    found = pid
                    break
            current += jump

        self._first_recipe_page_id = found if found is not None else -1
        return found

    def _load_hf_dataset(self, dataset_dir: Path):
        """
        Robustly load a HF dataset saved to disk.

        Handles:
        - DatasetDict saved via save_to_disk
        - Dataset saved via save_to_disk (wrapped into DatasetDict)
        - Bare directory with dataset_dict.json pointing to split folders
        - “List” feature type in dataset_info (older builder bug) -> coerced to Sequence
        """

        def _fix_feature_types(obj):
            if isinstance(obj, dict):
                fixed = {}
                for k, v in obj.items():
                    if k == "_type" and v == "List":
                        fixed[k] = "Sequence"
                    else:
                        fixed[k] = _fix_feature_types(v)
                return fixed
            if isinstance(obj, list):
                return [_fix_feature_types(x) for x in obj]
            return obj

        def _patch_dataset_info(split_path: Path) -> None:
            info_path = split_path / "dataset_info.json"
            if not info_path.exists():
                return
            try:
                data = json.loads(info_path.read_text())
                original = json.dumps(data, sort_keys=True)
                if "features" in data:
                    data["features"] = _fix_feature_types(data["features"])
                updated = json.dumps(data, sort_keys=True)
                if updated != original:
                    info_path.write_text(json.dumps(data, indent=2))
            except Exception:
                # best-effort; ignore
                return

        def _load_split(split_path: Path) -> Dataset:
            _patch_dataset_info(split_path)
            try:
                return load_from_disk(str(split_path))
            except Exception as exc:
                # Known issue: older dataset_info saved with "List" leads to dataclass error.
                # Rebuild from Arrow directly as a fallback.
                if "dataclass" not in str(exc):
                    raise
                try:
                    import pyarrow as pa
                except Exception as arrow_exc:  # pragma: no cover - defensive
                    raise RuntimeError(
                        f"pyarrow not available to rebuild split {split_path}: {arrow_exc}"
                    ) from exc

                state_path = split_path / "state.json"
                if not state_path.exists():
                    raise
                state = json.loads(state_path.read_text())
                data_files = [split_path / df["filename"] for df in state.get("_data_files", [])]
                tables = []
                for fpath in data_files:
                    with pa.ipc.open_stream(fpath) as reader:
                        tables.append(reader.read_all())
                if not tables:
                    raise RuntimeError(f"No data files found for split {split_path}")
                table = pa.concat_tables(tables)
                features = Features(
                    {
                        "id": Value("string"),
                        "page_num": Value("int32"),
                        "image_path": Value("string"),
                        "words": Sequence(Value("string")),
                        "bboxes": Sequence(Sequence(Value("int32"))),
                        "labels": Sequence(Value("int32")),
                        "label_names": Sequence(Value("string")),
                        "width": Value("int32"),
                        "height": Value("int32"),
                    }
                )
                rebuilt = Dataset.from_dict(table.to_pydict(), features=features)
                return rebuilt

        def _load_candidate(path: Path):
            # If there is a dataset_dict.json, assemble manually
            dict_path = path / "dataset_dict.json"
            if dict_path.exists():
                try:
                    meta = json.loads(dict_path.read_text())
                    splits = meta.get("splits") or []
                except Exception:
                    splits = []
                split_map = {}
                for split in splits:
                    sp = path / split
                    if sp.exists():
                        try:
                            split_map[split if split != "val" else "validation"] = _load_split(sp)
                        except Exception as exc:  # pragma: no cover - logging happens via caller
                            self.errors[f"dataset_split_{split}"] = str(exc)
                # fall back to any subdirs if splits empty
                if not split_map:
                    for sp in path.iterdir():
                        if sp.is_dir():
                            try:
                                split_map[sp.name if sp.name != "val" else "validation"] = _load_split(sp)
                            except Exception as exc:
                                self.errors[f"dataset_split_{sp.name}"] = str(exc)
                if split_map:
                    return DatasetDict(split_map)
            # try normal load_from_disk
            obj = load_from_disk(str(path))
            if isinstance(obj, DatasetDict):
                # normalize val -> validation
                if "val" in obj and "validation" not in obj:
                    obj["validation"] = obj.pop("val")
                return obj
            if isinstance(obj, Dataset):
                return DatasetDict({"train": obj})
            raise TypeError(f"Unsupported dataset object type: {type(obj)}")

        last_err: Optional[Exception] = None
        candidates = [dataset_dir]
        if dataset_dir.name != "dataset_dict":
            candidates.append(dataset_dir / "dataset_dict")
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                return _load_candidate(candidate)
            except Exception as exc:
                last_err = exc
                continue
        raise RuntimeError(f"Failed to load dataset from {dataset_dir}: {last_err}")

    def _run_inference(
        self,
        words: List[str],
        bboxes: List[List[int]],
        image_path: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        min_conf: float = 0.0,
        source: str = "dataset",
    ) -> Tuple[List[dict], dict]:
        assert self._model and self._processor
        image = Image.open(image_path).convert("RGB")
        encoding = self._processor(
            image,
            words,
            boxes=bboxes,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self._model(
                input_ids=encoding["input_ids"].to(self.device),
                attention_mask=encoding["attention_mask"].to(self.device),
                bbox=encoding["bbox"].to(self.device),
                pixel_values=encoding["pixel_values"].to(self.device),
            )
            logits = outputs.logits.cpu()
            probs = F.softmax(logits, dim=-1)

        word_ids = encoding.word_ids(batch_index=0)
        per_word: Dict[int, List[Tuple[int, float]]] = {}
        for idx, wid in enumerate(word_ids):
            if wid is None:
                continue
            conf, pred_id = probs[0, idx].max(dim=-1)
            per_word.setdefault(wid, []).append((int(pred_id), float(conf.item())))

        tokens: List[dict] = []
        for wid, scores in per_word.items():
            best_label, best_conf = max(scores, key=lambda t: t[1])
            if best_conf < min_conf:
                continue
            tokens.append(
                {
                    "text": words[wid],
                    "bbox": bboxes[wid],
                    "pred_label": self._model.config.id2label[int(best_label)],
                    "pred_id": int(best_label),
                    "confidence": best_conf,
                }
            )

        meta = {
            "model_dir": str(self.model_dir),
            "dataset_dir": str(self.dataset_dir),
            "device": str(self.device),
            "source": source,
            "width": width,
            "height": height,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "num_tokens": len(tokens),
        }
        return tokens, meta

    def _build_overlay(
        self,
        page_num: int,
        image_path: str,
        tokens: List[dict],
        grouped: bool,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> dict:
        grouped_tokens: Dict[str, List[dict]] = {}
        if grouped:
            for tok in tokens:
                grouped_tokens.setdefault(tok["pred_label"], []).append(tok)
        id2label = {int(k): v for k, v in self._model.config.id2label.items()}  # type: ignore
        label2id = {v: k for k, v in id2label.items()}
        result = {
            "page_num": page_num,
            "image_path": image_path,
            "image_url": f"/api/parse/boston/{int(page_num)}/image",
            "tokens": tokens,
            "grouped": grouped_tokens if grouped else None,
            "meta": {
                "model_dir": str(self.model_dir),
                "dataset_dir": str(self.dataset_dir),
                "device": str(self.device),
            },
            "label_map": {"id2label": id2label, "label2id": label2id},
        }
        # Include page dimensions if available (needed for demo overlay)
        if width is not None and height is not None:
            result["page"] = {
                "width": width,
                "height": height,
                "coordSpace": "pixel",
            }
        return result

    def predict_from_ocr(
        self,
        words: List[str],
        bboxes: List[List[int]],
        image_path: str,
        width: Optional[int],
        height: Optional[int],
        grouped: bool = True,
        min_conf: float = 0.0,
    ) -> dict:
        self.load()
        tokens, meta = self._run_inference(
            words=words,
            bboxes=bboxes,
            image_path=image_path,
            width=width,
            height=height,
            min_conf=min_conf,
            source="upload",
        )
        overlay = self._build_overlay(
            page_num=-1,
            image_path=image_path,
            tokens=tokens,
            grouped=grouped,
            width=width,
            height=height,
        )
        overlay["meta"].update(meta)
        return overlay


_service: Optional[LayoutLMService] = None


def get_service() -> LayoutLMService:
    global _service
    if _service is None:
        _service = LayoutLMService()
    return _service
