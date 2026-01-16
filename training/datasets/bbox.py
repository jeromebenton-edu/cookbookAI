"""BBox normalization helpers for LayoutLM-style models."""

from __future__ import annotations

from typing import Iterable, List, Sequence


def _clamp(val: float, lower: int = 0, upper: int = 1000) -> int:
    return max(lower, min(upper, int(val)))


def normalize_bbox(bbox: Sequence[float], width: int, height: int) -> List[int]:
    """Normalize a single bbox to 0..1000 space."""
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 values, got {bbox}")
    x0, y0, x1, y1 = bbox
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height must be positive, got {(width, height)}")
    nx0 = _clamp(1000 * x0 / width)
    ny0 = _clamp(1000 * y0 / height)
    nx1 = _clamp(1000 * x1 / width)
    ny1 = _clamp(1000 * y1 / height)
    if nx1 < nx0:
        nx1 = nx0
    if ny1 < ny0:
        ny1 = ny0
    return [nx0, ny0, nx1, ny1]


def normalize_bboxes(bboxes: Iterable[Sequence[float]], width: int, height: int) -> List[List[int]]:
    return [normalize_bbox(b, width, height) for b in bboxes]
