from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

MARKERS = {"Makefile", "frontend", "backend"}


def get_project_root(start: Optional[Path] = None) -> Path:
    """
    Walk parent directories looking for repo markers.
    Returns current working directory as a fallback.
    """
    start_path = start or Path(__file__).resolve().parent
    for parent in [start_path] + list(start_path.parents):
        marker_hits = [parent / m for m in MARKERS]
        if all(p.exists() for p in marker_hits):
            return parent
    return Path.cwd()


def get_pages_dir() -> Path:
    """
    Resolve the cookbook page images directory with env override.
    Default: <PROJECT_ROOT>/data/pages/boston
    """
    env_override = os.getenv("COOKBOOKAI_PAGES_DIR")
    root = get_project_root()
    if env_override:
        p = Path(env_override)
        if not p.is_absolute():
            p = root / p
        return p
    return root / "data" / "pages" / "boston"
