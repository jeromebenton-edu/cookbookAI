"""
Legacy label schema - DEPRECATED.

Use ml.config.labels instead for the canonical label taxonomy.
This file is kept for backward compatibility with existing scripts.
"""

from __future__ import annotations

# Import from canonical source
import sys
from pathlib import Path

# Add ml module to path
ml_path = Path(__file__).parent.parent.parent / "ml"
if str(ml_path) not in sys.path:
    sys.path.insert(0, str(ml_path))

from config.labels import LABELS_V3 as LABELS, label2id, id2label

# Re-export for backward compatibility
__all__ = ["LABELS", "label2id", "id2label"]
