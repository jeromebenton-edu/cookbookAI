"""
Legacy label list - DEPRECATED.

Use ml.config.labels instead for the canonical label taxonomy.
"""

# Import from canonical source
import sys
from pathlib import Path

ml_path = Path(__file__).parent.parent.parent / "ml"
if str(ml_path) not in sys.path:
    sys.path.insert(0, str(ml_path))

from config.labels import LABELS_V3 as LABELS

__all__ = ["LABELS"]
