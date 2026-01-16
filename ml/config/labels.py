"""
Canonical label taxonomy for LayoutLMv3 token classification.

This is the single source of truth for all label mappings used across:
- Dataset preparation (training/datasets/)
- Model training (training/train_layoutlmv3.py)
- Inference (backend/ml/, tools/generate_demo_fixtures.py)
- Evaluation metrics

Version: v3 (header-aware taxonomy)
Date: 2026-01-11

Changes from v2:
- Split TITLE into: PAGE_HEADER, SECTION_HEADER, RECIPE_TITLE
- This allows models to distinguish:
  * PAGE_HEADER: "BOSTON COOKING-SCHOOL COOK BOOK" (running headers)
  * SECTION_HEADER: "BISCUITS, BREAKFAST CAKES, ETC." (category headings)
  * RECIPE_TITLE: "Waffles" (actual recipe title)
"""

from __future__ import annotations
from typing import Dict, List

# Label taxonomy v3: Header-aware classification
LABELS_V3 = [
    "PAGE_HEADER",        # Book title, page numbers, running headers
    "SECTION_HEADER",     # Category headings (BISCUITS, SOUPS, etc.)
    "RECIPE_TITLE",       # Actual recipe title
    "INGREDIENT_LINE",    # Ingredient list items
    "INSTRUCTION_STEP",   # Cooking instruction steps
    "TIME",               # Time specifications
    "TEMP",               # Temperature specifications
    "SERVINGS",           # Serving size / yield
    "NOTE",               # Additional notes / tips
    "O",                  # Other / background
]

# Legacy label set (v2) - for backward compatibility
LABELS_V2 = [
    "TITLE",              # Generic title (no header distinction)
    "INGREDIENT_LINE",
    "INSTRUCTION_STEP",
    "TIME",
    "TEMP",
    "SERVINGS",
    "NOTE",
    "O",
]

# Default to v3
LABELS = LABELS_V3

def get_label_config(version: str = "v3") -> Dict[str, any]:
    """
    Get label configuration for a specific version.

    Args:
        version: "v2" (legacy) or "v3" (header-aware)

    Returns:
        Dictionary with:
        - labels: List of label names
        - label2id: Label name -> ID mapping
        - id2label: ID -> label name mapping
        - num_labels: Total number of labels
        - version: Version string
    """
    if version == "v2":
        labels = LABELS_V2
    elif version == "v3":
        labels = LABELS_V3
    else:
        raise ValueError(f"Unknown label version: {version}. Use 'v2' or 'v3'")

    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    return {
        "labels": labels,
        "label2id": label2id,
        "id2label": id2label,
        "num_labels": len(labels),
        "version": version,
    }


# Default mappings (v3)
label2id = {label: idx for idx, label in enumerate(LABELS)}
id2label = {idx: label for label, idx in label2id.items()}
num_labels = len(LABELS)


def map_v2_to_v3(v2_label: str) -> str:
    """
    Map legacy v2 label to v3 taxonomy.

    This is a HEURISTIC mapping for automated conversion.
    For production datasets, use proper relabeling with human review.

    Args:
        v2_label: Label from v2 schema

    Returns:
        Corresponding v3 label
    """
    if v2_label == "TITLE":
        # Default TITLE -> RECIPE_TITLE
        # Relabeling script will refine this to PAGE_HEADER/SECTION_HEADER
        return "RECIPE_TITLE"
    elif v2_label in LABELS_V3:
        # Already valid in v3
        return v2_label
    else:
        # Unknown -> O
        return "O"


def get_header_labels() -> List[str]:
    """Get labels that represent headers (not recipe content)."""
    return ["PAGE_HEADER", "SECTION_HEADER"]


def get_recipe_content_labels() -> List[str]:
    """Get labels that represent actual recipe content."""
    return ["RECIPE_TITLE", "INGREDIENT_LINE", "INSTRUCTION_STEP"]


def get_metadata_labels() -> List[str]:
    """Get labels for recipe metadata."""
    return ["TIME", "TEMP", "SERVINGS", "NOTE"]


# Export for backward compatibility
__all__ = [
    "LABELS",
    "LABELS_V2",
    "LABELS_V3",
    "label2id",
    "id2label",
    "num_labels",
    "get_label_config",
    "map_v2_to_v3",
    "get_header_labels",
    "get_recipe_content_labels",
    "get_metadata_labels",
]
