"""Shared helpers for rules."""

import re
from typing import List

from data.scripts import recipe_signals  # type: ignore


def has_quantity(tokens: List[str]) -> bool:
    for tok in tokens:
        if re.match(r"^\d+$", tok) or re.match(r"^\d+/\d+$", tok):
            return True
    return False


def has_unit(tokens: List[str]) -> bool:
    return any(tok.lower() in recipe_signals.UNITS for tok in tokens)


def cooking_verbs() -> set[str]:
    return set(recipe_signals.COOKING_VERBS)
