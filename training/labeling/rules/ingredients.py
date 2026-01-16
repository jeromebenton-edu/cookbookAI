"""Ingredient line heuristics."""

import re
from typing import List

from .utils import has_quantity, has_unit


def score_ingredient(text: str, tokens: List[str]) -> tuple[float, list[str]]:
    score = 0.0
    signals: list[str] = []

    if has_quantity(tokens):
        score += 0.4
        signals.append("quantity")
    if has_unit(tokens):
        score += 0.3
        signals.append("unit")
    if re.search(r"\bchopped\b|\bground\b|\bminced\b", text, re.I):
        score += 0.1
        signals.append("prep_word")
    if len(tokens) <= 2:
        score -= 0.2
        signals.append("too_short")
    return score, signals
