"""Instruction line heuristics."""

import re
from typing import List

from .utils import cooking_verbs


def score_instruction(text: str, tokens: List[str]) -> tuple[float, list[str]]:
    score = 0.0
    signals: list[str] = []

    if tokens:
        first = tokens[0].lower()
        if first in cooking_verbs():
            score += 0.4
            signals.append("verb_start")
    verb_hits = sum(1 for t in tokens if t.lower() in cooking_verbs())
    if verb_hits:
        score += min(0.2, 0.1 + 0.05 * verb_hits)
        signals.append("verb_density")
    if re.search(r"\.\s*$", text):
        score += 0.1
        signals.append("sentence")
    if len(tokens) < 3:
        score -= 0.2
        signals.append("too_short")
    return score, signals
