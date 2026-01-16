"""Notes / variation heuristics."""

import re
from typing import List

NOTE_PREFIX = re.compile(r"^(note|variation|to serve|n\.b\.)", re.I)


def score_note(text: str, tokens: List[str]) -> tuple[float, list[str]]:
    score = 0.0
    signals: list[str] = []

    if NOTE_PREFIX.search(text.strip()):
        score += 0.8
        signals.append("note_prefix")
    if len(tokens) > 12:
        score -= 0.2
        signals.append("long_line")
    return score, signals
