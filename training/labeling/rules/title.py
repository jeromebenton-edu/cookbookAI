"""Title heuristics."""

from typing import List


def score_title(text: str, tokens: List[str], y_pos_norm: float) -> tuple[float, list[str]]:
    score = 0.0
    signals: list[str] = []

    if len(tokens) <= 6 and text[:1].isupper():
        score += 0.4
        signals.append("short_title")
    if y_pos_norm is not None and y_pos_norm < 0.15:
        score += 0.3
        signals.append("top_of_page")
    if text.isupper():
        score += 0.2
        signals.append("all_caps")
    if len(tokens) <= 2:
        score += 0.1
        signals.append("single_line")
    return score, signals
