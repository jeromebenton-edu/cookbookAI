"""Confidence helpers for weak labels."""

import math
from typing import Iterable


def score_to_confidence(score: float) -> float:
    # Map score (roughly -1..1) to 0..1 via sigmoid-like function.
    return 1 / (1 + math.exp(-3 * score))


def average_confidences(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)
