"""Time, temperature, servings heuristics."""

import re
from typing import List

TIME_PATTERN = re.compile(r"\b\d+\s*(min|mins|minutes|hour|hours)\b", re.I)
TEMP_PATTERN = re.compile(r"\b\d{3}\s*°?\s*f\b|degrees", re.I)
SERVINGS_PATTERN = re.compile(r"\b(serves|servings|yield|makes)\b", re.I)


def score_time(text: str) -> tuple[float, list[str]]:
    if TIME_PATTERN.search(text):
        return 0.9, ["time_pattern"]
    return 0.0, []


def score_temp(text: str) -> tuple[float, list[str]]:
    if TEMP_PATTERN.search(text):
        return 0.9, ["temp_pattern"]
    return 0.0, []


def score_servings(text: str) -> tuple[float, list[str]]:
    if SERVINGS_PATTERN.search(text):
        return 0.8, ["servings_pattern"]
    return 0.0, []
