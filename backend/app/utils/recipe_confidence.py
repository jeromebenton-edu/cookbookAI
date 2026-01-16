from __future__ import annotations

from collections import Counter
import re
from typing import Dict, List

RECIPE_CONF_VERSION = 3

_FRACTION_RE = re.compile(r"\b\d+/\d+\b")
_UNIT_RE = re.compile(
    r"\b(?:tsp|tbsp|teaspoon(?:ful)?s?|tablespoon(?:ful)?s?|cup|cups|oz|ounce|ounces|lb|lbs?|pint|pints|quart|quarts?)\b",
    re.IGNORECASE,
)
_VERB_RE = re.compile(r"\b(beat|stir|mix|bake|boil|pour|add|serve|cook|strain)\b", re.IGNORECASE)
_NEGATIVE_RE = re.compile(r"(table of contents|contents|index|chapter|illustrations)", re.IGNORECASE)


def is_recipe_like(text: str) -> bool:
    text = text or ""
    if not text.strip():
        return False
    if _NEGATIVE_RE.search(text):
        return False
    has_fraction = bool(_FRACTION_RE.search(text))
    unit_hits = _UNIT_RE.findall(text)
    verb_hit = bool(_VERB_RE.search(text))
    # Relaxed from 2+ units to 1+ unit (small recipes may have fewer units)
    return (has_fraction or len(unit_hits) >= 1) and verb_hit


def _avg(values: List[float]) -> float:
    """Return average, guarding against empty lists."""
    return sum(values) / len(values) if values else 0.0


def _best_title(pred: dict) -> str:
    grouped = pred.get("grouped") or {}
    # Support both TITLE and RECIPE_TITLE (model may use either)
    title_tokens = grouped.get("TITLE") or grouped.get("RECIPE_TITLE") or []
    if not title_tokens:
        # fallback to tokens if grouped not present
        title_tokens = [
            t for t in pred.get("tokens", [])
            if (t.get("pred_label") or t.get("label")) in ("TITLE", "RECIPE_TITLE")
        ]
    if not title_tokens:
        return ""
    best = max(title_tokens, key=lambda t: float(t.get("confidence", 0.0)))
    return str(best.get("text", "")).strip()


def score_prediction(pred: dict) -> Dict[str, object]:
    """
    Score a prediction overlay for recipe-likeness.

    Rewards structure (title, ingredients, instructions, servings/time/temp) and healthy ratios of
    recipe tokens. Penalizes pages that are mostly OTHER/O tokens or missing core recipe signals.
    Returns a lightweight dict that can be cached and merged into API responses.
    """
    tokens = pred.get("tokens") or []
    if not tokens:
        return {
            "is_recipe_page": False,
            "recipe_confidence": 0.0,
            "label_counts": {},
            "title": "",
            "avg_token_confidence": 0.0,
            "recipe_token_ratio": 0.0,
            "recipe_conf_version": RECIPE_CONF_VERSION,
        }

    label_counts: Counter = Counter()
    recipe_tokens: List[dict] = []
    confs: List[float] = []

    for tok in tokens:
        label = tok.get("pred_label") or tok.get("label") or "O"
        label_counts[label] += 1
        if label != "O":
            recipe_tokens.append(tok)
            confs.append(float(tok.get("confidence", tok.get("score", 0.0))))

    total_tokens = len(tokens)
    recipe_ratio = len(recipe_tokens) / total_tokens if total_tokens else 0.0
    other_ratio = (label_counts.get("OTHER", 0) / len(recipe_tokens)) if recipe_tokens else 1.0
    avg_conf = _avg(confs)

    ing_tokens = label_counts.get("INGREDIENT_LINE", 0)
    instr_tokens = label_counts.get("INSTRUCTION_STEP", 0)
    # Support both TITLE and RECIPE_TITLE (model may use either)
    title_tokens = label_counts.get("TITLE", 0) + label_counts.get("RECIPE_TITLE", 0)
    support_tokens = label_counts.get("SERVINGS", 0) + label_counts.get("TIME", 0) + label_counts.get("TEMP", 0)

    # text-based hints for instructions (model can miss labels on narrative steps)
    verbs = {
        "bake",
        "boil",
        "mix",
        "stir",
        "cook",
        "blend",
        "heat",
        "simmer",
        "pour",
        "add",
        "combine",
        "cream",
        "beat",
        "spread",
        "sprinkle",
        "serve",
        "grate",
        "chop",
        "whisk",
        "fold",
        "marinate",
    }
    numbers_or_steps = 0
    verb_hits = 0
    quantity_hits = 0
    for tok in tokens:
        text = str(tok.get("text", "")).lower()
        if text.rstrip(".").isdigit():
            numbers_or_steps += 1
        if any(ch.isdigit() for ch in text) or "/" in text:
            quantity_hits += 1
        if text in verbs:
            verb_hits += 1

    instruction_signals = instr_tokens + min(numbers_or_steps, 6) * 0.5 + min(verb_hits, 6) * 0.6
    ingredient_signals = ing_tokens + min(quantity_hits, 12) * 0.3

    # Hard rejections for pathological label distributions
    # Lowered thresholds to catch smaller recipes (griddle cakes has 9 ingredients, 25 instructions)
    hard_reject = title_tokens > 50 or ing_tokens < 5 or instr_tokens < 10

    # Adjusted scoring to work better with smaller recipes
    # (griddle cakes: 9 ingredients, 25 instructions should score well)
    ingredient_score = min(1.0, ing_tokens / 25.0)  # Full score at 25+ ingredients
    instruction_score = min(1.0, instr_tokens / 30.0)  # Full score at 30+ instructions
    title_penalty = min(1.0, title_tokens / 40.0)

    score = 0.0 if hard_reject else (
        0.45 * ingredient_score + 0.45 * instruction_score + 0.10 * (1.0 - title_penalty)
    )
    score = max(0.0, min(1.0, score))

    text_blob = " ".join(str(t.get("text", "")) for t in tokens)
    recipe_like = (not hard_reject) and is_recipe_like(text_blob) and score >= 0.35  # Lowered from 0.55
    is_recipe_page = bool(recipe_like)

    return {
        "is_recipe_page": is_recipe_page,
        "recipe_confidence": round(score, 4),
        "label_counts": dict(label_counts),
        "ingredient_token_count": ing_tokens,
        "instruction_token_count": instr_tokens,
        "title_token_count": title_tokens,
        "title": _best_title(pred),
        "avg_token_confidence": round(avg_conf, 4),
        "recipe_token_ratio": round(recipe_ratio, 4),
        "recipe_conf_version": RECIPE_CONF_VERSION,
        "recipe_like": recipe_like,
        "recipe_score_components": {
            "ingredient_score": round(ingredient_score, 4),
            "instruction_score": round(instruction_score, 4),
            "title_penalty": round(title_penalty, 4),
        },
    }
