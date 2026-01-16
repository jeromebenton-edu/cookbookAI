from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from app.utils.spell_correct import correct_title, correct_ingredient


@dataclass
class Token:
    text: str
    bbox: List[int]
    label: str
    confidence: float

    @property
    def mid_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def mid_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2


@dataclass
class Line:
    id: str
    text: str
    bbox: List[int]
    confidence: float
    token_count: int
    tokens: List[Token]


def tokens_to_lines(tokens: List[Token], y_thresh: int = 12) -> List[List[Token]]:
    """Cluster tokens into lines using y midpoint proximity."""
    if not tokens:
        return []
    # sort by y, then x
    tokens = sorted(tokens, key=lambda t: (t.mid_y, t.mid_x))
    lines: List[List[Token]] = []
    current: List[Token] = []
    for tok in tokens:
        if not current:
            current.append(tok)
            continue
        last = current[-1]
        if abs(tok.mid_y - last.mid_y) <= y_thresh:
            current.append(tok)
        else:
            lines.append(sorted(current, key=lambda t: t.bbox[0]))
            current = [tok]
    if current:
        lines.append(sorted(current, key=lambda t: t.bbox[0]))
    return lines


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:])", r"\\1", text)
    return text


def _line_bbox(line: List[Token]) -> List[int]:
    xs0 = [t.bbox[0] for t in line]
    ys0 = [t.bbox[1] for t in line]
    xs1 = [t.bbox[2] for t in line]
    ys1 = [t.bbox[3] for t in line]
    return [min(xs0), min(ys0), max(xs1), max(ys1)]


def _lines_for_label(grouped: Dict[str, List[Token]], label: str, prefix: str, y_thresh: int = 10) -> List[Line]:
    toks = grouped.get(label) or []
    if not toks:
        return []
    lines = tokens_to_lines(toks, y_thresh=y_thresh)
    line_objs: List[Line] = []
    for idx, line in enumerate(lines, start=1):
        txt = clean_text(" ".join([t.text for t in line]))
        if not txt:
            continue

        # Apply spell correction to ingredients and instructions
        txt = correct_ingredient(txt)

        confs = [t.confidence for t in line]
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        line_objs.append(
            Line(
                id=f"{prefix}_{idx:04d}",
                text=txt,
                bbox=_line_bbox(line),
                confidence=avg_conf,
                token_count=len(line),
                tokens=line,
            )
        )
    return line_objs


def extract_title_heuristic(all_tokens: List[Token]) -> Tuple[Line | None, float]:
    """
    Heuristic fallback for title extraction when ML model fails.
    Strategy: Find first line of capitalized text near top of page that ends with period.
    """
    # Common instruction/continuation words to skip (likely from previous page)
    instruction_words = {"To", "Put", "Add", "Heat", "Cook", "Mix", "Stir", "Pour", "Follow", "Remove", "Place", "Serve"}

    # Filter to top portion of page (y < 400 pixels) and exclude page headers
    candidates = [
        t for t in all_tokens
        if t.bbox[1] < 400  # Top of page
        and t.bbox[1] > 50  # Below page header
        and len(t.text) > 1  # Not punctuation
        and (t.text[0].isupper() or t.text[0] == '(')  # Starts with capital or parenthesis
        and t.text not in instruction_words  # Skip instruction words
    ]

    if not candidates:
        return None, 0.0

    # Sort by position
    candidates = sorted(candidates, key=lambda t: (t.bbox[1], t.bbox[0]))

    # Find first group of tokens on same line (within 12px Y threshold)
    # This filters out scattered marginal notes
    title_tokens = []
    y_thresh = 12

    for token in candidates:
        if not title_tokens:
            title_tokens.append(token)
            continue

        # Check if token is on same line as previous tokens
        avg_y = sum(t.mid_y for t in title_tokens) / len(title_tokens)

        if abs(token.mid_y - avg_y) <= y_thresh:
            # Same line - add it
            title_tokens.append(token)

            # Stop at period
            if token.text.rstrip().endswith('.'):
                break

            # Stop if we've collected 8 tokens
            if len(title_tokens) >= 8:
                break
        else:
            # Different line
            # If we already have 2+ tokens on current line, we found a title candidate
            if len(title_tokens) >= 2:
                # Check if we should continue to next line or stop
                # Stop if current line has a period
                if any(t.text.rstrip().endswith('.') for t in title_tokens):
                    break
                # Otherwise reset and start new line
                title_tokens = [token]
            else:
                # Less than 2 tokens, likely scattered words - reset and continue
                title_tokens = [token]

    # Need at least 2 tokens to be a valid title
    if len(title_tokens) < 2:
        return None, 0.0

    # Sort tokens by X position (left to right) for proper reading order
    title_tokens = sorted(title_tokens, key=lambda t: t.bbox[0])

    # Combine and clean
    combined_text = " ".join(t.text for t in title_tokens)
    combined_text = combined_text.strip().lstrip("'\"\u2018\u2019\u201c\u201d\u201a\u201e").rstrip("'\"\u2018\u2019\u201c\u201d\u201a\u201e")
    combined_text = clean_text(combined_text)

    # Apply spell correction to fix common OCR errors
    combined_text = correct_title(combined_text)

    # Mark as low confidence since it's heuristic
    confidence = 0.5

    # Calculate bounding box
    x0 = min(t.bbox[0] for t in title_tokens)
    y0 = min(t.bbox[1] for t in title_tokens)
    x1 = max(t.bbox[2] for t in title_tokens)
    y1 = max(t.bbox[3] for t in title_tokens)

    title_line = Line(
        id="title_0001_heuristic",
        text=combined_text,
        bbox=[x0, y0, x1, y1],
        confidence=confidence,
        token_count=len(title_tokens),
        tokens=title_tokens,
    )
    return title_line, confidence


def extract_title_obj(grouped: Dict[str, List[Token]], all_tokens: List[Token] = None) -> Tuple[Line | None, float]:
    titles = grouped.get("RECIPE_TITLE") or []

    # Always try heuristic for comparison
    heuristic_title, heuristic_conf = None, 0.0
    if all_tokens:
        heuristic_title, heuristic_conf = extract_title_heuristic(all_tokens)

    # If no ML titles, use heuristic
    if not titles:
        if heuristic_title:
            return heuristic_title, heuristic_conf
        return None, 0.0

    # Filter to titles in upper portion of page only (ignore secondary recipes)
    # Take titles with y < 350 (roughly top third of typical page)
    upper_titles = [t for t in titles if t.bbox[1] < 350]

    # If no upper titles, fall back to heuristic
    if not upper_titles:
        if heuristic_title:
            return heuristic_title, heuristic_conf
        # If no heuristic either, use all ML titles
        upper_titles = titles

    # Use the filtered titles for further processing
    titles = upper_titles

    # Check if ML titles are in suspicious positions
    ml_positions = [t.bbox[1] for t in titles]
    avg_y_position = sum(ml_positions) / len(ml_positions)

    # If ML titles are still too far down (y > 400), prefer heuristic
    if avg_y_position > 400 and heuristic_title:
        return heuristic_title, heuristic_conf

    # If ML has too many tokens (>=8), likely concatenating - use heuristic
    if len(titles) >= 8 and heuristic_title:
        return heuristic_title, heuristic_conf

    # Group tokens by line (tokens with similar Y positions are on the same line)
    # Then sort by X position (left-to-right) within each line
    # This ensures we read "(Junket)" is the rightmost word in a title
    y_thresh = 12

    # Find tokens that are on the title line (use first token as reference)
    if not titles:
        return None, 0.0

    first_y = titles[0].mid_y
    title_line_tokens = [t for t in titles if abs(t.mid_y - first_y) <= y_thresh]

    # Sort by X position (left to right)
    sorted_titles = sorted(title_line_tokens, key=lambda t: t.bbox[0])

    # Take tokens until we hit a period
    primary_titles = []
    for token in sorted_titles:
        primary_titles.append(token)

        # Stop at first period
        if token.text.rstrip().endswith('.'):
            break

        # Safety: stop after 8 tokens max
        if len(primary_titles) >= 8:
            break

    # Combine title tokens into a single line
    combined_text = " ".join(t.text for t in primary_titles)

    # Clean: strip leading/trailing quotes, apostrophes, and whitespace
    # Handle both ASCII quotes ('") and Unicode quotes (''""‚„)
    combined_text = combined_text.strip().lstrip("'\"\u2018\u2019\u201c\u201d\u201a\u201e").rstrip("'\"\u2018\u2019\u201c\u201d\u201a\u201e")
    combined_text = clean_text(combined_text)

    # Apply spell correction to fix common OCR errors
    combined_text = correct_title(combined_text)

    # Calculate average confidence
    avg_confidence = sum(t.confidence for t in primary_titles) / len(primary_titles)

    # Calculate bounding box that encompasses all title tokens
    x0 = min(t.bbox[0] for t in primary_titles)
    y0 = min(t.bbox[1] for t in primary_titles)
    x1 = max(t.bbox[2] for t in primary_titles)
    y1 = max(t.bbox[3] for t in primary_titles)

    title_line = Line(
        id="title_0001",
        text=combined_text,
        bbox=[x0, y0, x1, y1],
        confidence=avg_confidence,
        token_count=len(primary_titles),
        tokens=primary_titles,
    )
    return title_line, avg_confidence


def _line_to_dict(line: Line, include_tokens: bool = False) -> dict:
    data = {
        "id": line.id,
        "text": line.text,
        "bbox": line.bbox,
        "confidence": line.confidence,
        "token_count": line.token_count,
    }
    if include_tokens:
        data["tokens"] = [t.__dict__ for t in line.tokens]
    return data


def recipe_from_prediction(pred: dict, include_raw: bool = False, include_lines: bool = True) -> dict:
    # Create list of ALL tokens (including "O") for heuristic fallback
    all_tokens = [
        Token(
            text=t["text"],
            bbox=t["bbox"],
            label=t.get("pred_label", t.get("label", "O")),
            confidence=float(t.get("confidence", t.get("score", 0.0))),
        )
        for t in pred.get("tokens", [])
    ]

    # Create list of labeled tokens only for ML-based extraction
    tokens = [t for t in all_tokens if t.label != "O"]

    grouped: Dict[str, List[Token]] = {}
    for tok in tokens:
        grouped.setdefault(tok.label, []).append(tok)

    # Pass all_tokens for heuristic fallback
    title_obj, title_conf = extract_title_obj(grouped, all_tokens)
    ing_lines = _lines_for_label(grouped, "INGREDIENT_LINE", "ing")
    instr_lines = _lines_for_label(grouped, "INSTRUCTION_STEP", "ins")

    overall_conf = _weighted_conf(
        [
            ("title", title_conf, 1.0 if title_obj else 0.0),
            ("ingredients", _avg_conf(ing_lines), 1.5 if ing_lines else 0.0),
            ("instructions", _avg_conf(instr_lines), 1.5 if instr_lines else 0.0),
        ]
    )

    # For historical recipes, instructions are often a single paragraph
    # Join all instruction lines into one if there are multiple short lines
    if instr_lines and len(instr_lines) > 1:
        # Check if lines look like a paragraph (most lines don't end with periods)
        non_terminal = sum(1 for line in instr_lines if not line.text.rstrip().endswith(('.', '!')))
        if non_terminal / len(instr_lines) > 0.5:  # More than half don't end with punctuation
            # Join into single instruction
            combined_text = " ".join(line.text for line in instr_lines)
            instructions = [combined_text]
        else:
            instructions = [line.text for line in instr_lines]
    else:
        instructions = [line.text for line in instr_lines]

    recipe = {
        "page_num": pred.get("page_num"),
        "title": title_obj.text if title_obj else "",
        "title_obj": _line_to_dict(title_obj) if title_obj else None,
        "ingredients_lines": [_line_to_dict(line) for line in ing_lines] if include_lines else None,
        "instruction_lines": [_line_to_dict(line) for line in instr_lines] if include_lines else None,
        "ingredients": [line.text for line in ing_lines],
        "instructions": instructions,
        "confidence": {
            "title": title_conf,
            "ingredients": _avg_conf(ing_lines),
            "instructions": _avg_conf(instr_lines),
            "overall": overall_conf,
        },
        "meta": {
            "source": "layoutlmv3_predictions",
            "notes": ["NOTE collapsed to O in training"],
        },
    }
    if include_raw:
        recipe["raw"] = {"grouped_tokens": {k: [t.__dict__ for t in v] for k, v in grouped.items()}}
    return recipe


def _weighted_conf(items: List[Tuple[str, float, float]]) -> float:
    total_weight = sum(w for _, _, w in items if w > 0)
    if total_weight == 0:
        return 0.0
    return sum(conf * w for _, conf, w in items) / total_weight


def _avg_conf(lines: List[Line]) -> float:
    if not lines:
        return 0.0
    return sum(l.confidence for l in lines) / len(lines)


# Cache helpers ---------------------------------------------------------------
def cache_recipe(out_path: Path, data: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))


def load_cached_recipe(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None
