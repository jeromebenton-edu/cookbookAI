import re
from typing import Any, Dict, List, Tuple


def normalize_box(
    box: List[int] | Tuple[int, int, int, int],
    size: Tuple[int, int]
) -> List[int]:
    width, height = size
    x0, y0, x1, y1 = box
    return [
        int(1000 * x0 / width),
        int(1000 * y0 / height),
        int(1000 * x1 / width),
        int(1000 * y1 / height)
    ]


def group_tokens_by_line(
    tokens: List[Dict[str, Any]],
    y_threshold: int = 18
) -> List[List[Dict[str, Any]]]:
    sorted_tokens = sorted(tokens, key=lambda t: (t["bbox"][1], t["bbox"][0]))
    lines: List[List[Dict[str, Any]]] = []
    current_line: List[Dict[str, Any]] = []
    current_y = None

    for token in sorted_tokens:
        y_center = (token["bbox"][1] + token["bbox"][3]) / 2
        if current_y is None or abs(y_center - current_y) > y_threshold:
            if current_line:
                lines.append(current_line)
            current_line = [token]
            current_y = y_center
        else:
            current_line.append(token)
    if current_line:
        lines.append(current_line)

    for line in lines:
        line.sort(key=lambda t: t["bbox"][0])
    return lines


def average_score(tokens: List[Dict[str, Any]]) -> float:
    if not tokens:
        return 0.0
    return sum(token.get("score", 0.0) for token in tokens) / len(tokens)


def extract_time_and_servings(text: str) -> Tuple[str | None, str | None]:
    servings_match = re.search(r"serves?\s*(\d+(?:-\d+)?)", text, re.I)
    time_match = re.search(r"(\d{1,3})\s*(minutes|min|hrs|hours)", text, re.I)

    servings = None
    if servings_match:
        servings = f"Serves {servings_match.group(1)}"

    time_value = None
    if time_match:
        time_value = f"{time_match.group(1)} min"

    return time_value, servings


def postprocess_tokens(tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
    lines = group_tokens_by_line(tokens)
    line_texts = [" ".join(token["text"] for token in line).strip() for line in lines]

    title_line_index = None
    for index, line in enumerate(lines):
        if any(token.get("label") == "TITLE" for token in line):
            title_line_index = index
            break

    title = line_texts[title_line_index] if title_line_index is not None else ""

    ingredient_lines = [
        line
        for line in lines
        if any(token.get("label") == "INGREDIENT_LINE" for token in line)
    ]
    instruction_lines = [
        line
        for line in lines
        if any(token.get("label") == "INSTRUCTION_STEP" for token in line)
    ]
    meta_lines = [
        line
        for line in lines
        if any(token.get("label") == "META" for token in line)
    ]

    ingredients = [" ".join(token["text"] for token in line).strip() for line in ingredient_lines]
    instructions = [" ".join(token["text"] for token in line).strip() for line in instruction_lines]

    meta_text = " ".join(
        token["text"] for line in meta_lines for token in line
    )
    time_text, servings_text = extract_time_and_servings(meta_text)

    field_confidence = {
        "title": average_score([token for line in lines for token in line if token.get("label") == "TITLE"]),
        "ingredients": average_score([token for line in ingredient_lines for token in line]),
        "instructions": average_score([token for line in instruction_lines for token in line]),
        "servings": average_score([token for line in meta_lines for token in line]),
        "time": average_score([token for line in meta_lines for token in line])
    }

    return {
        "title": title or "Untitled Recipe",
        "ingredients": ingredients,
        "instructions": instructions,
        "servings": servings_text or "Serves 2-4",
        "time": {
            "totalMinutes": int(re.search(r"(\d+)", time_text).group(1))
            if time_text
            else 45
        },
        "field_confidence": field_confidence
    }
