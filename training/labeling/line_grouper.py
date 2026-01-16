"""Group OCR tokens into lines using bbox proximity heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

BBox = Tuple[int, int, int, int]


@dataclass
class Line:
    line_id: int
    text: str
    word_indices: List[int]
    line_bbox: BBox
    features: dict


def group_lines(words: Sequence[str], bboxes: Sequence[BBox], page_height: int) -> List[Line]:
    indexed = []
    for idx, (word, bbox) in enumerate(zip(words, bboxes)):
        x0, y0, x1, y1 = bbox
        y_center = (y0 + y1) / 2
        indexed.append((idx, word, bbox, y_center))

    indexed.sort(key=lambda item: (item[3], item[2][0]))
    lines: List[List[Tuple[int, str, BBox]]] = []
    current: List[Tuple[int, str, BBox]] = []
    current_y = None

    # Estimate threshold: median height * 0.8
    heights = [int(bbox[3] - bbox[1]) for _, _, bbox, _ in indexed] or [1]
    heights_sorted = sorted(heights)
    median_height = heights_sorted[len(heights_sorted) // 2]
    y_threshold = max(4, int(median_height * 0.8))

    for item in indexed:
        _, _, bbox, y_center = item
        if current_y is None:
            current_y = y_center
        if current and abs(y_center - current_y) > y_threshold:
            lines.append(current)
            current = []
        current.append(item[:3])
        current_y = (current_y + y_center) / 2 if current_y is not None else y_center
    if current:
        lines.append(current)

    line_objs: List[Line] = []
    for line_id, tokens in enumerate(lines, start=1):
        tokens.sort(key=lambda t: t[2][0])
        word_indices = [int(idx) for idx, _, _ in tokens]
        texts = [w for _, w, _ in tokens]
        x0 = int(min(b[0] for _, _, b in tokens))
        y0 = int(min(b[1] for _, _, b in tokens))
        x1 = int(max(b[2] for _, _, b in tokens))
        y1 = int(max(b[3] for _, _, b in tokens))
        bbox = (x0, y0, x1, y1)
        avg_height = float(sum((b[3] - b[1]) for _, _, b in tokens) / len(tokens))
        avg_width = float(sum((b[2] - b[0]) for _, _, b in tokens) / len(tokens))
        x_span = float(x1 - x0)
        y_pos_norm = y0 / max(page_height, 1)
        line_objs.append(
            Line(
                line_id=line_id,
                text=" ".join(texts),
                word_indices=word_indices,
                line_bbox=bbox,
                features={
                    "avg_height": avg_height,
                    "avg_width": avg_width,
                    "x_span": x_span,
                    "y_pos_norm": y_pos_norm,
                },
            )
        )
    return line_objs
