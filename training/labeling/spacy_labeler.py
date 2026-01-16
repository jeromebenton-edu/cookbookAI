"""Weak labeling rules using spaCy + heuristics."""

from __future__ import annotations

import re
from typing import List, Sequence

import spacy
from spacy.language import Language

from .confidence import score_to_confidence
from .line_grouper import Line
from .rules import ingredients, instructions, metadata, notes, title
from .rules.utils import cooking_verbs
from .labels import LABELS

# Preload model lazily
_nlp: Language | None = None


def get_nlp() -> Language:
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        except OSError:
            _nlp = spacy.blank("en")
            _nlp.add_pipe("sentencizer")
    return _nlp


def label_lines(lines: Sequence[Line]) -> List[dict]:
    nlp = get_nlp()
    results: List[dict] = []
    for line in lines:
        doc = nlp(line.text)
        tokens = [token.text for token in doc]
        lower_tokens = [t.lower() for t in tokens]
        scores = {label: 0.0 for label in LABELS}
        signals: dict[str, list[str]] = {label: [] for label in LABELS}

        # Title
        s, sig = title.score_title(line.text, tokens, line.features.get("y_pos_norm"))
        scores["TITLE"] += s
        signals["TITLE"].extend(sig)

        # Ingredients
        s, sig = ingredients.score_ingredient(line.text, tokens)
        scores["INGREDIENT_LINE"] += s
        signals["INGREDIENT_LINE"].extend(sig)

        # Instructions
        s, sig = instructions.score_instruction(line.text, tokens)
        scores["INSTRUCTION_STEP"] += s
        signals["INSTRUCTION_STEP"].extend(sig)

        # Time/Temp/Servings
        t_score, t_sig = metadata.score_time(line.text)
        scores["TIME"] += t_score
        signals["TIME"].extend(t_sig)

        temp_score, temp_sig = metadata.score_temp(line.text)
        scores["TEMP"] += temp_score
        signals["TEMP"].extend(temp_sig)

        serv_score, serv_sig = metadata.score_servings(line.text)
        scores["SERVINGS"] += serv_score
        signals["SERVINGS"].extend(serv_sig)

        # Notes
        n_score, n_sig = notes.score_note(line.text, tokens)
        scores["NOTE"] += n_score
        signals["NOTE"].extend(n_sig)

        # Penalties
        if len(tokens) <= 1:
            for key in scores:
                scores[key] -= 0.1
                signals[key].append("very_short")
        if len(tokens) > 14:
            scores["TITLE"] -= 0.2
            signals["TITLE"].append("long_line")

        # Default O
        scores["O"] = 0.0

        best_label = max(scores.items(), key=lambda kv: kv[1])[0]
        confidence = score_to_confidence(scores[best_label])
        results.append(
            {
                "line_id": line.line_id,
                "text": line.text,
                "word_indices": line.word_indices,
                "line_bbox": line.line_bbox,
                "label": best_label,
                "confidence": confidence,
                "signals": signals[best_label],
            }
        )
    return results


def assign_token_labels(num_tokens: int, line_predictions: Sequence[dict], ocr_confidences: Sequence[int] | None) -> tuple[List[str], List[float]]:
    labels = ["O"] * num_tokens
    token_confidence = [0.0] * num_tokens

    for line in line_predictions:
        for idx in line["word_indices"]:
            if idx >= num_tokens:
                continue
            labels[idx] = line["label"]
            base_conf = line["confidence"]
            if ocr_confidences:
                conf_adjust = (ocr_confidences[idx] / 100) * 0.2  # small boost
            else:
                conf_adjust = 0.0
            token_confidence[idx] = min(1.0, base_conf + conf_adjust)

    return labels, token_confidence
