"""Heuristic recipe-page detection on OCR Parquet.

Example:
python data/scripts/detect_recipe_pages.py \
  --in data/ocr/boston_pages.parquet \
  --out_csv data/ocr/boston_recipe_candidates.csv \
  --out_jsonl data/ocr/boston_recipe_candidates.jsonl \
  --out_md data/ocr/boston_recipe_candidates_top50.md \
  --threshold 0.65
"""

import argparse
import json
import math
import pathlib
import re
import sys
from typing import Any, Dict, List

import pandas as pd

from . import recipe_signals as signals

TIME_PATTERN = re.compile(r"\b\d+\s*(?:min|mins|minutes|hour|hours)\b", re.I)
TEMP_PATTERN = re.compile(r"\b\d{3}\s*°?\s*f\b|degrees", re.I)
FRACTION_PATTERN = re.compile(r"\b\d+/\d+\b")
NUMERIC_PATTERN = re.compile(r"\b\d+\b")


def extract_features(text: str, words: List[str], line_count: int, avg_confidence: float | None) -> Dict[str, Any]:
    lower_text = text.lower() if text else ""
    tokens_lower = [w.lower() for w in words]

    units_present = any(unit in tokens_lower for unit in signals.UNITS)
    verbs_present = any(verb in tokens_lower for verb in signals.COOKING_VERBS)
    time_present = bool(TIME_PATTERN.search(lower_text))
    temp_present = bool(TEMP_PATTERN.search(lower_text))

    numeric_tokens = [w for w in tokens_lower if NUMERIC_PATTERN.fullmatch(w) or FRACTION_PATTERN.fullmatch(w)]
    numeric_density = len(numeric_tokens) / max(1, len(tokens_lower))

    short_line_density = 0.0
    if line_count:
        short_line_density = min(1.0, line_count / max(1, len(tokens_lower)))

    toc_hit = any(term in lower_text for term in signals.TOC_TERMS)
    low_conf_penalty = avg_confidence is not None and avg_confidence < 40
    few_tokens_penalty = len(tokens_lower) < 50

    title_like = False
    if line_count and text:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            top_line = lines[0]
            title_like = len(top_line.split()) <= 6 and top_line[:1].isupper()

    return {
        "units": units_present,
        "verbs": verbs_present,
        "time": time_present,
        "temp": temp_present,
        "numeric_density": numeric_density,
        "short_line_density": short_line_density,
        "toc": toc_hit,
        "low_conf": low_conf_penalty,
        "few_tokens": few_tokens_penalty,
        "title_like": title_like,
    }


def score_page(features: Dict[str, Any]) -> float:
    score = 0.0
    if features["units"]:
        score += 0.25
    if features["verbs"]:
        score += 0.20
    if features["time"] or features["temp"]:
        score += 0.20
    score += min(0.15, features["numeric_density"])  # cap contribution
    score += min(0.10, features["short_line_density"])  # proxy for ingredient lines
    if features["title_like"]:
        score += 0.10

    if features["toc"]:
        score -= 0.40
    if features["few_tokens"]:
        score -= 0.20
    if features["low_conf"]:
        score -= 0.10

    return max(0.0, min(1.0, score))


def explain(features: Dict[str, Any]) -> List[str]:
    reasons = []
    if features["units"]:
        reasons.append("ingredient units")
    if features["verbs"]:
        reasons.append("cooking verbs")
    if features["time"]:
        reasons.append("time pattern")
    if features["temp"]:
        reasons.append("temperature pattern")
    if features["numeric_density"] > 0.05:
        reasons.append("numeric density")
    if features["short_line_density"] > 0.05:
        reasons.append("short lines")
    if features["title_like"]:
        reasons.append("title-like top line")
    if features["toc"]:
        reasons.append("toc/index penalty")
    if features["few_tokens"]:
        reasons.append("few tokens penalty")
    if features["low_conf"]:
        reasons.append("low confidence penalty")
    return reasons


def build_excerpt(text: str, limit: int = 250) -> str:
    if not text:
        return ""
    clean = " ".join(text.split())
    return clean[:limit] + ("…" if len(clean) > limit else "")


def detect_recipes(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        text = row.get("text") or ""
        words_raw = row.get("words")
        if isinstance(words_raw, list):
            words = words_raw
        elif hasattr(words_raw, "tolist"):
            words = list(words_raw.tolist())
        elif words_raw is None:
            words = []
        else:
            words = [str(words_raw)]
        features = extract_features(
            text=text,
            words=words,
            line_count=int(row.get("line_count") or 0),
            avg_confidence=float(row.get("avg_confidence")) if row.get("avg_confidence") is not None else None,
        )
        score = score_page(features)
        likely = score >= threshold
        page_num = int(row.get("page_num"))
        image_path = row.get("image_path")
        detected_signals = explain(features)

        records.append(
            {
                "page_num": page_num,
                "image_path": image_path,
                "recipe_score": score,
                "likely": likely,
                "detected_signals": detected_signals,
                "token_count": int(row.get("token_count") or 0),
                "top_keywords_found": list(_top_keywords(words)),
                "excerpt": build_excerpt(text),
                "features": features,
            }
        )

    result = pd.DataFrame(records)
    return result.sort_values("recipe_score", ascending=False)


def _top_keywords(words: List[str], limit: int = 5):
    counts: Dict[str, int] = {}
    for w in words:
        lw = w.lower()
        if len(lw) < 3:
            continue
        counts[lw] = counts.get(lw, 0) + 1
    return [w for w, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]]


def export_outputs(df: pd.DataFrame, out_csv: pathlib.Path, out_jsonl: pathlib.Path, out_md: pathlib.Path, top_n: int):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")

    top_df = df.head(top_n)
    lines = ["# Top Recipe Candidates\n"]
    for _, row in top_df.iterrows():
        lines.append(
            f"- Page {int(row.page_num)} | Score {row.recipe_score:.2f} | Signals: {', '.join(row.detected_signals)}\n  - {row.excerpt}\n"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect recipe pages from OCR Parquet")
    parser.add_argument("--in", dest="input_path", required=True, help="Input Parquet path")
    parser.add_argument("--out_csv", required=True, help="Output CSV path")
    parser.add_argument("--out_jsonl", required=True, help="Output JSONL path")
    parser.add_argument("--out_md", required=True, help="Output markdown path")
    parser.add_argument("--threshold", type=float, default=0.65, help="Score threshold for likely recipes")
    parser.add_argument("--top_n", type=int, default=50, help="How many entries to include in markdown report")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    df = pd.read_parquet(args.input_path)
    result = detect_recipes(df, threshold=args.threshold)
    export_outputs(
        df=result,
        out_csv=pathlib.Path(args.out_csv),
        out_jsonl=pathlib.Path(args.out_jsonl),
        out_md=pathlib.Path(args.out_md),
        top_n=args.top_n,
    )
    print(
        f"Processed {len(result)} pages. Candidates >= {args.threshold}: {(result.recipe_score >= args.threshold).sum()}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
