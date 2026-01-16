"""Structural high-confidence filter for weak-labeled pages."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from tqdm import tqdm

LOG = logging.getLogger(__name__)


def stream_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
            count += 1
    return count


def compute_page_metrics(rec: dict) -> dict:
    labels = rec.get("labels") or []
    token_conf = rec.get("token_confidence") or rec.get("token_conf") or []
    lines = rec.get("lines") or []
    if token_conf:
        avg_token_conf = sum(token_conf) / max(1, len(token_conf))
    elif lines:
        avg_token_conf = sum(ln.get("confidence", 0) for ln in lines) / max(1, len(lines))
    else:
        avg_token_conf = 0.0
    total_tokens = len(labels)
    o_count = sum(1 for lbl in labels if lbl == "O" or lbl == "NOTE")
    o_ratio = o_count / max(1, total_tokens)
    label_counter = Counter(labels)
    ing_lines = sum(1 for ln in lines if ln.get("label") == "INGREDIENT_LINE")
    instr_lines = sum(1 for ln in lines if ln.get("label") == "INSTRUCTION_STEP")
    title_lines = sum(1 for ln in lines if ln.get("label") == "TITLE")
    labeled_token_ratio = (total_tokens - o_count) / max(1, total_tokens)
    return {
        "avg_token_conf": avg_token_conf,
        "o_ratio": o_ratio,
        "title_lines": title_lines,
        "ingredient_lines": ing_lines,
        "instruction_lines": instr_lines,
        "total_tokens": total_tokens,
        "labeled_token_ratio": labeled_token_ratio,
        "label_counts": dict(label_counter),
    }


def passes_structural(metrics: dict, args: argparse.Namespace, rec: dict) -> tuple[bool, List[str]]:
    reasons: List[str] = []
    if metrics["title_lines"] < 1:
        reasons.append("no_title")
    if metrics["ingredient_lines"] < args.min_ingredient_lines:
        reasons.append("few_ing")
    if metrics["instruction_lines"] < args.min_instruction_lines:
        reasons.append("few_instr")
    if metrics["avg_token_conf"] < args.min_avg_token_conf:
        reasons.append("low_conf")
    if metrics["o_ratio"] > args.max_o_ratio:
        reasons.append("high_o_ratio")
    if metrics["total_tokens"] < args.min_tokens:
        reasons.append("too_short")
    if metrics["labeled_token_ratio"] < args.min_labeled_token_ratio:
        reasons.append("low_labeled_ratio")
    return (len(reasons) == 0, reasons)


def summarize_best(pages: List[dict], top_k: int = 20) -> List[dict]:
    scored = []
    for p in pages:
        m = p["highconf_metrics"]
        score = (
            m["avg_token_conf"] * 0.6
            + (1 - m["o_ratio"]) * 0.2
            + min(1.0, m["labeled_token_ratio"] * 10) * 0.2
        )
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Structural high-confidence filter")
    parser.add_argument("--in_jsonl", required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--out_report_md", required=True)
    parser.add_argument("--out_stats_json", required=True)
    parser.add_argument("--min_avg_token_conf", type=float, default=0.60)
    parser.add_argument("--min_tokens", type=int, default=80)
    parser.add_argument("--max_o_ratio", type=float, default=0.92)
    parser.add_argument("--min_ingredient_lines", type=int, default=2)
    parser.add_argument("--min_instruction_lines", type=int, default=2)
    parser.add_argument("--min_labeled_token_ratio", type=float, default=0.03)
    parser.add_argument("--max_pages", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    total = 0
    kept: List[dict] = []
    reject_reasons = Counter()
    label_dist = Counter()
    kept_metrics: List[dict] = []
    sample_iter = stream_jsonl(Path(args.in_jsonl))
    for rec in tqdm(sample_iter, desc="Filtering pages"):
        total += 1
        metrics = compute_page_metrics(rec)
        ok, reasons = passes_structural(metrics, args, rec)
        if ok:
            rec["highconf_structural"] = True
            rec["highconf_metrics"] = metrics
            kept.append(rec)
            kept_metrics.append(metrics)
            label_dist.update(rec.get("labels") or [])
        else:
            for r in reasons:
                reject_reasons[r] += 1
        if args.max_pages and total >= args.max_pages:
            break

    write_jsonl(Path(args.out_jsonl), kept)

    def agg_metric(key: str):
        vals = [m[key] for m in kept_metrics]
        if not vals:
            return {"min": 0, "max": 0, "mean": 0}
        return {"min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals)}

    stats = {
        "total_input_pages": total,
        "kept_pages": len(kept),
        "reject_reasons": dict(reject_reasons),
        "label_distribution": dict(label_dist),
        "avg_token_conf": agg_metric("avg_token_conf"),
        "o_ratio": agg_metric("o_ratio"),
        "labeled_token_ratio": agg_metric("labeled_token_ratio"),
    }
    Path(args.out_stats_json).write_text(json.dumps(stats, indent=2))

    best = summarize_best(kept, top_k=20)
    lines = [
        "# Structural High-Confidence Filter Report",
        "",
        f"Input pages: {total}",
        f"Kept pages: {len(kept)}",
        f"Reject reasons: {dict(reject_reasons)}",
        f"Avg token conf (kept): {stats['avg_token_conf']}",
        f"O ratio (kept): {stats['o_ratio']}",
        f"Labeled token ratio (kept): {stats['labeled_token_ratio']}",
        "",
        "## Best pages (top 20)",
    ]
    for rec in best:
        m = rec["highconf_metrics"]
        lines.append(
            f"- Page {rec.get('page_num')}: conf={m['avg_token_conf']:.2f}, O={m['o_ratio']:.2f}, "
            f"ing={m['ingredient_lines']}, instr={m['instruction_lines']}, labeled_ratio={m['labeled_token_ratio']:.3f}"
        )
    Path(args.out_report_md).write_text("\n".join(lines))

    LOG.info("Filtered %s -> kept %s. Report: %s", total, len(kept), args.out_report_md)


if __name__ == "__main__":
    main()
