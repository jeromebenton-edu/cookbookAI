"""Select easy pages for human labeling from weak-labeled JSONL."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Iterable, List

from tqdm import tqdm

LOG = logging.getLogger(__name__)


def stream_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def write_jsonl(path: Path, recs: Iterable[dict]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in recs:
            f.write(json.dumps(rec) + "\n")
            count += 1
    return count


def metrics(rec: dict) -> dict:
    labels = rec.get("labels") or []
    lines = rec.get("lines") or []
    token_conf = rec.get("token_confidence") or rec.get("token_conf") or []
    if token_conf:
        avg_conf = sum(token_conf) / max(1, len(token_conf))
    elif lines:
        avg_conf = sum(ln.get("confidence", 0) for ln in lines) / max(1, len(lines))
    else:
        avg_conf = 0.0
    o_count = sum(1 for lbl in labels if lbl == "O" or lbl == "NOTE")
    total = len(labels)
    o_ratio = o_count / max(1, total)
    line_counts = Counter(ln.get("label") for ln in lines)
    return {
        "avg_conf": avg_conf,
        "o_ratio": o_ratio,
        "tokens": total,
        "title": line_counts.get("TITLE", 0),
        "ing": line_counts.get("INGREDIENT_LINE", 0),
        "instr": line_counts.get("INSTRUCTION_STEP", 0),
    }


def passes_easy(m: dict, args: argparse.Namespace) -> bool:
    if m["title"] != args.title_lines:
        return False
    if not (args.min_ing <= m["ing"] <= args.max_ing):
        return False
    if not (args.min_instr <= m["instr"] <= args.max_instr):
        return False
    if m["avg_conf"] < args.min_avg_conf:
        return False
    if m["o_ratio"] > args.max_o_ratio:
        return False
    if m["tokens"] < args.min_tokens:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Select easy pages for human labeling")
    parser.add_argument("--in_jsonl", required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--out_report_md", required=True)
    parser.add_argument("--max_pages", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title_lines", type=int, default=1)
    parser.add_argument("--min_ing", type=int, default=2)
    parser.add_argument("--max_ing", type=int, default=12)
    parser.add_argument("--min_instr", type=int, default=1)
    parser.add_argument("--max_instr", type=int, default=8)
    parser.add_argument("--min_avg_conf", type=float, default=0.60)
    parser.add_argument("--max_o_ratio", type=float, default=0.90)
    parser.add_argument("--min_tokens", type=int, default=80)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    kept: List[dict] = []
    for rec in tqdm(stream_jsonl(Path(args.in_jsonl)), desc="Selecting easy pages"):
        m = metrics(rec)
        if passes_easy(m, args):
            rec["easy_metrics"] = m
            kept.append(rec)
        if args.max_pages and len(kept) >= args.max_pages:
            break

    write_jsonl(Path(args.out_jsonl), kept)

    lines = [
        "# Easy pages report",
        f"Selected {len(kept)} pages (max {args.max_pages})",
        "",
        "## Pages (top by avg_conf)",
    ]
    kept_sorted = sorted(kept, key=lambda r: r["easy_metrics"]["avg_conf"], reverse=True)
    for rec in kept_sorted[: args.max_pages]:
        m = rec["easy_metrics"]
        title = ""
        ing = ""
        for ln in rec.get("lines") or []:
            if ln.get("label") == "TITLE" and not title:
                title = ln.get("text", "")
            if ln.get("label") == "INGREDIENT_LINE" and not ing:
                ing = ln.get("text", "")
        lines.append(
            f"- Page {rec.get('page_num')}: conf={m['avg_conf']:.2f}, O={m['o_ratio']:.2f}, ing={m['ing']}, instr={m['instr']}"
        )
        if title or ing:
            lines.append(f"  - {title} | {ing}")

    Path(args.out_report_md).write_text("\n".join(lines))
    LOG.info("Easy pages: %s saved to %s", len(kept), args.out_jsonl)


if __name__ == "__main__":
    main()
