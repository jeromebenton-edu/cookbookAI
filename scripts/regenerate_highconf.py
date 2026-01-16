#!/usr/bin/env python3
"""Regenerate structural high-confidence labels and dataset, relaxing thresholds until non-empty."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

LOG = logging.getLogger("regen_highconf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate high-confidence structural labels and dataset.")
    parser.add_argument("--weak_jsonl", default="data/_generated/labels/boston_body_weak.jsonl")
    parser.add_argument("--out_highconf_jsonl", default="data/_generated/labels/boston_body_weak_highconf_structural.jsonl")
    parser.add_argument("--out_dataset_dir", default="data/_generated/datasets/boston_layoutlmv3_highconf")
    parser.add_argument("--min_pages", type=int, default=25, help="Stop when at least this many pages pass.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    # structural thresholds
    parser.add_argument("--min_avg_token_conf", type=float, default=0.60)
    parser.add_argument("--min_tokens", type=int, default=80)
    parser.add_argument("--max_o_ratio", type=float, default=0.92)
    parser.add_argument("--min_ingredient_lines", type=int, default=2)
    parser.add_argument("--min_instruction_lines", type=int, default=2)
    parser.add_argument("--min_labeled_token_ratio", type=float, default=0.03)
    parser.add_argument("--max_iters", type=int, default=5)
    return parser.parse_args()


def relax_thresholds(th: Dict[str, float]) -> Dict[str, float]:
    """Relax thresholds slightly to admit more pages."""
    new = dict(th)
    new["min_avg_token_conf"] = max(0.50, new["min_avg_token_conf"] - 0.05)
    new["min_tokens"] = max(50, new["min_tokens"] - 10)
    new["max_o_ratio"] = min(0.98, new["max_o_ratio"] + 0.02)
    new["min_ingredient_lines"] = max(1, new["min_ingredient_lines"] - 1)
    new["min_instruction_lines"] = max(1, new["min_instruction_lines"] - 1)
    new["min_labeled_token_ratio"] = max(0.015, new["min_labeled_token_ratio"] - 0.005)
    return new


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in f if _.strip())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    weak_path = Path(args.weak_jsonl)
    if not weak_path.exists():
        fallback = Path("data/labels/boston_weak_labeled.jsonl")
        if fallback.exists():
            LOG.warning("Weak JSONL %s missing; falling back to %s", weak_path, fallback)
            weak_path = fallback
        else:
            raise FileNotFoundError(f"Weak JSONL not found: {weak_path}")

    out_highconf = Path(args.out_highconf_jsonl)
    out_dataset_dir = Path(args.out_dataset_dir)
    report_md = out_highconf.parent / "highconf_regeneration_report.md"
    config_json = out_highconf.parent / "highconf_regeneration_config.json"

    if args.overwrite:
        out_highconf.unlink(missing_ok=True)
        if out_dataset_dir.exists():
            shutil.rmtree(out_dataset_dir)

    attempts: List[Dict[str, float]] = []
    thresholds = {
        "min_avg_token_conf": args.min_avg_token_conf,
        "min_tokens": args.min_tokens,
        "max_o_ratio": args.max_o_ratio,
        "min_ingredient_lines": args.min_ingredient_lines,
        "min_instruction_lines": args.min_instruction_lines,
        "min_labeled_token_ratio": args.min_labeled_token_ratio,
    }

    kept = 0
    attempt_num = 0
    stats_path = out_highconf.parent / "highconf_filter_stats.json"
    filter_report = out_highconf.parent / "highconf_filter_report.md"

    while attempt_num < args.max_iters:
        attempt_num += 1
        attempts.append(dict(thresholds))
        LOG.info("Attempt %s with thresholds %s", attempt_num, thresholds)
        cmd_filter = [
            "python",
            "-m",
            "training.labeling.filter_highconf_structural",
            "--in_jsonl",
            str(weak_path),
            "--out_jsonl",
            str(out_highconf),
            "--out_report_md",
            str(filter_report),
            "--out_stats_json",
            str(stats_path),
            "--min_avg_token_conf",
            str(thresholds["min_avg_token_conf"]),
            "--min_tokens",
            str(thresholds["min_tokens"]),
            "--max_o_ratio",
            str(thresholds["max_o_ratio"]),
            "--min_ingredient_lines",
            str(thresholds["min_ingredient_lines"]),
            "--min_instruction_lines",
            str(thresholds["min_instruction_lines"]),
            "--min_labeled_token_ratio",
            str(thresholds["min_labeled_token_ratio"]),
            "--seed",
            str(args.seed),
        ]
        rc = subprocess.run(cmd_filter, capture_output=True, text=True)
        if rc.returncode != 0:
            LOG.error("Filter failed: %s", rc.stderr)
            raise SystemExit(1)
        kept = count_jsonl(out_highconf)
        LOG.info("Attempt %s kept %s pages", attempt_num, kept)
        if kept >= args.min_pages:
            break
        thresholds = relax_thresholds(thresholds)

    if kept == 0:
        LOG.error("No pages kept after %s attempts. Please broaden weak labels or inspect %s", attempt_num, filter_report)
        raise SystemExit(1)

    # Build dataset
    cmd_build = [
        "python",
        "-m",
        "training.datasets.build_layoutlm_dataset",
        "--in_jsonl",
        str(out_highconf),
        "--out_dir",
        str(out_dataset_dir),
        "--val_ratio",
        "0.15",
        "--test_ratio",
        "0.0",
        "--seed",
        str(args.seed),
        "--collapse_note",
        "--min_tokens",
        str(thresholds["min_tokens"]),
    ]
    rc = subprocess.run(cmd_build, capture_output=True, text=True)
    if rc.returncode != 0:
        LOG.error("Dataset build failed: %s", rc.stderr)
        raise SystemExit(1)

    report_lines = [
        "# Highconf Regeneration Report",
        f"Weak labels: {weak_path}",
        f"Output JSONL: {out_highconf}",
        f"Output dataset: {out_dataset_dir}",
        f"Attempts: {attempt_num}",
        f"Final kept pages: {kept}",
        f"Filter report: {filter_report}",
        f"Filter stats: {stats_path}",
        "",
        "## Attempts",
    ]
    for i, t in enumerate(attempts, 1):
        report_lines.append(f"- Attempt {i}: {t}")
    report_lines.append("")
    report_lines.append(f"Chosen thresholds: {thresholds}")
    report_lines.append(f"min_pages target: {args.min_pages}")
    report_md.write_text("\n".join(report_lines))

    config_json.write_text(
        json.dumps(
            {
                "attempts": attempts,
                "final_thresholds": thresholds,
                "kept_pages": kept,
                "min_pages": args.min_pages,
                "weak_jsonl": str(weak_path),
                "out_jsonl": str(out_highconf),
                "out_dataset_dir": str(out_dataset_dir),
            },
            indent=2,
        )
    )
    LOG.info("Regenerated highconf dataset with %s pages -> %s", kept, out_dataset_dir)


if __name__ == "__main__":
    main()
