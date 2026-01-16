#!/usr/bin/env python3
"""Option 2 pipeline runner: render -> OCR -> detect -> weak-label -> datasets -> overlays."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

LOG = logging.getLogger("option2")


def run(cmd: List[str], dry: bool = False) -> None:
    LOG.info("Running: %s", " ".join(cmd))
    if dry:
        return
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        sys.exit(res.returncode)


def file_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def render_pages(args, dry: bool) -> None:
    out_dir = Path("data/_generated/pages/boston_body")
    ensure_dir(out_dir)
    sample_png = out_dir / f"{args.start:04d}.png"
    if file_exists(sample_png) and not args.overwrite:
        LOG.info("Step 0: render skipped (files exist)")
        return
    cmd = [
        "python",
        "data/scripts/render_pdf_pages.py",
        "--pdf",
        args.pdf,
        "--out",
        str(out_dir),
        "--dpi",
        str(args.dpi),
        "--start",
        str(args.start),
        "--end",
        str(args.end),
        "--overwrite",
    ]
    run(cmd, dry=dry)


def ocr_pages(args, dry: bool) -> Path:
    out_jsonl = Path("data/_generated/ocr/boston_body.jsonl")
    ensure_dir(out_jsonl.parent)
    if file_exists(out_jsonl) and not args.overwrite:
        LOG.info("Step 0b: OCR skipped (file exists)")
        return out_jsonl
    cmd = [
        "python",
        "data/scripts/ocr_pages.py",
        "--images_dir",
        "data/_generated/pages/boston_body",
        "--out",
        str(out_jsonl),
        "--ocr_backend",
        args.ocr_backend,
        "--lang",
        "eng",
        "--max_pages",
        str(args.max_pages),
    ]
    if args.preprocess:
        cmd.append("--preprocess")
    run(cmd, dry=dry)
    return out_jsonl


def jsonl_to_parquet(args, ocr_jsonl: Path, dry: bool) -> Path:
    out_parquet = Path("data/_generated/ocr/boston_body.parquet")
    if file_exists(out_parquet) and not args.overwrite:
        LOG.info("Step 0c: jsonl->parquet skipped (file exists)")
        return out_parquet
    cmd = [
        "python",
        "data/scripts/jsonl_to_parquet.py",
        "--in",
        str(ocr_jsonl),
        "--out",
        str(out_parquet),
    ]
    run(cmd, dry=dry)
    return out_parquet


def detect_recipes(args, parquet_path: Path, dry: bool) -> Path:
    out_csv = Path("data/_generated/ocr/boston_recipe_candidates.csv")
    out_md = Path("data/_generated/ocr/boston_recipe_candidates_top50.md")
    ensure_dir(out_csv.parent)
    if file_exists(out_csv) and not args.overwrite:
        LOG.info("Step 1: detect skipped (file exists)")
        return out_csv
    cmd = [
        "python",
        "-m",
        "data.scripts.detect_recipe_pages",
        "--in",
        str(parquet_path),
        "--out_csv",
        str(out_csv),
        "--out_jsonl",
        "data/_generated/ocr/boston_recipe_candidates.jsonl",
        "--out_md",
        str(out_md),
        "--threshold",
        str(args.candidate_threshold),
    ]
    run(cmd, dry=dry)
    return out_csv


def weak_label(args, parquet_path: Path, candidates_csv: Path, dry: bool) -> Dict[str, Path]:
    out_full = Path("data/_generated/labels/boston_body_weak.jsonl")
    out_high = Path("data/_generated/labels/boston_body_weak_highconf.jsonl")
    ensure_dir(out_full.parent)
    if file_exists(out_full) and file_exists(out_high) and not args.overwrite:
        LOG.info("Step 2: weak labeling skipped (files exist)")
        return {"full": out_full, "high": out_high}
    cmd = [
        "python",
        "-m",
        "training.labeling.run_weak_labeling",
        "--parquet",
        str(parquet_path),
        "--recipe_candidates",
        str(candidates_csv),
        "--out",
        str(out_full),
        "--out_highconf",
        str(out_high),
        "--candidate_threshold",
        str(args.candidate_threshold),
        "--min_avg_line_conf",
        str(args.min_avg_line_conf),
        "--max_pages",
        str(args.max_pages),
    ]
    if args.use_all_pages:
        cmd.append("--use_all_pages")
    run(cmd, dry=dry)
    if out_high.exists() and out_high.stat().st_size == 0:
        LOG.warning("High-confidence output is empty. Consider lowering --min_avg_line_conf or --candidate_threshold.")
    return {"full": out_full, "high": out_high}


def structural_filter(args, weak_full: Path, dry: bool) -> Dict[str, Path]:
    out_jsonl = Path("data/_generated/labels/boston_body_weak_highconf_structural.jsonl")
    out_md = Path("data/_generated/labels/highconf_filter_report.md")
    out_stats = Path("data/_generated/labels/highconf_filter_stats.json")
    ensure_dir(out_jsonl.parent)
    if file_exists(out_jsonl) and not args.overwrite:
        LOG.info("Step 2b: structural filter skipped (file exists)")
        return {"structural": out_jsonl, "report": out_md, "stats": out_stats}
    cmd = [
        "python",
        "-m",
        "training.labeling.filter_highconf_structural",
        "--in_jsonl",
        str(weak_full),
        "--out_jsonl",
        str(out_jsonl),
        "--out_report_md",
        str(out_md),
        "--out_stats_json",
        str(out_stats),
        "--min_avg_token_conf",
        str(args.min_avg_token_conf),
        "--min_tokens",
        str(args.min_tokens_struct),
        "--max_o_ratio",
        str(args.max_o_ratio),
        "--min_ingredient_lines",
        str(args.min_ingredient_lines),
        "--min_instruction_lines",
        str(args.min_instruction_lines),
        "--min_labeled_token_ratio",
        str(args.min_labeled_token_ratio),
    ]
    run(cmd, dry=dry)
    try:
        stats = json.loads(out_stats.read_text())
        LOG.info("Structural highconf kept %s / %s pages (report: %s)", stats.get("kept_pages"), stats.get("total_input_pages"), out_md)
    except Exception:
        LOG.warning("Could not read structural stats at %s", out_stats)
    return {"structural": out_jsonl, "report": out_md, "stats": out_stats}


def easy_pages(args, weak_full: Path, dry: bool) -> None:
    out_jsonl = Path("data/_generated/labels/boston_body_easy_pages.jsonl")
    out_md = Path("data/_generated/labels/easy_pages_report.md")
    if file_exists(out_jsonl) and not args.overwrite:
        LOG.info("Easy pages selection skipped (exists)")
        return
    cmd = [
        "python",
        "-m",
        "training.labeling.select_easy_pages",
        "--in_jsonl",
        str(weak_full),
        "--out_jsonl",
        str(out_jsonl),
        "--out_report_md",
        str(out_md),
        "--max_pages",
        str(args.easy_max_pages),
    ]
    run(cmd, dry=dry)


def build_datasets(args, weak_paths: Dict[str, Path], dry: bool) -> None:
    targets = [
        ("full", weak_paths["full"], Path("data/_generated/datasets/boston_layoutlmv3_full")),
        ("highconf", weak_paths["high"], Path("data/_generated/datasets/boston_layoutlmv3_highconf")),
    ]
    for name, jsonl_path, out_dir in targets:
        ensure_dir(out_dir)
        ds_marker = out_dir / "dataset_dict"
        if ds_marker.exists() and not args.overwrite:
            LOG.info("Step 3: build %s skipped (exists)", name)
            continue
        cmd = [
            "python",
            "-m",
            "training.datasets.build_layoutlm_dataset",
            "--in_jsonl",
            str(jsonl_path),
            "--out_dir",
            str(out_dir),
            "--val_ratio",
            str(args.val_ratio),
            "--test_ratio",
            str(args.test_ratio),
            "--seed",
            str(args.seed),
            "--min_tokens",
            str(args.min_tokens),
        ]
        if args.collapse_note:
            cmd.append("--collapse_note")
        else:
            cmd.append("--no_collapse_note")
        run(cmd, dry=dry)
        if name == "highconf" and (not jsonl_path.exists() or jsonl_path.stat().st_size == 0):
            LOG.warning("Highconf JSONL empty; dataset will be empty.")


def sanity_overlays(args, dataset_dir: Path, dry: bool) -> None:
    ensure_dir(Path(args.overlay_out))
    cmd = [
        "python",
        "-m",
        "training.datasets.sanity_check",
        "--dataset_dir",
        str(dataset_dir),
        "--num_samples",
        str(args.num_overlay_samples),
        "--split",
        "train",
        "--out_dir",
        args.overlay_out,
    ]
    run(cmd, dry=dry)


def write_overlay_readme(out_dir: Path) -> None:
    text = [
        "# Sanity overlays",
        "",
        "Rendered samples with non-O labels drawn in red. Use to verify bbox normalization and label alignment.",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text("\n".join(text))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Option 2 pipeline end-to-end")
    parser.add_argument("--config", help="YAML config path", default=None)
    parser.add_argument("--pdf", help="PDF path")
    parser.add_argument("--start", type=int, default=120)
    parser.add_argument("--end", type=int, default=420)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--ocr_backend", default="tesseract")
    parser.add_argument("--candidate_threshold", type=float, default=0.30)
    parser.add_argument("--min_avg_line_conf", type=float, default=0.65)
    parser.add_argument("--max_pages", type=int, default=400)
    parser.add_argument("--use_all_pages", action="store_true", default=True)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_tokens", type=int, default=40)
    parser.add_argument("--min_tokens_struct", type=int, default=80)
    parser.add_argument("--min_avg_token_conf", type=float, default=0.60)
    parser.add_argument("--max_o_ratio", type=float, default=0.92)
    parser.add_argument("--min_ingredient_lines", type=int, default=2)
    parser.add_argument("--min_instruction_lines", type=int, default=2)
    parser.add_argument("--min_labeled_token_ratio", type=float, default=0.03)
    parser.add_argument("--num_overlay_samples", type=int, default=3)
    parser.add_argument("--easy_max_pages", type=int, default=50)
    parser.add_argument("--overlay_out", default="data/reports/sanity_overlays")
    parser.add_argument("--preprocess", action="store_true", default=False)
    parser.add_argument("--collapse_note", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument(
        "--start_after",
        choices=["render", "ocr", "detect", "weaklabel", "structural_filter", "easy_pages", "build_datasets", "sanity"],
        default=None,
    )
    parser.add_argument(
        "--stop_after",
        choices=["render", "ocr", "detect", "weaklabel", "structural_filter", "easy_pages", "build_datasets", "sanity"],
        default=None,
    )
    parser.add_argument("--dry_run", action="store_true", default=False)
    defaults = parser.parse_args([])
    args = parser.parse_args()
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())
        for k, v in cfg.items():
            if hasattr(args, k) and getattr(args, k) == getattr(defaults, k):
                setattr(args, k, v)
    if not args.pdf:
        parser.error("--pdf is required (or set in --config)")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    steps = ["render", "ocr", "detect", "weaklabel", "structural_filter", "easy_pages", "build_datasets", "sanity"]
    if args.start_after:
        start_idx = steps.index(args.start_after) + 1
        steps = steps[start_idx:]
    if args.stop_after:
        stop_idx = steps.index(args.stop_after) + 1
        steps = steps[:stop_idx]

    ocr_jsonl: Optional[Path] = None
    parquet_path: Optional[Path] = None
    candidates_csv: Optional[Path] = None
    weak_paths: Optional[Dict[str, Path]] = None
    structural_paths: Optional[Dict[str, Path]] = None

    for step in steps:
        if step == "render":
            render_pages(args, dry=args.dry_run)
        elif step == "ocr":
            render_pages(args, dry=args.dry_run)
            ocr_jsonl = ocr_pages(args, dry=args.dry_run)
            parquet_path = jsonl_to_parquet(args, ocr_jsonl, dry=args.dry_run)
        elif step == "detect":
            if parquet_path is None:
                render_pages(args, dry=args.dry_run)
                ocr_jsonl = ocr_pages(args, dry=args.dry_run)
                parquet_path = jsonl_to_parquet(args, ocr_jsonl, dry=args.dry_run)
            candidates_csv = detect_recipes(args, parquet_path, dry=args.dry_run)
        elif step == "weaklabel":
            if parquet_path is None or candidates_csv is None:
                render_pages(args, dry=args.dry_run)
                ocr_jsonl = ocr_pages(args, dry=args.dry_run)
                parquet_path = jsonl_to_parquet(args, ocr_jsonl, dry=args.dry_run)
                candidates_csv = detect_recipes(args, parquet_path, dry=args.dry_run)
            weak_paths = weak_label(args, parquet_path, candidates_csv, dry=args.dry_run)
        elif step == "structural_filter":
            if weak_paths is None:
                if parquet_path is None or candidates_csv is None:
                    render_pages(args, dry=args.dry_run)
                    ocr_jsonl = ocr_pages(args, dry=args.dry_run)
                    parquet_path = jsonl_to_parquet(args, ocr_jsonl, dry=args.dry_run)
                    candidates_csv = detect_recipes(args, parquet_path, dry=args.dry_run)
                weak_paths = weak_label(args, parquet_path, candidates_csv, dry=args.dry_run)
            structural_paths = structural_filter(args, weak_paths["full"], dry=args.dry_run)
        elif step == "easy_pages":
            if weak_paths is None:
                if parquet_path is None or candidates_csv is None:
                    render_pages(args, dry=args.dry_run)
                    ocr_jsonl = ocr_pages(args, dry=args.dry_run)
                    parquet_path = jsonl_to_parquet(args, ocr_jsonl, dry=args.dry_run)
                    candidates_csv = detect_recipes(args, parquet_path, dry=args.dry_run)
                weak_paths = weak_label(args, parquet_path, candidates_csv, dry=args.dry_run)
            easy_pages(args, weak_paths["full"], dry=args.dry_run)
        elif step == "build_datasets":
            if structural_paths is None or weak_paths is None:
                if parquet_path is None or candidates_csv is None:
                    render_pages(args, dry=args.dry_run)
                    ocr_jsonl = ocr_pages(args, dry=args.dry_run)
                    parquet_path = jsonl_to_parquet(args, ocr_jsonl, dry=args.dry_run)
                    candidates_csv = detect_recipes(args, parquet_path, dry=args.dry_run)
                weak_paths = weak_label(args, parquet_path, candidates_csv, dry=args.dry_run)
                structural_paths = structural_filter(args, weak_paths["full"], dry=args.dry_run)
            # swap highconf to structural output
            weak_paths_for_build = {
                "full": weak_paths["full"],
                "high": structural_paths["structural"] if structural_paths else weak_paths["high"],
            }
            build_datasets(args, weak_paths_for_build, dry=args.dry_run)
        elif step == "sanity":
            dataset_dir = Path("data/_generated/datasets/boston_layoutlmv3_full")
            sanity_overlays(args, dataset_dir=dataset_dir, dry=args.dry_run)
            write_overlay_readme(Path(args.overlay_out))
        else:
            LOG.error("Unknown step: %s", step)
            sys.exit(1)


if __name__ == "__main__":
    main()
