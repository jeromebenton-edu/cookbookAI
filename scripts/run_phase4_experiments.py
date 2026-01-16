#!/usr/bin/env python3
"""One-command Phase 4 runner: debug smoke, Stage A, Stage B, inference, eval, report."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from datasets import load_from_disk

LOG = logging.getLogger("phase4_runner")


def run_cmd(cmd, log_file: Path, dry: bool = False) -> int:
    LOG.info("Running: %s", " ".join(cmd))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        log_file.write_text("DRY RUN: " + " ".join(cmd))
        return 0
    with log_file.open("w") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return res.returncode


def load_ds_sizes(ds_path: Path) -> Dict[str, int]:
    if ds_path.name == "dataset_dict":
        ds = load_from_disk(str(ds_path))
        base = ds_path.parent
    else:
        ds = load_from_disk(str(ds_path / "dataset_dict"))
        base = ds_path
    sizes = {}
    for split in ("train", "val", "test"):
        if split in ds:
            sizes[split] = len(ds[split])
    sizes["base_dir"] = str(base)
    return sizes


def load_ds(path: Path):
    if path.name == "dataset_dict":
        return load_from_disk(str(path))
    return load_from_disk(str(path / "dataset_dict"))


def pick_inference_page(ds_path: Path, preferred_split: str, page_num_arg: str) -> tuple[Optional[int], Optional[str], List[str]]:
    """Return chosen page_num, split, and warnings."""
    warnings: List[str] = []
    ds = load_ds(ds_path)
    splits_order = []
    if preferred_split in ds:
        splits_order.append(preferred_split)
    alt = "val" if preferred_split.startswith("val") else "validation"
    if alt in ds and alt not in splits_order:
        splits_order.append(alt)
    for s in ("train", "val", "validation"):
        if s in ds and s not in splits_order:
            splits_order.append(s)

    def page_exists(num: int) -> bool:
        for s in ds:
            if "page_num" in ds[s].features and num in set(ds[s]["page_num"]):
                return True
        return False

    chosen_page: Optional[int] = None
    chosen_split: Optional[str] = None

    if page_num_arg != "auto":
        try:
            requested = int(page_num_arg)
            if page_exists(requested):
                chosen_page = requested
                # find split containing it (first match)
                for s in splits_order:
                    if requested in set(ds[s]["page_num"]):
                        chosen_split = s
                        break
            else:
                warnings.append(f"Requested page_num {requested} not found; falling back to auto selection.")
        except ValueError:
            warnings.append(f"Invalid infer_page_num {page_num_arg}; falling back to auto.")

    if chosen_page is None:
        for s in splits_order:
            if len(ds[s]) > 0:
                chosen_split = s
                chosen_page = int(ds[s][0]["page_num"])
                break

    if chosen_page is None:
        warnings.append("No pages available for inference.")
    return chosen_page, chosen_split, warnings


def merge_config(args, cfg_path: Optional[str]) -> argparse.Namespace:
    defaults = vars(args).copy()
    if cfg_path:
        cfg = yaml.safe_load(Path(cfg_path).read_text())
        for k, v in cfg.items():
            if defaults.get(k) == args.__dict__.get(k):
                setattr(args, k, v)
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 4 training/inference/eval")
    parser.add_argument("--config", help="YAML config path")
    parser.add_argument("--highconf_dataset_dir", required=False)
    parser.add_argument("--full_dataset_dir", required=False)
    parser.add_argument("--infer_page_num", default="auto", help='Page num for inference, or "auto"')
    parser.add_argument("--infer_split", default="validation", choices=["validation", "val", "train"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stageA_epochs", type=float, default=5)
    parser.add_argument("--stageB_epochs", type=float, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--learning_rate_A", type=float, default=3e-5)
    parser.add_argument("--learning_rate_B", type=float, default=2e-5)
    parser.add_argument("--fp16", default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--eval_pages", type=int, default=10)
    parser.add_argument("--auto_regen_highconf", action="store_true", default=True)
    parser.add_argument("--no_auto_regen_highconf", action="store_false", dest="auto_regen_highconf")
    parser.add_argument("--min_highconf_pages", type=int, default=25)
    parser.add_argument("--run_pass1_only", action="store_true")
    parser.add_argument("--skip_pass1", action="store_true")
    parser.add_argument("--skip_pass2", action="store_true")
    parser.add_argument("--skip_pass3", action="store_true")
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--output_dir", default=None, help="Base output dir; default data/reports/phase4_runs/<run_id>")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def safe_dataset_dir(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.name != "dataset_dict" else p.parent


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    args = merge_config(args, args.config)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir) if args.output_dir else Path("data/reports/phase4_runs") / run_id
    if args.overwrite and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    env = {
        "python": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    (out_root / "env.json").write_text(json.dumps(env, indent=2))
    (out_root / "config.json").write_text(json.dumps(vars(args), indent=2))

    # datasets
    highconf_dir = safe_dataset_dir(args.highconf_dataset_dir or "data/_generated/datasets/boston_layoutlmv3_highconf")
    full_dir = safe_dataset_dir(args.full_dataset_dir or "data/_generated/datasets/boston_layoutlmv3_full")
    highconf_sizes = load_ds_sizes(highconf_dir)
    full_sizes = load_ds_sizes(full_dir)
    dataset_summary = {"highconf": highconf_sizes, "full": full_sizes}
    (out_root / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2))
    LOG.info("Highconf sizes: %s | Full sizes: %s", highconf_sizes, full_sizes)

    highconf_train = highconf_sizes.get("train", 0)
    use_full_for_stageA = highconf_train < 10
    regen_attempted = False

    # Optional auto regeneration of highconf set
    if args.auto_regen_highconf and use_full_for_stageA:
        regen_attempted = True
        regen_log = out_root / "regenerate_highconf.log"
        cmd_regen = [
            "python",
            "scripts/regenerate_highconf.py",
            "--out_dataset_dir",
            str(highconf_dir),
            "--min_pages",
            str(args.min_highconf_pages),
            "--overwrite",
        ]
        rc = run_cmd(cmd_regen, regen_log, dry=args.dry_run)
        report_note = f"- Highconf regeneration attempted rc={rc} log={regen_log}"
        # reload sizes
        try:
            highconf_sizes = load_ds_sizes(highconf_dir)
            highconf_train = highconf_sizes.get("train", 0)
            use_full_for_stageA = highconf_train < 10
            report_note += f" | new highconf train={highconf_train}"
        except Exception as e:
            report_note += f" | reload failed: {e}"
        dataset_summary = {"highconf": highconf_sizes, "full": full_sizes}
        (out_root / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2))
    else:
        report_note = None

    # seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # fp16
    fp16_flag = False
    if args.fp16 == "true":
        fp16_flag = True
    elif args.fp16 == "auto":
        fp16_flag = torch.cuda.is_available()

    report_lines = ["# Phase 4 Run", f"- run_id: {run_id}", f"- fp16: {fp16_flag} (cuda_available={torch.cuda.is_available()})"]
    report_lines.append(f"- datasets: highconf {highconf_dir} (train={highconf_train}), full {full_dir}")
    if use_full_for_stageA:
        report_lines.append("- WARNING: highconf train < 10; Stage A will use full dataset.")
    if report_note:
        report_lines.append(report_note)

    # Pass 1 smoke
    stageA_ckpt: Optional[Path] = None
    stageB_ckpt: Optional[Path] = None

    if not args.skip_pass1:
        cmd = [
            "python",
            "-m",
            "training.train_layoutlmv3",
            "--highconf_dataset_dir",
            str(full_dir if use_full_for_stageA else highconf_dir),
            "--full_dataset_dir",
            str(full_dir),
            "--stage",
            "highconf",
            "--num_train_epochs_stageA",
            "1",
            "--batch_size",
            "1",
            "--eval_batch_size",
            "1",
            "--gradient_accumulation_steps",
            "1",
            "--learning_rate",
            str(args.learning_rate_A),
            "--output_dir",
            str(out_root / "models" / "smoke"),
            "--debug",
        ]
        if fp16_flag:
            cmd.append("--fp16")
        rc = run_cmd(cmd, out_root / "pass1_logs.txt", dry=args.dry_run)
        report_lines.append(f"- Pass1 smoke test rc={rc}")
        if args.run_pass1_only:
            (out_root / "REPORT.md").write_text("\n".join(report_lines))
            return

    # Pass 2 Stage A
    if not args.skip_pass2:
        stageA_out = Path("models") / f"layoutlmv3_boston_stageA_highconf_{run_id}"
        cmd = [
            "python",
            "-m",
            "training.train_layoutlmv3",
            "--highconf_dataset_dir",
            str(full_dir if use_full_for_stageA else highconf_dir),
            "--full_dataset_dir",
            str(full_dir),
            "--stage",
            "highconf",
            "--num_train_epochs_stageA",
            str(args.stageA_epochs),
            "--batch_size",
            str(args.batch_size),
            "--eval_batch_size",
            str(args.eval_batch_size),
            "--gradient_accumulation_steps",
            str(args.grad_accum),
            "--learning_rate",
            str(args.learning_rate_A),
            "--max_length",
            str(args.max_length),
            "--seed",
            str(args.seed),
            "--output_dir",
            str(stageA_out),
        ]
        if fp16_flag:
            cmd.append("--fp16")
        rc = run_cmd(cmd, out_root / "pass2_logs.txt", dry=args.dry_run)
        report_lines.append(f"- Pass2 StageA rc={rc} out={stageA_out}")
        stageA_ckpt = stageA_out / "layoutlmv3_boston_stageA_highconf"

    # Pass 3 Stage B
    final_model_dir = None
    if not args.skip_pass3:
        stageB_out = Path("models") / f"layoutlmv3_boston_stageB_full_{run_id}"
        cmd = [
            "python",
            "-m",
            "training.train_layoutlmv3",
            "--highconf_dataset_dir",
            str(highconf_dir),
            "--full_dataset_dir",
            str(full_dir),
            "--stage",
            "full",
            "--num_train_epochs_stageB",
            str(args.stageB_epochs),
            "--batch_size",
            str(args.batch_size),
            "--eval_batch_size",
            str(args.eval_batch_size),
            "--gradient_accumulation_steps",
            str(args.grad_accum),
            "--learning_rate",
            str(args.learning_rate_B),
            "--max_length",
            str(args.max_length),
            "--seed",
            str(args.seed),
            "--output_dir",
            str(stageB_out),
        ]
        if stageA_ckpt:
            cmd.extend(["--init_checkpoint", str(stageA_ckpt)])
        if fp16_flag:
            cmd.append("--fp16")
        rc = run_cmd(cmd, out_root / "pass3_logs.txt", dry=args.dry_run)
        report_lines.append(f"- Pass3 StageB rc={rc} out={stageB_out}")
        final_model_dir = stageB_out / "layoutlmv3_boston_final"

    # Inference and eval
    inference_json = None
    overlay_png = None
    chosen_page = None
    chosen_split = None
    inference_warnings: List[str] = []
    if not args.skip_inference and final_model_dir:
        chosen_page, chosen_split, inference_warnings = pick_inference_page(full_dir, args.infer_split, str(args.infer_page_num))
        if chosen_page is None:
            report_lines.append("- WARNING: Inference skipped (no pages available).")
        else:
            inf_dir = out_root / "inference"
            inf_dir.mkdir(parents=True, exist_ok=True)
            inference_json = inf_dir / f"page_{int(chosen_page):04d}_predictions.json"
            cmd_pred = [
                "python",
                "-m",
                "training.infer.predict_page",
                "--model_dir",
                str(final_model_dir),
                "--dataset_dir",
                str(full_dir),
                "--page_num",
                str(chosen_page),
                "--out_json",
                str(inference_json),
            ]
            run_cmd(cmd_pred, inf_dir / "predict.log", dry=args.dry_run)
            overlay_dir = inf_dir / "overlays"
            overlay_dir.mkdir(parents=True, exist_ok=True)
            overlay_png = overlay_dir / f"page_{int(chosen_page):04d}_pred_overlay.png"
            cmd_overlay = [
                "python",
                "-m",
                "training.infer.render_predictions",
                "--pred_json",
                str(inference_json),
                "--out_png",
                str(overlay_png),
            ]
            run_cmd(cmd_overlay, inf_dir / "render.log", dry=args.dry_run)

    # Weak-label eval
    eval_json = out_root / "evaluation" / "weak_label_eval.json"
    eval_md = out_root / "evaluation" / "weak_label_eval.md"
    eval_csv = out_root / "evaluation" / "weak_label_eval.csv"
    if final_model_dir:
        eval_json.parent.mkdir(parents=True, exist_ok=True)
        cmd_eval = [
            "python",
            "-m",
            "training.eval.weak_label_eval",
            "--model_dir",
            str(final_model_dir),
            "--dataset_dir",
            str(full_dir),
            "--split",
            "val",
            "--num_pages",
            str(args.eval_pages),
            "--out_json",
            str(eval_json),
            "--out_md",
            str(eval_md),
            "--out_csv",
            str(eval_csv),
        ]
        run_cmd(cmd_eval, out_root / "evaluation" / "eval.log", dry=args.dry_run)

    # Final report
    report_lines.append("")
    report_lines.append("## Outputs")
    if stageA_ckpt:
        report_lines.append(f"- StageA checkpoint: {stageA_ckpt}")
    if final_model_dir:
        report_lines.append(f"- Final model: {final_model_dir}")
    if chosen_page is not None:
        report_lines.append(f"- Inference page: {chosen_page} (split={chosen_split or 'n/a'})")
    if inference_json:
        report_lines.append(f"- Inference JSON: {inference_json}")
    if overlay_png:
        report_lines.append(f"- Overlay PNG: {overlay_png}")
    if eval_json.exists():
        report_lines.append(f"- Weak-label eval: {eval_json}")
    for w in inference_warnings:
        report_lines.append(f"- WARNING: {w}")

    (out_root / "REPORT.md").write_text("\n".join(report_lines))
    LOG.info("Run complete. Report at %s", out_root / "REPORT.md")


if __name__ == "__main__":
    main()
