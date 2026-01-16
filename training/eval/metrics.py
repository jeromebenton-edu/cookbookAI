"""Token classification metrics using seqeval + confusion matrix output."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from seqeval.metrics import classification_report as seqeval_classification_report
from seqeval.metrics import f1_score as seqeval_f1_score
from seqeval.metrics import precision_score as seqeval_precision_score
from seqeval.metrics import recall_score as seqeval_recall_score
from sklearn.metrics import (
    classification_report as sklearn_classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_token_metrics(y_true_flat: np.ndarray, y_pred_flat: np.ndarray, label_names: List[str]) -> Dict:
    """
    Compute token-level classification metrics for non-BIO token classification.

    This function computes metrics appropriate for standard token classification
    where labels are not in BIO format. It focuses on non-O labels for the main
    metrics (precision/recall/F1) while also providing full confusion matrix.

    Args:
        y_true_flat: Flattened array of true label IDs (already filtered, no -100)
        y_pred_flat: Flattened array of predicted label IDs (same length as y_true_flat)
        label_names: List of label names indexed by label ID

    Returns:
        Dictionary with precision, recall, f1, per-label report, and confusion matrix
    """
    # Get unique labels present in data
    all_label_ids = sorted(set(y_true_flat) | set(y_pred_flat))
    all_label_names = [label_names[i] for i in all_label_ids if i < len(label_names)]

    # Identify non-O labels for computing main metrics
    non_o_indices = [i for i, name in enumerate(all_label_names) if name != 'O']

    # Compute micro-average metrics over non-O labels (main metrics)
    if non_o_indices:
        # Filter to only non-O labels
        mask = np.isin(y_true_flat, [all_label_ids[i] for i in non_o_indices])
        y_true_non_o = y_true_flat[mask]
        y_pred_non_o = y_pred_flat[mask]

        if len(y_true_non_o) > 0:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true_non_o,
                y_pred_non_o,
                labels=[all_label_ids[i] for i in non_o_indices],
                average='micro',
                zero_division=0
            )
        else:
            precision, recall, f1 = 0.0, 0.0, 0.0
    else:
        precision, recall, f1 = 0.0, 0.0, 0.0

    # Compute per-label metrics using sklearn
    per_label_report = sklearn_classification_report(
        y_true_flat,
        y_pred_flat,
        labels=all_label_ids,
        target_names=all_label_names,
        output_dict=True,
        zero_division=0
    )

    # Compute confusion matrix over all labels (including O for debugging)
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=all_label_ids)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "report": per_label_report,
        "labels": all_label_names,
        "confusion_matrix": cm.tolist(),
    }

    return metrics


def compute_seqeval(y_true: List[List[str]], y_pred: List[List[str]]) -> Dict:
    """
    Compute metrics using seqeval (for BIO-tagged sequences).

    NOTE: This function is kept for backwards compatibility but will issue
    warnings if used with non-BIO labels. For standard token classification
    without BIO tagging, use compute_token_metrics instead.
    """
    # flatten for confusion matrix
    flat_true = [lbl for seq in y_true for lbl in seq]
    flat_pred = [lbl for seq in y_pred for lbl in seq]
    labels = sorted(set(flat_true) | set(flat_pred))
    metrics = {
        "precision": seqeval_precision_score(y_true, y_pred),
        "recall": seqeval_recall_score(y_true, y_pred),
        "f1": seqeval_f1_score(y_true, y_pred),
        "report": seqeval_classification_report(y_true, y_pred, output_dict=True),
        "labels": labels,
    }
    cm = confusion_matrix(flat_true, flat_pred, labels=labels)
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def save_report(metrics: Dict, out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2))
    lines = [
        "# Token classification report",
        f"Precision: {metrics['precision']:.3f}",
        f"Recall: {metrics['recall']:.3f}",
        f"F1: {metrics['f1']:.3f}",
        "",
        "## Per-label",
    ]
    for lbl, row in metrics["report"].items():
        if lbl in {"macro avg", "weighted avg", "micro avg"}:
            continue
        lines.append(f"- {lbl}: P={row['precision']:.3f}, R={row['recall']:.3f}, F1={row['f1-score']:.3f}, support={row['support']}")
    out_md.write_text("\n".join(lines))


def save_confusion_csv(metrics: Dict, out_csv: Path) -> None:
    labels = metrics.get("labels", [])
    cm = np.array(metrics.get("confusion_matrix", []))
    if cm.size == 0:
        return
    rows = ["," + ",".join(labels)]
    for lbl, row in zip(labels, cm):
        rows.append(lbl + "," + ",".join(map(str, row)))
    out_csv.write_text("\n".join(rows))
