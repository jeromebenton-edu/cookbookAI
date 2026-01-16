"""
Header-aware evaluation metrics for v3 label taxonomy.

Specifically measures:
1. Recipe title correctness (independent of page headers)
2. Header false positive rate (PAGE_HEADER predicted as RECIPE_TITLE)
3. Title anchor accuracy (IoU/overlap of predicted RECIPE_TITLE bbox)
4. Section header accuracy
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_confusion_matrix(
    true_labels: List[str],
    pred_labels: List[str],
    labels_of_interest: List[str],
) -> Dict[str, Dict[str, int]]:
    """
    Compute confusion matrix for specific label pairs.

    Args:
        true_labels: Ground truth labels
        pred_labels: Predicted labels
        labels_of_interest: Labels to track in confusion matrix

    Returns:
        Nested dict: {true_label: {pred_label: count}}
    """
    confusion = defaultdict(lambda: defaultdict(int))

    for true_lbl, pred_lbl in zip(true_labels, pred_labels):
        if true_lbl in labels_of_interest or pred_lbl in labels_of_interest:
            confusion[true_lbl][pred_lbl] += 1

    return {k: dict(v) for k, v in confusion.items()}


def compute_header_title_metrics(predictions_flat, labels_flat, id2label) -> Dict:
    """
    Compute header/title-specific metrics.

    Focuses on measuring:
    - How often PAGE_HEADER is confused with RECIPE_TITLE
    - How often RECIPE_TITLE is confused with PAGE_HEADER
    - Precision/Recall for each header/title type

    Args:
        predictions_flat: Flattened model predictions (already filtered for padding)
        labels_flat: Flattened ground truth labels (already filtered for padding)
        id2label: ID to label name mapping

    Returns:
        Dictionary of metrics
    """
    # Convert IDs to label names
    all_true_labels = [id2label[label_id] for label_id in labels_flat]
    all_pred_labels = [id2label[pred_id] for pred_id in predictions_flat]

    # Compute overall counts
    total_tokens = len(all_true_labels)

    # Count true positives, false positives, false negatives for key labels
    header_labels = ["PAGE_HEADER", "SECTION_HEADER", "RECIPE_TITLE"]

    tp = {lbl: 0 for lbl in header_labels}
    fp = {lbl: 0 for lbl in header_labels}
    fn = {lbl: 0 for lbl in header_labels}

    for true_lbl, pred_lbl in zip(all_true_labels, all_pred_labels):
        for lbl in header_labels:
            if true_lbl == lbl and pred_lbl == lbl:
                tp[lbl] += 1
            elif true_lbl != lbl and pred_lbl == lbl:
                fp[lbl] += 1
            elif true_lbl == lbl and pred_lbl != lbl:
                fn[lbl] += 1

    # Compute precision, recall, F1
    metrics = {}

    for lbl in header_labels:
        precision = tp[lbl] / (tp[lbl] + fp[lbl]) if (tp[lbl] + fp[lbl]) > 0 else 0.0
        recall = tp[lbl] / (tp[lbl] + fn[lbl]) if (tp[lbl] + fn[lbl]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[f"{lbl}_precision"] = precision
        metrics[f"{lbl}_recall"] = recall
        metrics[f"{lbl}_f1"] = f1

    # Confusion counts
    confusion = compute_confusion_matrix(all_true_labels, all_pred_labels, header_labels)

    # Specific error rates
    page_header_as_recipe_title = confusion.get("PAGE_HEADER", {}).get("RECIPE_TITLE", 0)
    recipe_title_as_page_header = confusion.get("RECIPE_TITLE", {}).get("PAGE_HEADER", 0)
    total_page_headers = confusion.get("PAGE_HEADER", {}).get("PAGE_HEADER", 0) + page_header_as_recipe_title + sum(
        confusion.get("PAGE_HEADER", {}).values()
    )
    total_recipe_titles = confusion.get("RECIPE_TITLE", {}).get("RECIPE_TITLE", 0) + recipe_title_as_page_header + sum(
        confusion.get("RECIPE_TITLE", {}).values()
    )

    metrics["header_false_positive_rate"] = (
        page_header_as_recipe_title / total_page_headers if total_page_headers > 0 else 0.0
    )
    metrics["title_false_negative_rate"] = (
        recipe_title_as_page_header / total_recipe_titles if total_recipe_titles > 0 else 0.0
    )

    metrics["header_title_confusion_count"] = page_header_as_recipe_title + recipe_title_as_page_header

    # Overall label distribution
    true_dist = Counter(all_true_labels)
    pred_dist = Counter(all_pred_labels)

    metrics["total_tokens"] = total_tokens
    metrics["true_label_distribution"] = dict(true_dist)
    metrics["pred_label_distribution"] = dict(pred_dist)

    return metrics


def compute_title_anchor_accuracy(eval_dataset, predictions_batched, id2label, iou_threshold=0.3) -> Dict:
    """
    Compute title anchor accuracy: how well predicted RECIPE_TITLE bboxes overlap with ground truth.

    For each page:
    1. Extract all RECIPE_TITLE tokens (both true and predicted)
    2. Compute union bbox for each
    3. Calculate IoU
    4. Report % pages with IoU >= threshold

    Args:
        eval_dataset: Evaluation dataset (iterable of pages)
        predictions_batched: Batched model predictions (2D array: batch_size x seq_len)
        id2label: ID to label mapping
        iou_threshold: Minimum IoU to count as correct anchor

    Returns:
        Metrics dict with anchor accuracy
    """
    pages_with_title = 0
    pages_with_correct_anchor = 0

    # Flatten predictions to match dataset iteration
    # predictions_batched is (batch_size, seq_len)
    # We need to iterate through it page by page
    num_predictions = len(predictions_batched)
    num_dataset_pages = len(list(eval_dataset))

    # Reset iterator
    eval_dataset_iter = iter(eval_dataset)

    # Iterate through available predictions
    for batch_idx in range(min(num_predictions, num_dataset_pages)):
        try:
            example = next(eval_dataset_iter)
        except StopIteration:
            break

        true_label_ids = example["labels"]
        pred_label_ids = predictions_batched[batch_idx]
        bboxes = example["bboxes"]

        # Find true RECIPE_TITLE tokens
        true_title_bboxes = []
        for i, (label_id, bbox) in enumerate(zip(true_label_ids, bboxes)):
            if label_id != -100 and id2label[label_id] == "RECIPE_TITLE":
                true_title_bboxes.append(bbox)

        if not true_title_bboxes:
            # No recipe title on this page
            continue

        pages_with_title += 1

        # Find predicted RECIPE_TITLE tokens
        pred_title_bboxes = []
        for i, (label_id, bbox) in enumerate(zip(pred_label_ids, bboxes)):
            if i < len(true_label_ids) and true_label_ids[i] != -100:  # Skip padding
                if id2label[int(label_id)] == "RECIPE_TITLE":
                    pred_title_bboxes.append(bbox)

        if not pred_title_bboxes:
            # Predicted no recipe title
            continue

        # Compute union bboxes
        true_bbox = union_bbox(true_title_bboxes)
        pred_bbox = union_bbox(pred_title_bboxes)

        # Compute IoU
        iou = bbox_iou(true_bbox, pred_bbox)

        if iou >= iou_threshold:
            pages_with_correct_anchor += 1

    anchor_accuracy = (
        pages_with_correct_anchor / pages_with_title if pages_with_title > 0 else 0.0
    )

    return {
        "title_anchor_accuracy": anchor_accuracy,
        "pages_with_title": pages_with_title,
        "pages_with_correct_anchor": pages_with_correct_anchor,
        "anchor_iou_threshold": iou_threshold,
    }


def union_bbox(bboxes: List[List[int]]) -> Tuple[int, int, int, int]:
    """Compute union bounding box from list of bboxes."""
    if not bboxes:
        return (0, 0, 0, 0)

    x1 = min(bbox[0] for bbox in bboxes)
    y1 = min(bbox[1] for bbox in bboxes)
    x2 = max(bbox[2] for bbox in bboxes)
    y2 = max(bbox[3] for bbox in bboxes)

    return (x1, y1, x2, y2)


def bbox_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union of two bboxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Compute intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        # No intersection
        return 0.0

    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)

    # Compute union
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = bbox1_area + bbox2_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def print_demo_scorecard(metrics: Dict) -> None:
    """
    Print a demo-oriented scorecard showing key metrics.

    Args:
        metrics: Combined metrics dictionary (may have 'eval_' prefix on keys)
    """
    # Helper to get metric with or without eval_ prefix
    def get_metric(key: str, default=0.0):
        return metrics.get(f'eval_{key}', metrics.get(key, default))

    # Get the classification report (more reliable than custom header metrics)
    eval_report = metrics.get('eval_report', metrics.get('report', {}))

    # Helper to get metrics from sklearn classification report
    def get_report_metric(label: str, metric_name: str, default=0.0):
        if label in eval_report:
            return eval_report[label].get(metric_name, default)
        return default

    print("\n" + "=" * 80)
    print("DEMO SCORECARD - Header-Aware Evaluation")
    print("=" * 80)

    print("\n📍 Title Anchor Accuracy")
    print(f"  Anchor Accuracy (IoU >= {get_metric('anchor_iou_threshold', 0.3)}): "
          f"{get_metric('title_anchor_accuracy', 0.0):.1%}")
    print(f"  Pages with correct anchor: {int(get_metric('pages_with_correct_anchor', 0))} / "
          f"{int(get_metric('pages_with_title', 0))}")

    print("\n🎯 Recipe Title Performance")
    # NOTE: sklearn report has swapped precision/recall if y_true and y_pred were swapped
    # Use custom metrics which we know are correct
    print(f"  Precision: {get_metric('RECIPE_TITLE_precision', 0.0):.1%}")
    print(f"  Recall:    {get_metric('RECIPE_TITLE_recall', 0.0):.1%}")
    print(f"  F1:        {get_metric('RECIPE_TITLE_f1', 0.0):.1%}")
    # Support from true label distribution (sklearn report may be wrong)
    true_dist = metrics.get('eval_true_label_distribution', metrics.get('true_label_distribution', {}))
    print(f"  Support:   {true_dist.get('RECIPE_TITLE', 0)}")

    print("\n📄 Page Header Performance")
    # Use custom metrics which we know are correct
    print(f"  Precision: {get_metric('PAGE_HEADER_precision', 0.0):.1%}")
    print(f"  Recall:    {get_metric('PAGE_HEADER_recall', 0.0):.1%}")
    print(f"  F1:        {get_metric('PAGE_HEADER_f1', 0.0):.1%}")
    # Support from true label distribution (sklearn report may be wrong)
    true_dist = metrics.get('eval_true_label_distribution', metrics.get('true_label_distribution', {}))
    print(f"  Support:   {true_dist.get('PAGE_HEADER', 0)}")

    print("\n🔀 Section Header Performance")
    # Use custom metrics which we know are correct
    print(f"  Precision: {get_metric('SECTION_HEADER_precision', 0.0):.1%}")
    print(f"  Recall:    {get_metric('SECTION_HEADER_recall', 0.0):.1%}")
    print(f"  F1:        {get_metric('SECTION_HEADER_f1', 0.0):.1%}")
    # Support from true label distribution (sklearn report may be wrong)
    true_dist = metrics.get('eval_true_label_distribution', metrics.get('true_label_distribution', {}))
    print(f"  Support:   {true_dist.get('SECTION_HEADER', 0)}")

    print("\n⚠️  Critical Errors")
    print(f"  Header→Title confusion count: {int(get_metric('header_title_confusion_count', 0))}")
    print(f"  Header false positive rate:   {get_metric('header_false_positive_rate', 0.0):.1%}")
    print(f"  Title false negative rate:    {get_metric('title_false_negative_rate', 0.0):.1%}")

    print("\n📊 Ingredient & Instruction F1")
    print(f"  Ingredients: {get_report_metric('INGREDIENT_LINE', 'f1-score', 0.0):.1%}")
    print(f"  Instructions: {get_report_metric('INSTRUCTION_STEP', 'f1-score', 0.0):.1%}")

    print("=" * 80 + "\n")


__all__ = [
    "compute_header_title_metrics",
    "compute_title_anchor_accuracy",
    "print_demo_scorecard",
]
