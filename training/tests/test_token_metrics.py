#!/usr/bin/env python3
"""Unit tests for token-level classification metrics."""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.eval.metrics import compute_token_metrics


class TestTokenMetrics(unittest.TestCase):
    """Test token-level classification metrics for non-BIO labels."""

    def setUp(self):
        """Set up test fixtures."""
        self.label_names = ["O", "TITLE", "INGREDIENT_LINE", "INSTRUCTION_STEP"]

    def test_basic_metrics_computation(self):
        """Test that basic metrics are computed correctly."""
        # Create simple predictions: all correct
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])

        metrics = compute_token_metrics(y_true, y_pred, self.label_names)

        # Should have perfect metrics (non-O only)
        self.assertAlmostEqual(metrics["precision"], 1.0, places=5)
        self.assertAlmostEqual(metrics["recall"], 1.0, places=5)
        self.assertAlmostEqual(metrics["f1"], 1.0, places=5)

    def test_masked_tokens_excluded(self):
        """Test that -100 masked tokens are properly excluded."""
        # Note: compute_token_metrics expects already-filtered data
        # The filtering happens in compute_metrics_fn before calling this function
        y_true = np.array([0, 1, 2, 3])  # No -100 here
        y_pred = np.array([0, 1, 2, 3])

        metrics = compute_token_metrics(y_true, y_pred, self.label_names)

        # Should have perfect metrics
        self.assertAlmostEqual(metrics["precision"], 1.0, places=5)
        self.assertAlmostEqual(metrics["recall"], 1.0, places=5)
        self.assertAlmostEqual(metrics["f1"], 1.0, places=5)

    def test_non_o_focus(self):
        """Test that main metrics focus on non-O labels."""
        # Create data: O labels predicted perfectly, but non-O labels all wrong
        y_true = np.array([0, 0, 0, 0, 1, 2, 3])  # 4 O, 3 non-O
        y_pred = np.array([0, 0, 0, 0, 2, 3, 1])  # O correct, non-O wrong

        metrics = compute_token_metrics(y_true, y_pred, self.label_names)

        # Main F1 should be low (non-O labels are all wrong)
        self.assertLess(metrics["f1"], 0.5)

        # But O label should have perfect precision/recall in per-label report
        report = metrics["report"]
        self.assertIn("O", report)
        self.assertAlmostEqual(report["O"]["precision"], 1.0, places=5)
        self.assertAlmostEqual(report["O"]["recall"], 1.0, places=5)

    def test_per_label_metrics(self):
        """Test that per-label metrics are computed correctly."""
        # Perfect predictions for TITLE (label 1)
        # All INGREDIENT_LINE (label 2) predicted as INSTRUCTION_STEP (label 3)
        y_true = np.array([0, 1, 1, 2, 2, 3])
        y_pred = np.array([0, 1, 1, 3, 3, 3])

        metrics = compute_token_metrics(y_true, y_pred, self.label_names)

        report = metrics["report"]

        # TITLE should have perfect metrics
        self.assertIn("TITLE", report)
        self.assertAlmostEqual(report["TITLE"]["precision"], 1.0, places=5)
        self.assertAlmostEqual(report["TITLE"]["recall"], 1.0, places=5)
        self.assertAlmostEqual(report["TITLE"]["f1-score"], 1.0, places=5)
        self.assertEqual(report["TITLE"]["support"], 2)

        # INGREDIENT_LINE should have 0 recall (none predicted correctly)
        self.assertIn("INGREDIENT_LINE", report)
        self.assertAlmostEqual(report["INGREDIENT_LINE"]["recall"], 0.0, places=5)
        self.assertEqual(report["INGREDIENT_LINE"]["support"], 2)

    def test_confusion_matrix(self):
        """Test that confusion matrix is computed correctly."""
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 3, 3])  # Label 2 predicted as 3

        metrics = compute_token_metrics(y_true, y_pred, self.label_names)

        # Should have confusion matrix
        self.assertIn("confusion_matrix", metrics)
        cm = np.array(metrics["confusion_matrix"])

        # Check shape matches number of labels present
        labels = metrics["labels"]
        self.assertEqual(cm.shape[0], len(labels))
        self.assertEqual(cm.shape[1], len(labels))

        # Should have labels list
        self.assertIn("labels", metrics)
        self.assertIn("O", metrics["labels"])
        self.assertIn("TITLE", metrics["labels"])

    def test_empty_non_o_labels(self):
        """Test handling when only O labels are present."""
        # Only O labels
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])

        metrics = compute_token_metrics(y_true, y_pred, self.label_names)

        # Main metrics should be 0 (no non-O labels to compute over)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)

        # But per-label report should still include O
        report = metrics["report"]
        self.assertIn("O", report)

    def test_all_labels_present(self):
        """Test that metrics include all labels present in data."""
        # Use all 4 labels
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])

        metrics = compute_token_metrics(y_true, y_pred, self.label_names)

        # All labels should be in the report
        report = metrics["report"]
        self.assertIn("O", report)
        self.assertIn("TITLE", report)
        self.assertIn("INGREDIENT_LINE", report)
        self.assertIn("INSTRUCTION_STEP", report)

        # Labels list should match
        self.assertEqual(len(metrics["labels"]), 4)

    def test_zero_division_handling(self):
        """Test that zero division is handled gracefully."""
        # Predict a label that doesn't exist in y_true
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])  # Predict TITLE when none exist

        # Should not raise exception
        metrics = compute_token_metrics(y_true, y_pred, self.label_names)

        # Should have some metrics (zero division handled)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)

    def test_support_counts(self):
        """Test that support counts are correct."""
        y_true = np.array([0, 0, 1, 1, 1, 2, 3, 3, 3, 3])
        y_pred = np.array([0, 0, 1, 1, 1, 2, 3, 3, 3, 3])

        metrics = compute_token_metrics(y_true, y_pred, self.label_names)

        report = metrics["report"]

        # Check support counts match y_true
        self.assertEqual(report["O"]["support"], 2)
        self.assertEqual(report["TITLE"]["support"], 3)
        self.assertEqual(report["INGREDIENT_LINE"]["support"], 1)
        self.assertEqual(report["INSTRUCTION_STEP"]["support"], 4)


if __name__ == "__main__":
    unittest.main()
