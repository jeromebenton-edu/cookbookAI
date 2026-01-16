#!/usr/bin/env python3
"""Unit tests for CollapseDetectionCallback."""

import unittest
from unittest.mock import Mock, MagicMock, patch
import torch
import numpy as np
from collections import Counter

# Mock missing modules before importing
import sys
from pathlib import Path

# Don't mock sklearn/seqeval - they are now required dependencies
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestCollapseDetectionCallback(unittest.TestCase):
    """Test CollapseDetectionCallback finds eval_dataset correctly."""

    def setUp(self):
        """Set up test fixtures."""
        # Import after path is set
        from training.train_layoutlmv3 import CollapseDetectionCallback

        self.label_list = ["O", "TITLE", "INGREDIENT_LINE", "INSTRUCTION_STEP"]
        # Create a mock eval dataset
        self.eval_dataset = self.create_mock_dataset(10)
        self.callback = CollapseDetectionCallback(
            label_list=self.label_list,
            eval_dataset=self.eval_dataset,  # Pass eval_dataset explicitly
            collapse_threshold=0.9,
            patience=2,
            stage_name="test"
        )

    def create_mock_dataset(self, num_samples=10):
        """Create a mock dataset for testing."""
        mock_dataset = []
        for i in range(num_samples):
            mock_dataset.append({
                "input_ids": torch.randint(0, 1000, (512,)),
                "attention_mask": torch.ones(512),
                "bbox": torch.zeros((512, 4)),
                "pixel_values": torch.randn(3, 224, 224),
                "labels": torch.randint(0, len(self.label_list), (512,)),
            })
        return mock_dataset

    def create_mock_model(self):
        """Create a mock model that returns logits."""
        model = Mock()
        model.device = torch.device("cpu")
        model.eval = Mock()

        # Mock the model output
        def mock_forward(**kwargs):
            batch_size = kwargs["input_ids"].shape[0]
            seq_len = kwargs["input_ids"].shape[1]
            num_labels = len(self.label_list)

            # Create mock logits (batch_size, seq_len, num_labels)
            logits = torch.randn(batch_size, seq_len, num_labels)

            # Make O (label 0) have higher logits to simulate collapse
            logits[:, :, 0] += 2.0

            output = Mock()
            output.logits = logits
            return output

        model.side_effect = mock_forward
        return model

    def create_mock_trainer(self, eval_dataset):
        """Create a mock trainer with eval_dataset."""
        trainer = Mock()
        trainer.eval_dataset = eval_dataset

        # Mock get_eval_dataloader
        def mock_get_eval_dataloader():
            from torch.utils.data import DataLoader
            return DataLoader(eval_dataset, batch_size=2)

        trainer.get_eval_dataloader = mock_get_eval_dataloader
        return trainer

    def test_uses_explicitly_passed_eval_dataset(self):
        """Test that callback uses eval_dataset passed at construction time."""
        # Create a new callback with explicit eval_dataset
        eval_dataset = self.create_mock_dataset(10)
        from training.train_layoutlmv3 import CollapseDetectionCallback
        callback = CollapseDetectionCallback(
            label_list=self.label_list,
            eval_dataset=eval_dataset,  # Pass explicitly
            collapse_threshold=0.9,
            patience=2,
            stage_name="test"
        )

        model = self.create_mock_model()

        # Mock args and state
        args = Mock()
        args.per_device_eval_batch_size = 2

        state = Mock()
        state.epoch = 1

        control = Mock()

        # Call the callback WITHOUT passing trainer or eval_dataset in kwargs
        with patch('training.train_layoutlmv3.collate_fn') as mock_collate_fn:
            mock_collate_fn.return_value = {}

            # This should NOT warn about missing eval_dataset
            with patch('training.train_layoutlmv3.LOG') as mock_log:
                callback.on_epoch_end(
                    args=args,
                    state=state,
                    control=control,
                    model=model,
                    # No trainer, no eval_dataset in kwargs - should use stored one
                )

                # Check that warning was NOT called
                warning_calls = [call for call in mock_log.warning.call_args_list
                                if "No eval_dataset found" in str(call)]
                self.assertEqual(len(warning_calls), 0,
                               "Should not warn about missing eval_dataset when passed at construction")

    def test_finds_eval_dataset_from_trainer_attribute(self):
        """Test that callback finds eval_dataset from trainer.eval_dataset as fallback."""
        eval_dataset = self.create_mock_dataset(10)
        trainer = self.create_mock_trainer(eval_dataset)

        # Create callback WITHOUT eval_dataset (to test fallback)
        from training.train_layoutlmv3 import CollapseDetectionCallback
        callback = CollapseDetectionCallback(
            label_list=self.label_list,
            eval_dataset=None,  # Not passed - should fall back to trainer
            collapse_threshold=0.9,
            patience=2,
            stage_name="test"
        )

        model = self.create_mock_model()

        # Mock args and state
        args = Mock()
        args.per_device_eval_batch_size = 2

        state = Mock()
        state.epoch = 1

        control = Mock()

        # Call the callback with trainer in kwargs
        with patch('training.train_layoutlmv3.collate_fn') as mock_collate_fn:
            mock_collate_fn.return_value = {}

            # This should NOT warn about missing eval_dataset
            with patch('training.train_layoutlmv3.LOG') as mock_log:
                callback.on_epoch_end(
                    args=args,
                    state=state,
                    control=control,
                    model=model,
                    trainer=trainer,  # Pass trainer in kwargs
                )

                # Check that warning was NOT called
                warning_calls = [call for call in mock_log.warning.call_args_list
                                if "No eval_dataset found" in str(call)]
                self.assertEqual(len(warning_calls), 0,
                               "Should not warn about missing eval_dataset when trainer.eval_dataset exists")

    def test_finds_eval_dataset_from_kwargs(self):
        """Test that callback finds eval_dataset from kwargs['eval_dataset']."""
        eval_dataset = self.create_mock_dataset(10)
        model = self.create_mock_model()

        # Mock args and state
        args = Mock()
        args.per_device_eval_batch_size = 2

        state = Mock()
        state.epoch = 1

        control = Mock()

        # Call the callback with eval_dataset in kwargs (no trainer)
        with patch('training.train_layoutlmv3.collate_fn') as mock_collate_fn:
            mock_collate_fn.return_value = {}

            with patch('training.train_layoutlmv3.LOG') as mock_log:
                self.callback.on_epoch_end(
                    args=args,
                    state=state,
                    control=control,
                    model=model,
                    eval_dataset=eval_dataset,  # Pass eval_dataset directly in kwargs
                )

                # Check that warning was NOT called
                warning_calls = [call for call in mock_log.warning.call_args_list
                                if "No eval_dataset found" in str(call)]
                self.assertEqual(len(warning_calls), 0,
                               "Should not warn about missing eval_dataset when passed in kwargs")

    def test_finds_eval_dataset_from_dataloader(self):
        """Test that callback finds eval_dataset from trainer.get_eval_dataloader()."""
        eval_dataset = self.create_mock_dataset(10)
        trainer = Mock()

        # Don't set trainer.eval_dataset (to test fallback)
        del trainer.eval_dataset

        # Mock get_eval_dataloader to return a dataloader with dataset
        def mock_get_eval_dataloader():
            dataloader = Mock()
            dataloader.dataset = eval_dataset
            return dataloader

        trainer.get_eval_dataloader = mock_get_eval_dataloader

        model = self.create_mock_model()

        # Mock args and state
        args = Mock()
        args.per_device_eval_batch_size = 2

        state = Mock()
        state.epoch = 1

        control = Mock()

        # Call the callback with trainer (but no eval_dataset attribute)
        with patch('training.train_layoutlmv3.collate_fn') as mock_collate_fn:
            mock_collate_fn.return_value = {}

            with patch('training.train_layoutlmv3.LOG') as mock_log:
                self.callback.on_epoch_end(
                    args=args,
                    state=state,
                    control=control,
                    model=model,
                    trainer=trainer,  # Pass trainer without eval_dataset attribute
                )

                # Check that warning was NOT called
                warning_calls = [call for call in mock_log.warning.call_args_list
                                if "No eval_dataset found" in str(call)]
                self.assertEqual(len(warning_calls), 0,
                               "Should not warn when eval_dataset available via get_eval_dataloader")

    def test_warns_when_no_eval_dataset_available(self):
        """Test that callback warns when no eval_dataset is available."""
        # Create callback WITHOUT eval_dataset to test warning
        from training.train_layoutlmv3 import CollapseDetectionCallback
        callback_no_data = CollapseDetectionCallback(
            label_list=self.label_list,
            eval_dataset=None,  # Explicitly no eval_dataset
            collapse_threshold=0.9,
            patience=2,
            stage_name="test"
        )

        model = self.create_mock_model()

        # Mock args and state
        args = Mock()
        args.per_device_eval_batch_size = 2

        state = Mock()
        state.epoch = 1

        control = Mock()

        # Call the callback with no eval_dataset anywhere
        with patch('training.train_layoutlmv3.LOG') as mock_log:
            callback_no_data.on_epoch_end(
                args=args,
                state=state,
                control=control,
                model=model,
                # No trainer, no eval_dataset in kwargs
            )

            # Check that warning WAS called
            warning_calls = [call for call in mock_log.warning.call_args_list
                            if "No eval_dataset found" in str(call)]
            self.assertGreater(len(warning_calls), 0,
                             "Should warn about missing eval_dataset when none available")

    def test_detects_collapse(self):
        """Test that callback detects label collapse correctly."""
        # Create dataset where all labels are O (label 0)
        eval_dataset = []
        for i in range(5):
            eval_dataset.append({
                "input_ids": torch.randint(0, 1000, (512,)),
                "attention_mask": torch.ones(512),
                "bbox": torch.zeros((512, 4)),
                "pixel_values": torch.randn(3, 224, 224),
                "labels": torch.zeros(512, dtype=torch.long),  # All O
            })

        trainer = self.create_mock_trainer(eval_dataset)

        # Create model that always predicts O
        model = Mock()
        model.device = torch.device("cpu")
        model.eval = Mock()

        def mock_forward(**kwargs):
            batch_size = kwargs["input_ids"].shape[0]
            seq_len = kwargs["input_ids"].shape[1]
            num_labels = len(self.label_list)

            # Create logits that strongly favor O (label 0)
            logits = torch.randn(batch_size, seq_len, num_labels)
            logits[:, :, 0] += 10.0  # Very high logits for O

            output = Mock()
            output.logits = logits
            return output

        model.side_effect = mock_forward

        # Mock args and state
        args = Mock()
        args.per_device_eval_batch_size = 2

        state = Mock()
        state.epoch = 1

        control = Mock()

        # Call the callback
        with patch('training.train_layoutlmv3.collate_fn') as mock_collate_fn:
            # Simple collate function that returns batch as-is
            mock_collate_fn.side_effect = lambda batch: {
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
                "bbox": torch.stack([b["bbox"] for b in batch]),
                "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
                "labels": torch.stack([b["labels"] for b in batch]),
            }

            with patch('training.train_layoutlmv3.LOG') as mock_log:
                self.callback.on_epoch_end(
                    args=args,
                    state=state,
                    control=control,
                    model=model,
                    trainer=trainer,
                )

                # Check that collapse warning WAS called
                warning_calls = [call for call in mock_log.warning.call_args_list
                                if "COLLAPSE WARNING" in str(call)]
                self.assertGreater(len(warning_calls), 0,
                                 "Should warn about label collapse when O dominates")


if __name__ == "__main__":
    unittest.main()
