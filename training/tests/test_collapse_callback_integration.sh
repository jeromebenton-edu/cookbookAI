#!/bin/bash
# Integration test for CollapseDetectionCallback eval_dataset fix
# This script verifies that the callback can find eval_dataset during training

set -euo pipefail

echo "=============================================="
echo "Testing CollapseDetectionCallback eval_dataset discovery"
echo "=============================================="

# Create a minimal test to verify callback finds eval_dataset
python3 << 'PYTHON_TEST'
import sys
from pathlib import Path
from unittest.mock import Mock

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Mock dependencies
sys.modules['seqeval'] = Mock()
sys.modules['seqeval.metrics'] = Mock()
sys.modules['sklearn'] = Mock()
sys.modules['sklearn.metrics'] = Mock()

from training.train_layoutlmv3 import CollapseDetectionCallback

# Create callback
callback = CollapseDetectionCallback(
    label_list=["O", "TITLE", "INGREDIENT_LINE"],
    collapse_threshold=0.9,
    patience=2,
)

# Test 1: Can find eval_dataset from trainer
print("\n[TEST 1] Finding eval_dataset from trainer.eval_dataset...")
mock_trainer = Mock()
mock_dataset = [{"labels": [0, 1, 2]}]
mock_trainer.eval_dataset = mock_dataset

mock_args = Mock()
mock_state = Mock()
mock_state.epoch = 1
mock_control = Mock()
mock_model = None  # Will return early

# This should NOT print warning about missing dataset
import io
import logging
from unittest.mock import patch

# Capture LOG output
log_capture = io.StringIO()
handler = logging.StreamHandler(log_capture)
logging.getLogger("training").addHandler(handler)
logging.getLogger("training").setLevel(logging.WARNING)

callback.on_epoch_end(
    args=mock_args,
    state=mock_state,
    control=mock_control,
    model=mock_model,
    trainer=mock_trainer,  # Pass trainer with eval_dataset
)

log_output = log_capture.getvalue()
if "No eval_dataset found" in log_output:
    print("❌ FAILED: Callback warned about missing eval_dataset when trainer.eval_dataset exists")
    sys.exit(1)
else:
    print("✅ PASSED: Callback did not warn when trainer.eval_dataset exists")

# Test 2: Can find eval_dataset from kwargs
print("\n[TEST 2] Finding eval_dataset from kwargs['eval_dataset']...")
log_capture2 = io.StringIO()
handler2 = logging.StreamHandler(log_capture2)
logging.getLogger("training").handlers = [handler2]

callback.on_epoch_end(
    args=mock_args,
    state=mock_state,
    control=mock_control,
    model=mock_model,
    eval_dataset=mock_dataset,  # Pass directly in kwargs
)

log_output2 = log_capture2.getvalue()
if "No eval_dataset found" in log_output2:
    print("❌ FAILED: Callback warned about missing eval_dataset when passed in kwargs")
    sys.exit(1)
else:
    print("✅ PASSED: Callback did not warn when eval_dataset in kwargs")

# Test 3: Warns when no eval_dataset
print("\n[TEST 3] Warning when no eval_dataset available...")
log_capture3 = io.StringIO()
handler3 = logging.StreamHandler(log_capture3)
logging.getLogger("training").handlers = [handler3]

callback.on_epoch_end(
    args=mock_args,
    state=mock_state,
    control=mock_control,
    model=mock_model,
    # No trainer, no eval_dataset
)

log_output3 = log_capture3.getvalue()
if "No eval_dataset found" not in log_output3:
    print("❌ FAILED: Callback should warn when no eval_dataset available")
    sys.exit(1)
else:
    print("✅ PASSED: Callback correctly warns when no eval_dataset")

print("\n" + "="*50)
print("✅ ALL TESTS PASSED!")
print("="*50)

PYTHON_TEST

echo ""
echo "✅ Integration test completed successfully!"
