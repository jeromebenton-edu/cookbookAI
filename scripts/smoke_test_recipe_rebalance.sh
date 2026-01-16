#!/bin/bash
# Smoke test for recipe dataset rebalancing implementation
# Runs quick verification of all fixes before full training

set -euo pipefail

echo "========================================================================"
echo "SMOKE TEST: Recipe Dataset Rebalancing"
echo "========================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

FAILURES=0

# Function to print test result
print_result() {
    local test_name="$1"
    local result="$2"
    if [ "$result" = "PASS" ]; then
        echo -e "${GREEN}✓${NC} $test_name"
    else
        echo -e "${RED}✗${NC} $test_name"
        FAILURES=$((FAILURES + 1))
    fi
}

echo "Step 1: Check dataset split names"
echo "------------------------------------"
python3 << 'PYTHON_TEST_1'
from datasets import load_from_disk
import sys

try:
    ds = load_from_disk("data/datasets/boston_layoutlmv3_recipe_only/dataset_dict")
    splits = list(ds.keys())

    # Check for 'validation' (not 'val')
    if "validation" not in splits:
        print(f"FAIL: Expected 'validation' split, found: {splits}")
        sys.exit(1)

    # Check validation is not empty
    if len(ds["validation"]) == 0:
        print("FAIL: Validation split is empty")
        sys.exit(1)

    print(f"PASS: Found splits: {splits}")
    print(f"PASS: Validation size: {len(ds['validation'])} examples")

except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
PYTHON_TEST_1

print_result "Dataset has 'validation' split" $?
echo ""

echo "Step 2: Check ingredient coverage in selection"
echo "------------------------------------------------"
python3 << 'PYTHON_TEST_2'
import json
import sys

try:
    stats = json.load(open("data/datasets/boston_layoutlmv3_recipe_only/stats.json"))
    dist = stats.get("label_distribution", {})

    # Check INGREDIENT_LINE exists and is > 0
    ing = dist.get("INGREDIENT_LINE", {})
    ing_count = ing.get("count", 0)
    ing_percent = ing.get("percent", 0) * 100

    if ing_count == 0:
        print(f"FAIL: INGREDIENT_LINE has 0 tokens")
        sys.exit(1)

    if ing_percent < 3.0:
        print(f"WARN: INGREDIENT_LINE only {ing_percent:.2f}% (expected >5%)")

    print(f"PASS: INGREDIENT_LINE: {ing_count} tokens ({ing_percent:.2f}%)")

    # Check INSTRUCTION_STEP still exists
    inst = dist.get("INSTRUCTION_STEP", {})
    inst_count = inst.get("count", 0)
    inst_percent = inst.get("percent", 0) * 100

    if inst_count == 0:
        print(f"FAIL: INSTRUCTION_STEP has 0 tokens")
        sys.exit(1)

    print(f"PASS: INSTRUCTION_STEP: {inst_count} tokens ({inst_percent:.2f}%)")

except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
PYTHON_TEST_2

print_result "Label distribution looks healthy" $?
echo ""

echo "Step 3: Run sanity checks"
echo "--------------------------"
python scripts/sanity_check_encoded_labels.py \
  --dataset-dir data/datasets/boston_layoutlmv3_recipe_only/dataset_dict \
  --num-samples 10 \
  --splits train validation 2>&1 | head -50

print_result "Sanity checks completed" $?
echo ""

echo "Step 4: Quick 3-epoch training smoke test"
echo "------------------------------------------"
echo "This will verify:"
echo "  - Trainer can find validation split"
echo "  - CollapseDetectionCallback runs"
echo "  - INGREDIENT_LINE F1 is no longer 0.0"
echo ""

# Run 3-epoch smoke test
python -m training.train_layoutlmv3 \
  --stage full \
  --use_recipe_only \
  --num_train_epochs_stageB 3 \
  --batch_size 2 \
  --eval_batch_size 2 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --gradient_accumulation_steps 2 \
  --logging_steps 10 \
  --eval_steps 100 \
  --save_steps 200 \
  --save_total_limit 1 \
  --output_dir models/layoutlmv3_boston_smoke_test \
  --seed 42 \
  --run_sanity_checks

TRAIN_EXIT=$?
print_result "3-epoch smoke test completed" $TRAIN_EXIT
echo ""

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "Step 5: Check training results"
    echo "-------------------------------"

    # Check if trainer_state.json exists
    if [ -f "models/layoutlmv3_boston_smoke_test/trainer_state.json" ]; then
        echo "✓ Training completed successfully"

        # Extract final metrics if available
        python3 << 'PYTHON_TEST_5'
import json
import sys

try:
    state = json.load(open("models/layoutlmv3_boston_smoke_test/trainer_state.json"))

    # Get best metrics
    best_metric = state.get("best_metric")
    log_history = state.get("log_history", [])

    # Find last eval metrics
    eval_metrics = [log for log in log_history if "eval_f1" in log]

    if eval_metrics:
        last_eval = eval_metrics[-1]

        print(f"\nFinal Evaluation Metrics (Epoch {last_eval.get('epoch', '?')}):")
        print(f"  Overall F1:        {last_eval.get('eval_f1', 0):.4f}")
        print(f"  INGREDIENT_LINE:   {last_eval.get('eval_f1_INGREDIENT_LINE', 0):.4f}")
        print(f"  INSTRUCTION_STEP:  {last_eval.get('eval_f1_INSTRUCTION_STEP', 0):.4f}")
        print(f"  TITLE:             {last_eval.get('eval_f1_TITLE', 0):.4f}")

        # Check if INGREDIENT_LINE F1 is no longer 0
        ing_f1 = last_eval.get('eval_f1_INGREDIENT_LINE', 0)
        if ing_f1 == 0.0:
            print(f"\n⚠️  WARNING: INGREDIENT_LINE F1 is still 0.0!")
            sys.exit(1)
        else:
            print(f"\n✓ INGREDIENT_LINE F1 > 0 (no longer stuck!)")
    else:
        print("No evaluation metrics found in log history")
        sys.exit(1)

except Exception as e:
    print(f"Error checking metrics: {e}")
    sys.exit(1)
PYTHON_TEST_5

        print_result "INGREDIENT_LINE F1 > 0.0" $?
    else
        echo "✗ trainer_state.json not found"
        FAILURES=$((FAILURES + 1))
    fi
fi

echo ""
echo "========================================================================"
echo "SMOKE TEST SUMMARY"
echo "========================================================================"

if [ $FAILURES -eq 0 ]; then
    echo -e "${GREEN}✓ ALL CHECKS PASSED${NC}"
    echo ""
    echo "The implementation is working correctly!"
    echo ""
    echo "Next steps:"
    echo "  1. Review the 3-epoch results above"
    echo "  2. If INGREDIENT_LINE F1 > 0.0, proceed with full training:"
    echo ""
    echo "     python -m training.train_layoutlmv3 \\"
    echo "       --stage full \\"
    echo "       --use_recipe_only \\"
    echo "       --num_train_epochs_stageB 10 \\"
    echo "       --batch_size 4 \\"
    echo "       --eval_batch_size 4 \\"
    echo "       --output_dir models/layoutlmv3_boston_recipe_only_v3"
    echo ""
    exit 0
else
    echo -e "${RED}✗ $FAILURES CHECKS FAILED${NC}"
    echo ""
    echo "Please review the errors above and fix before proceeding."
    exit 1
fi
