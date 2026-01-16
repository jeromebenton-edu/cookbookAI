#!/bin/bash
# V3 Training Pipeline - Command Reference
# Complete workflow from relabeling to deployment

set -e  # Exit on error

echo "================================"
echo "V3 TRAINING PIPELINE"
echo "================================"

# Configuration
DATASET_ROOT="data/datasets/boston_layoutlmv3_v3"
MODEL_OUTPUT="models/layoutlmv3_v3_production"
DEMO_PAGES="79,96,100,105,110,115,120"

echo ""
echo "Step 1: Relabel dataset (v2 → v3)"
echo "================================"
python tools/relabel_suggest.py \
  --input data/datasets/boston_layoutlmv3_full/merged_pages_with_heuristics.jsonl \
  --output data/processed/v3_headers_titles/boston_v3_suggested.jsonl \
  --stats data/processed/v3_headers_titles/relabel_stats.json

echo ""
echo "✓ Relabeling complete. Check stats:"
cat data/processed/v3_headers_titles/relabel_stats.json | python -m json.tool | head -30

echo ""
echo "Step 2: Build HuggingFace dataset"
echo "================================"
python tools/build_v3_dataset.py \
  --input data/processed/v3_headers_titles/boston_v3_suggested.jsonl \
  --output $DATASET_ROOT \
  --demo-pages "$DEMO_PAGES"

echo ""
echo "✓ Dataset built. Check manifest:"
cat $DATASET_ROOT/dataset_manifest.json | python -m json.tool | head -40

echo ""
echo "Step 3: Train v3 model"
echo "================================"
echo "IMPORTANT: This will take 2-4 hours on GPU"
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

python tools/train_v3_model.py \
  --dataset $DATASET_ROOT/dataset_dict \
  --output $MODEL_OUTPUT \
  --epochs 20 \
  --batch-size 4 \
  --learning-rate 5e-5

echo ""
echo "✓ Training complete! Check report:"
cat $MODEL_OUTPUT/training_report.json | python -m json.tool | head -50

echo ""
echo "Step 4: Generate demo fixtures"
echo "================================"
python tools/generate_demo_fixtures.py \
  --examples-config tools/demo_examples_config.json \
  --model-checkpoint $MODEL_OUTPUT \
  --output-root frontend/src/demo_examples

echo ""
echo "✓ Fixtures generated. Verify titles:"
echo "Example 01:"
jq '.extractedRecipe.title' frontend/src/demo_examples/example_01/prediction.json
echo "Example 02:"
jq '.extractedRecipe.title' frontend/src/demo_examples/example_02/prediction.json

echo ""
echo "Step 5: Verify demo page"
echo "================================"
echo "Starting Next.js dev server..."
cd frontend
npm run dev &
DEV_PID=$!
sleep 10

echo ""
echo "✓ Dev server running at http://localhost:3001/demo"
echo "  Check that:"
echo "  - Product mode shows recipe titles (NOT page headers)"
echo "  - Inspector → Tokens shows RECIPE_TITLE labels"
echo "  - Section overlays anchor to recipe title"
echo ""
echo "Press Ctrl+C to stop dev server"
wait $DEV_PID

echo ""
echo "================================"
echo "V3 PIPELINE COMPLETE!"
echo "================================"
echo "Trained model: $MODEL_OUTPUT"
echo "Demo fixtures: frontend/src/demo_examples/"
echo "Documentation: TRAINING_V3_README.md"
