# LayoutLMv3 Training Plan

## Current Status

✅ **Coordinate Scaling**: Already implemented in [SectionOverlayViewer.tsx:103-107](frontend/components/ai/SectionOverlayViewer.tsx#L103-L107)
✅ **Label Hygiene**: Fixed mock mode heuristic to skip page headers (top 50px) and require mixed case
✅ **Training Infrastructure**: Existing [train_layoutlmv3.py](training/train_layoutlmv3.py) with full pipeline
✅ **Training Data**: 614 labeled examples in `data/datasets/boston_layoutlmv3_full/dataset_dict`

## What We Need to Do

### 1. Train a Real LayoutLMv3 Model

**Why**: Mock mode heuristics are insufficient for production-quality extraction. The current heuristic:
- Only tags words with digits as ingredients
- Only tags specific action verbs as instructions
- Misses most of the actual recipe content

**Command**:
```bash
python training/train_layoutlmv3.py \
  --dataset data/datasets/boston_layoutlmv3_full/dataset_dict \
  --output_dir models/layoutlmv3_boston_v1 \
  --num_train_epochs 10 \
  --per_device_train_batch_size 4 \
  --learning_rate 5e-5 \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --logging_steps 50
```

**Requirements**:
- GPU recommended (training on CPU will be very slow)
- ~16GB GPU memory for batch size 4
- ~2-4 hours training time on GPU

### 2. Update Fixture Generator to Use Trained Model

**Current**: [generate_demo_fixtures.py:181](tools/generate_demo_fixtures.py#L181) uses mock mode
**Target**: Load and use the trained checkpoint

**Changes needed in `generate_demo_fixtures.py`**:

```python
# Load real model instead of mock mode
if model_checkpoint and Path(model_checkpoint).exists():
    processor = LayoutLMv3Processor.from_pretrained(model_checkpoint)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_checkpoint)
    model.eval()
    use_real_model = True
else:
    print("[WARN] No model checkpoint, using mock mode")
    use_real_model = False
```

Then in token labeling:
```python
if use_real_model:
    # Run actual model inference
    predictions = run_model_inference(image, ocr_tokens, model, processor)
    for tok, pred in zip(tokens, predictions):
        tok["label"] = pred["label"]
        tok["conf"] = pred["confidence"]
else:
    # Current mock mode heuristic
    apply_mock_labels(tokens)
```

### 3. Regenerate Demo Fixtures with Real Model

**Command**:
```bash
python tools/generate_demo_fixtures.py \
  --examples-config tools/demo_examples_config.json \
  --model-checkpoint models/layoutlmv3_boston_v1/checkpoint-best \
  --output-root frontend/src/demo_examples
```

**Expected improvements**:
- Title detection: Find actual recipe title, not page headers
- Ingredients: Detect full ingredient lines, not just words with numbers
- Instructions: Detect full instruction steps, not just action verbs
- Higher confidence scores based on model certainty

### 4. Verify Fixture Quality

**Validation script**: [tools/validate_demo_fixture.py](tools/validate_demo_fixture.py)

```bash
python tools/validate_demo_fixture.py frontend/src/demo_examples/example_01
python tools/validate_demo_fixture.py frontend/src/demo_examples/example_02
```

**Expected output**:
- ✓ Schema validation passes
- ✓ Title matches expected recipe (not page header)
- ✓ Ingredients count > 5 (not just 3)
- ✓ Instructions count > 0 (currently 0)

## Timeline

1. **Train model** (2-4 hours on GPU, 24+ hours on CPU)
2. **Update fixture generator** (30 min coding)
3. **Regenerate fixtures** (< 5 min)
4. **Validate and test** (15 min)

## Success Metrics

### Before (Mock Mode)
- example_01: 3 ingredients, 0 instructions
- example_02: 0 ingredients, 0 instructions
- Title: Generic page headers tagged as TITLE

### After (Real Model)
- example_01: 8+ ingredients, 4+ instructions
- example_02: 6+ ingredients, 5+ instructions
- Title: Actual recipe title correctly identified
- Section overlays: Accurate bounding boxes for title/ingredients/instructions
- Token overlays: Real model predictions with meaningful confidence scores

## Next Steps

**Option A: Train Now (Recommended if GPU available)**
```bash
cd /home/jerome/projects/cookbookAI
python training/train_layoutlmv3.py \
  --dataset data/datasets/boston_layoutlmv3_full/dataset_dict \
  --output_dir models/layoutlmv3_boston_v1 \
  --num_train_epochs 10
```

**Option B: Use Existing Checkpoint (If available)**
Check if there's already a trained model:
```bash
find models/ -name "pytorch_model.bin" -o -name "model.safetensors"
```

**Option C: Deploy with Mock Mode (Current state)**
The demo works with mock mode for basic visualization, but predictions are not production-quality.

## Additional Improvements (Optional)

1. **Two-stage training**: Train on high-confidence subset first, then full dataset
2. **Model ensembling**: Train multiple checkpoints and ensemble predictions
3. **Active learning**: Identify low-confidence predictions for manual review
4. **Cross-validation**: Ensure model generalizes beyond Boston Cooking School cookbook
