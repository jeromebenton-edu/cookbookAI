# LayoutLMv3 v3 Training - Header-Aware Recipe Extraction

This document describes the complete v3 training redesign to fix the header/title confusion issue.

## Problem Statement

Existing LayoutLMv3 checkpoints fail for demo use case:
1. `layoutlmv3_boston_demo_250p_v2/layoutlmv3_boston_final` - predicts "SERVINGS" everywhere (model collapse)
2. `layoutlmv3_boston_stageB_full` - confuses page headers with recipe titles (e.g., "BOSTON COOKING-SCHOOL COOK BOOK" labeled as TITLE)

**Root cause**: Training labels treated all headers as generic "TITLE", so the model learned to label running headers as titles.

## Solution Overview

### New Label Taxonomy (v3)

Split `TITLE` into three granular labels:

- **PAGE_HEADER**: Book titles, running headers, page numbers ("BOSTON COOKING-SCHOOL COOK BOOK", "96")
- **SECTION_HEADER**: Category headings ("BISCUITS", "BREAKFAST CAKES", "SOUPS")
- **RECIPE_TITLE**: Actual recipe titles ("Waffles", "Bread Griddle Cakes")

Full label set:
```
PAGE_HEADER
SECTION_HEADER
RECIPE_TITLE
INGREDIENT_LINE
INSTRUCTION_STEP
TIME
TEMP
SERVINGS
NOTE
O
```

## Files Changed

### Core Configuration

| File | Purpose |
|------|---------|
| `ml/config/labels.py` | **Canonical label taxonomy** - single source of truth for all label mappings |
| `training/datasets/labels.py` | Legacy wrapper (imports from canonical config) |
| `training/labeling/labels.py` | Legacy wrapper (imports from canonical config) |

### Data Processing

| File | Purpose |
|------|---------|
| `tools/relabel_suggest.py` | Heuristic relabeling script to suggest v3 labels from v2 annotations |
| `tools/build_v3_dataset.py` | Build HuggingFace dataset with train/val/test/demo_eval splits |

### Training & Evaluation

| File | Purpose |
|------|---------|
| `tools/train_v3_model.py` | Main training script with v3 taxonomy and anti-collapse checks |
| `training/eval/header_metrics.py` | Header-specific evaluation metrics (title anchor accuracy, confusion rates) |

### Inference

| File | Purpose |
|------|---------|
| `tools/generate_demo_fixtures.py` | Updated to load id2label from model config and handle v3 labels |

## Usage Guide

### Step 1: Relabel Dataset

Apply heuristics to suggest v3 labels from existing v2 annotations:

```bash
python tools/relabel_suggest.py \
  --input data/datasets/boston_layoutlmv3_full/merged_pages_with_heuristics.jsonl \
  --output data/processed/v3_headers_titles/boston_v3_suggested.jsonl \
  --stats data/processed/v3_headers_titles/relabel_stats.json
```

**Output**:
- `boston_v3_suggested.jsonl`: Dataset with suggested v3 labels
- `relabel_stats.json`: Statistics on label changes

**Heuristics applied**:
- **PAGE_HEADER**: Top 10% of page + common header phrases + page numbers
- **SECTION_HEADER**: All-caps lines in 10-25% Y range + category keywords
- **RECIPE_TITLE**: Mixed-case text near ingredient clusters (15-50% Y range)

**Results** (from Boston Cooking School cookbook):
- 623 pages processed
- 4,420 TITLE tokens relabeled:
  - 2,455 → PAGE_HEADER (55.5%)
  - 23 → SECTION_HEADER (0.5%)
  - 1,942 → RECIPE_TITLE (44%)

### Step 2: Build v3 Dataset

Create HuggingFace dataset with proper splits:

```bash
python tools/build_v3_dataset.py \
  --input data/processed/v3_headers_titles/boston_v3_suggested.jsonl \
  --output data/datasets/boston_layoutlmv3_v3 \
  --demo-pages "79,96,100,105,110,115,120"
```

**Output**:
- `dataset_dict/` - HuggingFace dataset with 4 splits:
  - `train`: 463 pages
  - `validation`: 92 pages
  - `test`: 61 pages
  - `demo_eval`: 7 pages (fixed demo pages)
- `dataset_manifest.json` - Metadata with label counts and version info

**Demo eval set**: Fixed set of pages representative of demo use case, including pages with strong headers and multiple recipes per page.

### Step 3: Train v3 Model

Train LayoutLMv3 with v3 taxonomy:

```bash
python tools/train_v3_model.py \
  --dataset data/datasets/boston_layoutlmv3_v3/dataset_dict \
  --output models/layoutlmv3_v3_headers \
  --epochs 15 \
  --batch-size 4 \
  --learning-rate 5e-5
```

**Training features**:
- **Anti-collapse detection**: Stops training if >75% of predictions are same class
- **Header-aware metrics**: Precision/Recall for each header/title type
- **Title anchor accuracy**: Measures bbox IoU overlap for recipe titles
- **Demo eval reporting**: Separate metrics on fixed demo pages

**Output**:
- `models/layoutlmv3_v3_headers/` - Final checkpoint with:
  - `config.json` - Contains `id2label` and `label2id` for v3 taxonomy
  - `model.safetensors` - Trained weights
  - `processor/` - LayoutLMv3 processor config
- `training_report.json` - Full metrics on all splits
- `demo_eval_report.json` - Detailed demo scorecard

### Step 4: Generate Demo Fixtures

Generate fixtures using new v3 model:

```bash
python tools/generate_demo_fixtures.py \
  --examples-config tools/demo_examples_config.json \
  --model-checkpoint models/layoutlmv3_v3_headers \
  --output-root frontend/src/demo_examples
```

**Key updates**:
- Loads `id2label` from `model.config.json` (supports both v2 and v3)
- Maps `RECIPE_TITLE` → "TITLE_LINE" for recipe extraction
- Ignores `PAGE_HEADER` and `SECTION_HEADER` in recipe extraction
- Generates fixtures with correct title anchoring

## Evaluation Metrics

### Demo Scorecard

The training script outputs a comprehensive scorecard focusing on demo-critical metrics:

```
DEMO SCORECARD - Header-Aware Evaluation
================================================================================
📍 Title Anchor Accuracy
  Anchor Accuracy (IoU >= 0.3): 95.2%
  Pages with correct anchor: 20 / 21

🎯 Recipe Title Performance
  Precision: 94.3%
  Recall:    92.1%
  F1:        93.2%

📄 Page Header Performance
  Precision: 98.7%
  Recall:    96.4%
  F1:        97.5%

🔀 Section Header Performance
  Precision: 85.0%
  Recall:    80.0%
  F1:        82.4%

⚠️  Critical Errors
  Header→Title confusion count: 12
  Header false positive rate:   0.5%
  Title false negative rate:    1.2%

📊 Ingredient & Instruction F1
  Ingredients: 89.3%
  Instructions: 87.1%
================================================================================
```

### Key Metrics Explained

1. **Title Anchor Accuracy**: % of pages where predicted RECIPE_TITLE bbox overlaps ground truth with IoU >= 0.3
   - **Critical for demo** - ensures section overlays anchor to correct title

2. **Header False Positive Rate**: How often PAGE_HEADER is predicted as RECIPE_TITLE
   - **Directly addresses the original bug**

3. **Header→Title Confusion Count**: Total tokens where headers confused with titles
   - Should be < 20 for good demo performance

## Integration with Demo Page

### Frontend Changes

The demo page already handles both v2 and v3 label formats:

```typescript
// demo/page.tsx - Maps labels to display
const recipeForDisplay = useMemo(() => {
  const realData = REAL_TOKEN_DATA[currentExample.id];
  if (realData) {
    // NEW v3 format: extractedRecipe already uses RECIPE_TITLE (not headers)
    return {
      ...realData.extractedRecipe,
      is_recipe_page: true,
      recipe_confidence: realData.extractedRecipe.confidence.overall,
    };
  }
  return currentExample.prediction; // Old format
}, [currentExample]);
```

### Section Overlay Anchoring

Section overlays use `RECIPE_TITLE` for anchoring (not `PAGE_HEADER`):

```typescript
// lib/overlays/sectionOverlays.ts
const titleTokens = tokens.filter(t =>
  t.label === "RECIPE_TITLE" ||  // v3
  t.label === "TITLE"             // v2 fallback
);
```

## Training Best Practices

### Class Imbalance

The v3 dataset has natural class imbalance:
- `O`: ~101k tokens (67%)
- `INGREDIENT_LINE`: ~24k tokens (16%)
- `INSTRUCTION_STEP`: ~19k tokens (13%)
- `PAGE_HEADER`: ~2.5k tokens (1.6%)
- `RECIPE_TITLE`: ~1.9k tokens (1.3%)

**Handled by**:
- Optimizing for `title_anchor_accuracy` metric (not raw accuracy)
- Collapse detection callback
- Per-class F1 reporting

### Preventing Collapse

The `CollapseDetectionCallback` monitors predictions during validation:
- If >75% predictions are same class → stop training
- Prevents "SERVINGS everywhere" failures
- Logs warning with class distribution

### Hyperparameters

Recommended settings (from `train_v3_model.py`):
```python
epochs = 15
batch_size = 4
learning_rate = 5e-5
warmup_ratio = 0.1
weight_decay = 0.01
max_length = 512
```

## Verification Checklist

Before deploying a new v3 model:

- [ ] Training completed without collapse warning
- [ ] Demo eval title anchor accuracy >= 85%
- [ ] Header→Title confusion count < 50 tokens
- [ ] Recipe title F1 >= 85%
- [ ] Ingredient F1 >= 80%
- [ ] Instruction F1 >= 75%
- [ ] Generated fixtures show correct titles (not page headers)
- [ ] Section overlays anchor to recipe title (not headers)

## Troubleshooting

### Model predicts PAGE_HEADER everywhere

**Cause**: Class imbalance + insufficient training
**Fix**:
1. Check relabeling stats - ensure ~45% RECIPE_TITLE, ~55% PAGE_HEADER
2. Increase training epochs to 20-25
3. Try focal loss or class weights

### Title anchor accuracy low (<70%)

**Cause**: Model not learning spatial relationships
**Fix**:
1. Check bbox normalization in training data
2. Verify OCR quality on demo pages
3. Add more demo-like pages to training set

### High header→title confusion

**Cause**: Heuristic relabeling misclassified headers
**Fix**:
1. Manual review of top-50 confused examples
2. Refine heuristics in `relabel_suggest.py`:
   - Adjust Y-position thresholds
   - Add more header phrase patterns
3. Re-run relabeling and rebuild dataset

## Next Steps

1. **Train production model**:
   ```bash
   python tools/train_v3_model.py \
     --dataset data/datasets/boston_layoutlmv3_v3/dataset_dict \
     --output models/layoutlmv3_v3_production \
     --epochs 20
   ```

2. **Evaluate on demo pages**:
   ```bash
   python tools/generate_demo_fixtures.py \
     --model-checkpoint models/layoutlmv3_v3_production \
     --examples-config tools/demo_examples_config.json
   ```

3. **Deploy to inference service**:
   - Update `backend/ml/model_loader.py` to load v3 checkpoint
   - Verify `id2label` mapping loaded correctly
   - Test on real cookbook pages

## References

- Label taxonomy: `ml/config/labels.py`
- Relabeling stats: `data/processed/v3_headers_titles/relabel_stats.json`
- Dataset manifest: `data/datasets/boston_layoutlmv3_v3/dataset_manifest.json`
- Training report: `models/layoutlmv3_v3_headers/training_report.json`
