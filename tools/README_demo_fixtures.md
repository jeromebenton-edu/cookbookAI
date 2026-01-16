# Demo Fixture Generation

This directory contains tooling to generate **real token-level prediction fixtures** for the `/demo` page.

## Overview

The `/demo` page is **network-free** - it doesn't require a backend ML service. Instead, it uses pre-generated prediction fixtures that are bundled with the frontend code.

These fixtures are created by running:
1. **OCR** (Tesseract) on the demo scan images to extract words + bounding boxes
2. **LayoutLMv3 inference** to predict labels (TITLE, INGREDIENT_LINE, INSTRUCTION_STEP, etc.)
3. **Grouping** tokens into lines and sections
4. **Extraction** of structured recipe data

The result is a canonical `prediction.json` file following the `DemoPrediction` schema (see [`frontend/lib/demoPredictionTypes.ts`](../frontend/lib/demoPredictionTypes.ts)).

## Requirements

### Python Dependencies

```bash
pip install pytesseract Pillow torch transformers
```

### System Dependencies

- **Tesseract OCR**: Install system package
  ```bash
  # Ubuntu/Debian
  sudo apt-get install tesseract-ocr

  # macOS
  brew install tesseract

  # Windows
  # Download installer from https://github.com/UB-Mannheim/tesseract/wiki
  ```

### Optional: LayoutLMv3 Model Checkpoint

If you have a fine-tuned LayoutLMv3 checkpoint, provide the path with `--model-checkpoint`.

If not provided, the script runs in **MOCK MODE** which uses heuristic labeling (still useful for testing the pipeline).

## Usage

### Basic Usage (Mock Mode)

Generate fixtures using heuristic predictions (no ML model required):

```bash
python tools/generate_demo_fixtures.py \\
  --examples-config tools/demo_examples_config.json \\
  --output-root frontend/src/demo_examples
```

### With LayoutLMv3 Model

Generate fixtures using a trained model:

```bash
python tools/generate_demo_fixtures.py \\
  --examples-config tools/demo_examples_config.json \\
  --model-checkpoint path/to/layoutlmv3-cookbook-finetuned \\
  --output-root frontend/src/demo_examples
```

## Output

For each example in the config, generates:

```
frontend/src/demo_examples/
├── example_01/
│   └── prediction.json    # DemoPrediction schema
└── example_02/
    └── prediction.json
```

### Output Schema

See [`frontend/lib/demoPredictionTypes.ts`](../frontend/lib/demoPredictionTypes.ts) for the complete TypeScript schema.

**Key fields**:
- `tokens[]`: Word-level predictions from LayoutLMv3
- `lines[]`: Grouped into logical lines (ingredient lines, instruction steps)
- `sections`: Title, Ingredients, Instructions regions
- `extractedRecipe`: Final structured output (title, ingredients[], instructions[])
- `page`: Image dimensions and coordinate space

**Coordinate Space**: All bounding boxes are in **pixel coordinates** relative to the original scan image dimensions (e.g., 500x819px).

## Adding a New Example

1. **Add the scan image** to `frontend/public/demo_examples/example_XX/page.png`

2. **Update config** in [`tools/demo_examples_config.json`](./demo_examples_config.json):
   ```json
   {
     "examples": [
       {
         "exampleId": "example_03",
         "scanImagePath": "frontend/public/demo_examples/example_03/page.png",
         "cookbookPage": 42,
         "description": "My Recipe from Some Cookbook"
       }
     ]
   }
   ```

3. **Run the generator**:
   ```bash
   python tools/generate_demo_fixtures.py \\
     --examples-config tools/demo_examples_config.json
   ```

4. **Update frontend** to load the new example in [`frontend/lib/bundledExamples.ts`](../frontend/lib/bundledExamples.ts)

## Validation

Validate generated fixtures:

```bash
python tools/validate_demo_fixture.py \\
  frontend/src/demo_examples/example_01/prediction.json
```

Checks:
- Schema version matches
- Token count > 50
- Title extracted (non-empty)
- At least 1 ingredient line and 1 instruction line
- Section bboxes have reasonable sizes

## Troubleshooting

### "pytesseract not available"

Install Tesseract OCR (see Requirements above).

### "transformers/torch not available"

```bash
pip install torch transformers
```

Or run in mock mode (no `--model-checkpoint` flag).

### Boxes Don't Align

Check that:
1. Coordinate space is `"px"` in `prediction.json`
2. `page.width` and `page.height` match the actual scan image dimensions
3. Frontend overlay viewer uses `naturalWidth/naturalHeight` for scaling

### Low Quality Predictions (Mock Mode)

Mock mode uses simple heuristics. For production-quality fixtures, train a LayoutLMv3 model and provide `--model-checkpoint`.

## Development

### Script Architecture

```
generate_demo_fixtures.py
├── run_ocr()                # Tesseract OCR
├── run_layoutlmv3_inference()  # ML model (or mock)
├── group_tokens_into_lines()   # Y-coordinate clustering
├── create_sections_from_lines()  # Semantic regions
├── extract_recipe()        # Structured output
└── main()                  # Orchestration
```

### Mock Mode Details

When no model is provided, the script uses heuristics:
- **TITLE**: Early in page (y < 200), capitalized, short word
- **INGREDIENT_LINE**: Contains numbers, units ("cup", "tsp"), or common ingredients
- **INSTRUCTION_STEP**: Action verbs ("mix", "bake", "stir")
- **Confidence**: Random 0.85-0.92

This is **good enough for UI development** but not for production demos.

## CI/CD Integration

Add to GitHub Actions:

```yaml
- name: Generate demo fixtures
  run: |
    pip install pytesseract Pillow
    sudo apt-get install -y tesseract-ocr
    python tools/generate_demo_fixtures.py \\
      --examples-config tools/demo_examples_config.json

- name: Validate fixtures
  run: |
    python tools/validate_demo_fixture.py \\
      frontend/src/demo_examples/*/prediction.json
```

## License

Same as parent project (MIT).
