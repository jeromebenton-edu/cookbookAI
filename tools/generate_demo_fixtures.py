#!/usr/bin/env python3
"""
Generate real token-level prediction fixtures for offline demo examples.

This script runs OCR + LayoutLMv3 inference on demo scan images and produces
canonical prediction JSON files that serve as the single source of truth for
the /demo page (network-free).

Usage:
    python tools/generate_demo_fixtures.py \\
        --examples-config tools/demo_examples_config.json \\
        --output-root frontend/src/demo_examples

Requirements:
    - pytesseract (or easyocr/paddleocr)
    - PIL/Pillow
    - torch
    - transformers (for LayoutLMv3)

Schema version: demo_pred_v1
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

try:
    import pytesseract
    from PIL import Image
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    print("Warning: pytesseract not available. Install with: pip install pytesseract Pillow")

try:
    import torch
    from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
    HAS_LAYOUTLM = True
except ImportError:
    HAS_LAYOUTLM = False
    print("Warning: transformers/torch not available. Install with: pip install torch transformers")


# Label mappings - loaded from model config at runtime
# Default v3 taxonomy (will be overridden if model has config.json)
DEFAULT_LABEL_MAP_V3 = {
    0: "PAGE_HEADER",
    1: "SECTION_HEADER",
    2: "RECIPE_TITLE",
    3: "INGREDIENT_LINE",
    4: "INSTRUCTION_STEP",
    5: "TIME",
    6: "TEMP",
    7: "SERVINGS",
    8: "NOTE",
    9: "O",
}

# Legacy v2 map for mock mode
DEFAULT_LABEL_MAP_V2 = {
    0: "TITLE",
    1: "INGREDIENT_LINE",
    2: "INSTRUCTION_STEP",
    3: "TIME",
    4: "TEMP",
    5: "SERVINGS",
    6: "NOTE",
    7: "O",
}


def run_ocr(image_path: Path) -> Tuple[List[Dict], int, int]:
    """
    Run OCR on image and return word-level tokens with bounding boxes.

    Returns:
        (tokens, image_width, image_height)
        tokens: List[{text, bbox: [x1,y1,x2,y2]}]
    """
    if not HAS_TESSERACT:
        raise RuntimeError("Tesseract not available. Install pytesseract.")

    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    # Get OCR data at word level
    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    tokens = []
    n_boxes = len(ocr_data['text'])

    for i in range(n_boxes):
        text = ocr_data['text'][i].strip()
        conf = int(ocr_data['conf'][i])

        # Skip empty text or low confidence
        if not text or conf < 0:
            continue

        x, y, w, h = (
            ocr_data['left'][i],
            ocr_data['top'][i],
            ocr_data['width'][i],
            ocr_data['height'][i]
        )

        bbox = [x, y, x + w, y + h]

        tokens.append({
            "text": text,
            "bbox": bbox,
            "ocr_conf": conf / 100.0  # Normalize to 0-1
        })

    print(f"  OCR extracted {len(tokens)} tokens from {width}x{height}px image")
    return tokens, width, height


def run_layoutlmv3_inference(
    image_path: Path,
    ocr_tokens: List[Dict],
    model_checkpoint: Optional[str] = None
) -> List[Dict]:
    """
    Run LayoutLMv3 inference to get token-level predictions.

    Args:
        image_path: Path to scan image
        ocr_tokens: OCR tokens with text + bbox
        model_checkpoint: Path to fine-tuned checkpoint (or None for mock)

    Returns:
        tokens: List[{id, text, bbox, label, conf}]
    """
    if not HAS_LAYOUTLM or model_checkpoint is None:
        # MOCK MODE: Generate fake but plausible predictions
        print("  [MOCK MODE] Generating synthetic predictions (no model available)")
        return generate_mock_predictions(ocr_tokens)

    # Load model and processor
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_checkpoint)
    model.eval()

    # Load id2label from model config (critical for v3 labels!)
    id2label = model.config.id2label
    print(f"  Model has {len(id2label)} labels: {list(id2label.values())[:5]}...")

    # Prepare inputs
    image = Image.open(image_path).convert("RGB")
    words = [t["text"] for t in ocr_tokens]
    boxes = [t["bbox"] for t in ocr_tokens]

    # Normalize boxes to 0-1000 scale (LayoutLMv3 expects this)
    width, height = image.size
    normalized_boxes = [
        [
            int(1000 * (b[0] / width)),
            int(1000 * (b[1] / height)),
            int(1000 * (b[2] / width)),
            int(1000 * (b[3] / height)),
        ]
        for b in boxes
    ]

    encoding = processor(
        image,
        words,
        boxes=normalized_boxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
    )

    # Run inference
    with torch.no_grad():
        outputs = model(**encoding)
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        probs = torch.softmax(outputs.logits, dim=-1).squeeze()

    # Map predictions back to tokens
    tokens = []
    for idx, (word, bbox, pred_id) in enumerate(zip(words, boxes, predictions)):
        if isinstance(pred_id, list):
            pred_id = pred_id[0]  # Handle batch dim

        # Use model's id2label mapping
        label = id2label.get(pred_id, "O")
        conf = float(probs[idx, pred_id])

        tokens.append({
            "id": idx,
            "text": word,
            "bbox": bbox,
            "label": label,
            "conf": conf,
        })

    print(f"  LayoutLMv3 predicted {len(tokens)} tokens")
    return tokens


def generate_mock_predictions(ocr_tokens: List[Dict]) -> List[Dict]:
    """
    Generate mock predictions when no model is available.

    Uses heuristics based on position and text patterns to assign plausible labels.
    """
    tokens = []

    for idx, ocr_tok in enumerate(ocr_tokens):
        text = ocr_tok["text"]
        bbox = ocr_tok["bbox"]
        y_center = (bbox[1] + bbox[3]) / 2

        # Heuristic labeling (very basic)
        label = "O"
        conf = 0.85

        # Title heuristic: Skip page header region (top 50px), look for titles in 50-250px range
        # Also require at least one lowercase letter to avoid all-caps headers
        if (50 < y_center < 250 and
            text[0].isupper() and
            len(text) > 3 and
            any(c.islower() for c in text)):
            label = "TITLE"
            conf = 0.92

        # Ingredient heuristic: contains numbers, units, or common ingredient words
        elif any(c.isdigit() for c in text) or text.lower() in ["cup", "cups", "tsp", "tbsp", "flour", "sugar", "salt", "butter", "milk", "eggs"]:
            label = "INGREDIENT_LINE"
            conf = 0.88

        # Instruction heuristic: action verbs
        elif text.lower() in ["mix", "stir", "bake", "add", "beat", "fold", "cook", "heat", "pour", "sift", "combine"]:
            label = "INSTRUCTION_STEP"
            conf = 0.86

        tokens.append({
            "id": idx,
            "text": text,
            "bbox": bbox,
            "label": label,
            "conf": conf,
        })

    return tokens


def group_tokens_into_lines(tokens: List[Dict], height: int) -> List[Dict]:
    """
    Group tokens into lines using Y-coordinate clustering.

    Args:
        tokens: Token list with bbox
        height: Image height (for normalization)

    Returns:
        lines: List[{id, kind, bbox, tokenIds, conf, text}]
    """
    if not tokens:
        return []

    # Sort tokens by Y position, then X
    sorted_tokens = sorted(tokens, key=lambda t: (t["bbox"][1], t["bbox"][0]))

    lines = []
    current_line_tokens = []
    current_y_band = None
    Y_TOLERANCE = height * 0.015  # 1.5% of image height

    for token in sorted_tokens:
        y_center = (token["bbox"][1] + token["bbox"][3]) / 2

        if current_y_band is None or abs(y_center - current_y_band) > Y_TOLERANCE:
            # Start new line
            if current_line_tokens:
                lines.append(create_line_from_tokens(current_line_tokens, len(lines)))
            current_line_tokens = [token]
            current_y_band = y_center
        else:
            # Continue current line
            current_line_tokens.append(token)

    # Add final line
    if current_line_tokens:
        lines.append(create_line_from_tokens(current_line_tokens, len(lines)))

    print(f"  Grouped {len(tokens)} tokens into {len(lines)} lines")
    return lines


def create_line_from_tokens(tokens: List[Dict], line_id: int) -> Dict:
    """Create a line object from a list of tokens."""
    # Determine line kind from majority label
    label_counts = {}
    for t in tokens:
        label = t["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    majority_label = max(label_counts.items(), key=lambda x: x[1])[0]

    # Map label to line kind
    # v3 labels: Ignore PAGE_HEADER and SECTION_HEADER for recipe extraction
    if majority_label in ("RECIPE_TITLE", "TITLE"):  # Support both v3 and legacy v2
        kind = "TITLE_LINE"
    elif majority_label == "PAGE_HEADER":
        kind = "PAGE_HEADER"
    elif majority_label == "SECTION_HEADER":
        kind = "SECTION_HEADER"
    elif majority_label == "INGREDIENT_LINE":
        kind = "INGREDIENT_LINE"
    elif majority_label == "INSTRUCTION_STEP":
        kind = "INSTRUCTION_STEP"
    else:
        kind = "OTHER"

    # Compute union bbox
    x1 = min(t["bbox"][0] for t in tokens)
    y1 = min(t["bbox"][1] for t in tokens)
    x2 = max(t["bbox"][2] for t in tokens)
    y2 = max(t["bbox"][3] for t in tokens)

    # Average confidence
    avg_conf = sum(t["conf"] for t in tokens) / len(tokens)

    # Concatenate text
    text = " ".join(t["text"] for t in tokens)

    return {
        "id": line_id,
        "kind": kind,
        "bbox": [x1, y1, x2, y2],
        "tokenIds": [t["id"] for t in tokens],
        "conf": avg_conf,
        "text": text,
    }


def create_sections_from_lines(lines: List[Dict], width: int) -> Dict:
    """
    Create section-level regions from lines.

    Args:
        lines: Line list
        width: Image width (for column detection)

    Returns:
        sections: {title?, ingredients?, instructions?}
    """
    sections = {}

    # Title section
    title_lines = [l for l in lines if l["kind"] == "TITLE_LINE"]
    if title_lines:
        bbox = union_boxes([l["bbox"] for l in title_lines])
        sections["title"] = {
            "bbox": bbox,
            "lineIds": [l["id"] for l in title_lines],
            "conf": sum(l["conf"] for l in title_lines) / len(title_lines),
        }

    # Ingredients section (detect two-column)
    ing_lines = [l for l in lines if l["kind"] == "INGREDIENT_LINE"]
    if ing_lines:
        x_centers = [(l["bbox"][0] + l["bbox"][2]) / 2 for l in ing_lines]
        x_min, x_max = min(x_centers), max(x_centers)
        x_spread = x_max - x_min

        if x_spread > width * 0.4 and len(ing_lines) >= 4:
            # Two columns
            x_median = (x_min + x_max) / 2
            left_lines = [l for l in ing_lines if (l["bbox"][0] + l["bbox"][2]) / 2 < x_median]
            right_lines = [l for l in ing_lines if (l["bbox"][0] + l["bbox"][2]) / 2 >= x_median]

            sections["ingredients"] = []
            if left_lines:
                sections["ingredients"].append({
                    "bbox": union_boxes([l["bbox"] for l in left_lines]),
                    "lineIds": [l["id"] for l in left_lines],
                    "conf": sum(l["conf"] for l in left_lines) / len(left_lines),
                })
            if right_lines:
                sections["ingredients"].append({
                    "bbox": union_boxes([l["bbox"] for l in right_lines]),
                    "lineIds": [l["id"] for l in right_lines],
                    "conf": sum(l["conf"] for l in right_lines) / len(right_lines),
                })
        else:
            # Single column
            sections["ingredients"] = [{
                "bbox": union_boxes([l["bbox"] for l in ing_lines]),
                "lineIds": [l["id"] for l in ing_lines],
                "conf": sum(l["conf"] for l in ing_lines) / len(ing_lines),
            }]

    # Instructions section
    inst_lines = [l for l in lines if l["kind"] == "INSTRUCTION_STEP"]
    if inst_lines:
        sections["instructions"] = {
            "bbox": union_boxes([l["bbox"] for l in inst_lines]),
            "lineIds": [l["id"] for l in inst_lines],
            "conf": sum(l["conf"] for l in inst_lines) / len(inst_lines),
        }

    return sections


def union_boxes(boxes: List[List[int]]) -> List[int]:
    """Compute union bounding box."""
    if not boxes:
        return [0, 0, 0, 0]
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return [x1, y1, x2, y2]


def extract_recipe(lines: List[Dict], sections: Dict) -> Dict:
    """
    Extract structured recipe from lines.

    Returns:
        ExtractedRecipeV1 object
    """
    # Title
    title_lines = [l for l in lines if l["kind"] == "TITLE_LINE"]
    title = " ".join(l["text"] for l in title_lines) if title_lines else "Unknown Recipe"

    # Ingredients
    ing_lines = [l for l in lines if l["kind"] == "INGREDIENT_LINE"]
    ingredients = [l["text"] for l in sorted(ing_lines, key=lambda x: x["bbox"][1])]

    # Instructions
    inst_lines = [l for l in lines if l["kind"] == "INSTRUCTION_STEP"]
    instructions = [l["text"] for l in sorted(inst_lines, key=lambda x: x["bbox"][1])]

    # Confidence scores
    title_conf = sections.get("title", {}).get("conf", 0.9)
    ing_sections = sections.get("ingredients", [])
    ing_conf = (sum(s["conf"] for s in ing_sections) / len(ing_sections)) if ing_sections else 0.9
    inst_conf = sections.get("instructions", {}).get("conf", 0.9)
    overall_conf = (title_conf + ing_conf + inst_conf) / 3

    return {
        "title": title,
        "ingredients": ingredients,
        "instructions": instructions,
        "confidence": {
            "title": title_conf,
            "ingredients": ing_conf,
            "instructions": inst_conf,
            "overall": overall_conf,
        },
    }


def generate_fixture(
    example_id: str,
    image_path: Path,
    model_checkpoint: Optional[str],
    cookbook_page: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate complete demo prediction fixture for one example.

    Returns:
        DemoPrediction object
    """
    print(f"\n[{example_id}] Generating fixture...")
    print(f"  Scan image: {image_path}")

    # Step 1: OCR
    ocr_tokens, width, height = run_ocr(image_path)

    # Step 2: LayoutLMv3 inference
    tokens = run_layoutlmv3_inference(image_path, ocr_tokens, model_checkpoint)

    # Step 3: Group into lines
    lines = group_tokens_into_lines(tokens, height)

    # Step 4: Create sections
    sections = create_sections_from_lines(lines, width)

    # Step 5: Extract recipe
    extracted_recipe = extract_recipe(lines, sections)

    # Step 6: Assemble prediction
    prediction = {
        "schemaVersion": "demo_pred_v1",
        "page": {
            "width": width,
            "height": height,
            "coordSpace": "px",
        },
        "tokens": tokens,
        "lines": lines,
        "sections": sections,
        "extractedRecipe": extracted_recipe,
        "meta": {
            "exampleId": example_id,
            "cookbookPage": cookbook_page,
            "scanIndex": None,  # Not applicable for demo fixtures
            "generatedAt": datetime.utcnow().isoformat() + "Z",
            "modelId": str(model_checkpoint) if model_checkpoint else "mock",
            "ocrEngine": "pytesseract" if HAS_TESSERACT else "mock",
        },
    }

    print(f"  ✓ Generated prediction:")
    print(f"    - Title: {extracted_recipe['title']}")
    print(f"    - {len(tokens)} tokens, {len(lines)} lines")
    print(f"    - {len(extracted_recipe['ingredients'])} ingredients, {len(extracted_recipe['instructions'])} instructions")

    return prediction


def main():
    parser = argparse.ArgumentParser(description="Generate demo prediction fixtures")
    parser.add_argument(
        "--examples-config",
        type=Path,
        required=True,
        help="Path to demo_examples_config.json",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=Path,
        default=None,
        help="Path to LayoutLMv3 checkpoint (if None, uses mock mode)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("frontend/src/demo_examples"),
        help="Output root directory",
    )

    args = parser.parse_args()

    # Load config
    with open(args.examples_config) as f:
        config = json.load(f)

    print(f"Loaded {len(config['examples'])} examples from config")

    # Generate fixtures
    for example in config["examples"]:
        example_id = example["exampleId"]
        image_path = Path(example["scanImagePath"])
        cookbook_page = example.get("cookbookPage")

        if not image_path.exists():
            print(f"ERROR: Scan image not found: {image_path}")
            sys.exit(1)

        # Generate prediction
        prediction = generate_fixture(
            example_id,
            image_path,
            args.model_checkpoint,
            cookbook_page,
        )

        # Write to output
        output_dir = args.output_root / example_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "prediction.json"

        with open(output_path, "w") as f:
            json.dump(prediction, f, indent=2)

        print(f"  ✓ Wrote {output_path}")

    print(f"\n✅ Generated {len(config['examples'])} fixtures successfully")


if __name__ == "__main__":
    main()
