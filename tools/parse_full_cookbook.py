#!/usr/bin/env python3
"""
Parse the full Fanny Farmer 1918 cookbook using the balanced LayoutLMv3 model.

This script:
1. Loads the balanced model (layoutlmv3_v3_manual_59pages_balanced)
2. Runs inference on all ~623 pages
3. Extracts recipes with ingredients and instructions
4. Saves recipe JSON files to frontend/public/recipes/boston/
5. Generates an index.json for the frontend
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

import torch
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.utils.recipe_extraction import recipe_from_prediction
from backend.app.utils.ocr_enhanced import postprocess_word


def slugify(text: str) -> str:
    """Convert title to URL-friendly slug."""
    import re

    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def run_inference_on_page(
    model,
    processor,
    dataset,
    page_index,
    page_num: int,
    device: torch.device,
) -> Optional[dict]:
    """
    Run inference on a single page and return prediction dict.

    Returns None if page not found in dataset.
    """
    if page_num not in page_index:
        return None

    split, idx = page_index[page_num]
    record = dict(dataset[split][idx])

    # Load image
    image_path = record["image_path"]
    if not Path(image_path).is_absolute():
        image_path = PROJECT_ROOT / image_path

    if not Path(image_path).exists():
        print(f"Warning: Image not found for page {page_num}: {image_path}")
        return None

    image = Image.open(image_path).convert("RGB")

    # Apply OCR post-processing corrections before using words
    raw_words = [w for w in record["words"] if w]  # Filter empty strings
    words = [postprocess_word(w) for w in raw_words]  # Fix common OCR errors
    bboxes = record["bboxes"][:len(words)]  # Match length

    # Prepare inputs
    encoding = processor(
        image,
        words,
        boxes=bboxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )

    # Get word_ids before moving to device
    word_ids = encoding.word_ids(batch_index=0)

    # Move to device
    encoding_tensors = {k: v.to(device) for k, v in encoding.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**encoding_tensors)
        predictions = outputs.logits.argmax(-1).squeeze().tolist()

        # Get probabilities for confidence scores
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Decode predictions and build token list
    id2label = model.config.id2label

    # Aggregate predictions per word (take max confidence)
    per_word = {}
    for idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        conf = probs[0, idx].max().item()
        pred_id = predictions[idx] if isinstance(predictions, list) else predictions

        if wid not in per_word or conf > per_word[wid][1]:
            per_word[wid] = (int(pred_id), float(conf))

    # Build tokens list
    tokens = []
    for wid in sorted(per_word.keys()):
        if wid >= len(words):
            continue
        pred_id, conf = per_word[wid]
        label = id2label[pred_id]

        tokens.append({
            "text": words[wid],
            "bbox": bboxes[wid],
            "pred_label": label,
            "label": label,
            "pred_id": pred_id,
            "confidence": conf,
            "score": conf,
        })

    # Build prediction dict matching backend format
    return {
        "page_num": page_num,
        "image_path": str(image_path),
        "image_url": f"/recipes/boston/pages/{page_num:04d}.png",
        "tokens": tokens,
        "meta": {
            "model": "layoutlmv3_v3_manual_59pages_balanced",
            "num_tokens": len(tokens),
        },
    }


def is_recipe_page(prediction: dict, min_ingredients: int = 1, min_instructions: int = 1) -> bool:
    """
    Determine if a page is likely a recipe page based on predictions.

    Heuristic: Must have RECIPE_TITLE OR (ingredients AND instructions)
    """
    tokens = prediction.get("tokens", [])

    has_title = any(t.get("pred_label") == "RECIPE_TITLE" for t in tokens)
    ingredient_count = sum(1 for t in tokens if t.get("pred_label") == "INGREDIENT_LINE")
    instruction_count = sum(1 for t in tokens if t.get("pred_label") == "INSTRUCTION_STEP")

    # Accept if has title, or if has both ingredients and instructions
    return has_title or (ingredient_count >= min_ingredients and instruction_count >= min_instructions)


def main():
    """Main parsing loop."""
    print("=" * 80)
    print("CookbookAI - Full Cookbook Parser")
    print("=" * 80)

    # Configuration
    model_path = PROJECT_ROOT / "models/layoutlmv3_v3_manual_59pages_balanced"
    dataset_path = PROJECT_ROOT / "data/datasets/boston_layoutlmv3_v3_manual_expanded2/dataset_dict"
    output_dir = PROJECT_ROOT / "frontend/public/recipes/boston"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean old recipe files (but keep pages/ subdirectory and any backup directories)
    print(f"\nCleaning old recipe files from {output_dir}...")
    for old_file in output_dir.glob("*.json"):
        old_file.unlink()
    print("Old recipe files removed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load model
    print(f"\nLoading model from {model_path}...")
    processor = LayoutLMv3Processor.from_pretrained(model_path)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_path).to(device)
    model.eval()  # Set to evaluation mode for deterministic predictions
    print(f"Model loaded with {len(model.config.id2label)} labels")

    # Load dataset
    print(f"\nLoading dataset from {dataset_path}...")
    dataset = load_from_disk(str(dataset_path))

    # Build page index
    page_index = {}
    for split in dataset.keys():
        for idx, num in enumerate(dataset[split]["page_num"]):
            page_index[int(num)] = (split, idx)

    print(f"Dataset loaded: {len(page_index)} pages indexed")

    # Get all page numbers (only recipe pages: 69-535)
    # Pages 1-68: front matter, table of contents
    # Pages 536+: back matter, index, not recipes
    all_pages = sorted([p for p in page_index.keys() if 69 <= p <= 535])
    print(f"\nProcessing {len(all_pages)} pages (from {min(all_pages)} to {max(all_pages)})...")
    print(f"Note: Skipping pages 1-68 (front matter) and 536+ (back matter)")

    # Parse all pages
    recipes = []
    recipe_pages = []
    non_recipe_pages = []
    errors = []

    for page_num in tqdm(all_pages, desc="Processing pages"):
        try:
            # Run inference
            prediction = run_inference_on_page(
                model, processor, dataset, page_index, page_num, device
            )

            if prediction is None:
                errors.append({"page": page_num, "error": "Page not found in dataset"})
                continue

            # Check if it's a recipe page
            if not is_recipe_page(prediction):
                non_recipe_pages.append(page_num)
                continue

            # Extract recipe
            recipe = recipe_from_prediction(prediction, include_raw=False, include_lines=True)

            # Skip if no title and no content (need at least some structure)
            has_content = recipe.get("ingredients") or recipe.get("instructions")
            if not recipe.get("title") and not has_content:
                non_recipe_pages.append(page_num)
                continue

            # Generate recipe ID
            title = recipe["title"] if recipe["title"] else f"Recipe from page {page_num}"
            slug = slugify(title) if recipe["title"] else f"recipe-{page_num}"
            recipe_id = f"{slug}-p{page_num:04d}" if slug else f"recipe-p{page_num:04d}"

            # Build final recipe JSON in BostonRecipe format
            recipe_json = {
                "id": recipe_id,
                "book": "Boston Cooking-School Cook Book",
                "year": 1918,
                "title": title,
                "category": "Historical",
                "tags": ["1918", "fanny-farmer", "historical"],
                "time": {
                    "prep": "Variable",
                    "cook": "Variable",
                    "total": "See instructions"
                },
                "servings": "Variable",
                "ingredients": recipe["ingredients"],
                "instructions": recipe["instructions"],
                "notes": [],
                "source": {
                    "page": page_num,
                    "pdf_url": f"https://archive.org/details/bostoncookingsch00farm/page/{page_num}"
                },
                "ai": {
                    "page_image": f"/recipes/boston/pages/{page_num:04d}.png",
                    "tokens": [],  # Could include token data here if needed
                    "field_confidence": {
                        "title": recipe["confidence"]["title"],
                        "ingredients": recipe["confidence"]["ingredients"],
                        "instructions": recipe["confidence"]["instructions"],
                    },
                },
            }

            # Save recipe
            recipe_file = output_dir / f"{recipe_id}.json"
            recipe_file.write_text(json.dumps(recipe_json, indent=2))

            recipes.append({
                "id": recipe_id,
                "title": title,
                "page": page_num,
                "confidence": recipe["confidence"]["overall"],
            })
            recipe_pages.append(page_num)

        except Exception as e:
            errors.append({"page": page_num, "error": str(e)})
            print(f"\nError processing page {page_num}: {e}")

    # Generate index in frontend-compatible format
    index = {
        "collection": "Boston Cooking-School Cook Book",
        "year": 1918,
        "total_pages": len(all_pages),
        "recipe_pages": len(recipe_pages),
        "non_recipe_pages": len(non_recipe_pages),
        "errors": len(errors),
        "recipes": [
            {
                "id": r["id"],
                "title": r["title"],
                "category": "Historical",  # Could be improved with classification
                "tags": ["1918", "fanny-farmer", "historical"],
            }
            for r in sorted(recipes, key=lambda r: r["page"])
        ],
        "model": "layoutlmv3_v3_manual_59pages_balanced",
        "generated_at": "2026-01-14",
    }

    index_file = output_dir / "index.json"
    index_file.write_text(json.dumps(index, indent=2))

    # Print summary
    print("\n" + "=" * 80)
    print("PARSING COMPLETE")
    print("=" * 80)
    print(f"Total pages processed: {len(all_pages)}")
    print(f"Recipe pages found: {len(recipe_pages)}")
    print(f"Non-recipe pages: {len(non_recipe_pages)}")
    print(f"Errors: {len(errors)}")
    print(f"\nRecipes saved to: {output_dir}")
    print(f"Index saved to: {index_file}")

    if errors:
        print(f"\nErrors encountered: {len(errors)}")
        for err in errors[:5]:
            print(f"  Page {err['page']}: {err['error']}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
