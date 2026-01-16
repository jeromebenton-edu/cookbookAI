#!/usr/bin/env python3
"""
Validate demo prediction fixtures.

Checks that generated fixtures conform to the DemoPrediction schema
and contain reasonable data.

Usage:
    python tools/validate_demo_fixture.py \\
        frontend/src/demo_examples/example_01/prediction.json
"""

import argparse
import json
import sys
from pathlib import Path


def validate_fixture(fixture_path: Path) -> bool:
    """
    Validate a demo prediction fixture.

    Returns:
        True if valid, False otherwise
    """
    print(f"Validating {fixture_path}...")

    with open(fixture_path) as f:
        data = json.load(f)

    errors = []
    warnings = []

    # Schema version
    if data.get("schemaVersion") != "demo_pred_v1":
        errors.append(f"Invalid schemaVersion: {data.get('schemaVersion')}")

    # Page info
    page = data.get("page", {})
    if not all(k in page for k in ["width", "height", "coordSpace"]):
        errors.append("Missing required page fields")

    if page.get("coordSpace") not in ["px", "norm_1000"]:
        errors.append(f"Invalid coordSpace: {page.get('coordSpace')}")

    # Tokens
    tokens = data.get("tokens", [])
    if len(tokens) < 50:
        warnings.append(f"Low token count: {len(tokens)} (expected > 50)")

    required_token_fields = ["id", "text", "bbox", "label", "conf"]
    for i, tok in enumerate(tokens[:5]):  # Check first 5
        missing = [f for f in required_token_fields if f not in tok]
        if missing:
            errors.append(f"Token {i} missing fields: {missing}")

    # Lines
    lines = data.get("lines", [])
    if len(lines) < 5:
        warnings.append(f"Low line count: {len(lines)} (expected > 5)")

    # Sections
    sections = data.get("sections", {})
    if "title" not in sections:
        warnings.append("No title section found")
    if "ingredients" not in sections:
        errors.append("No ingredients section found")
    if "instructions" not in sections:
        errors.append("No instructions section found")

    # Extracted recipe
    recipe = data.get("extractedRecipe", {})
    title = recipe.get("title", "")
    if not title or title == "Unknown Recipe":
        errors.append("Title not extracted")

    ingredients = recipe.get("ingredients", [])
    if len(ingredients) < 1:
        errors.append("No ingredients extracted")

    instructions = recipe.get("instructions", [])
    if len(instructions) < 1:
        errors.append("No instructions extracted")

    # Confidence scores
    conf = recipe.get("confidence", {})
    for key in ["title", "ingredients", "instructions", "overall"]:
        if key not in conf:
            errors.append(f"Missing confidence.{key}")
        elif not (0 <= conf[key] <= 1):
            errors.append(f"Invalid confidence.{key}: {conf[key]}")

    # Meta
    meta = data.get("meta", {})
    if "exampleId" not in meta:
        errors.append("Missing meta.exampleId")
    if "generatedAt" not in meta:
        warnings.append("Missing meta.generatedAt")

    # Print results
    print(f"\n  Tokens: {len(tokens)}")
    print(f"  Lines: {len(lines)}")
    print(f"  Title: {title}")
    print(f"  Ingredients: {len(ingredients)}")
    print(f"  Instructions: {len(instructions)}")

    if errors:
        print(f"\n  ❌ ERRORS ({len(errors)}):")
        for err in errors:
            print(f"    - {err}")

    if warnings:
        print(f"\n  ⚠️  WARNINGS ({len(warnings)}):")
        for warn in warnings:
            print(f"    - {warn}")

    if not errors:
        print(f"\n  ✅ Fixture is valid")
        return True
    else:
        print(f"\n  ❌ Fixture is invalid")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate demo prediction fixtures")
    parser.add_argument(
        "fixtures",
        type=Path,
        nargs="+",
        help="Prediction JSON file(s) to validate",
    )

    args = parser.parse_args()

    all_valid = True
    for fixture_path in args.fixtures:
        if not fixture_path.exists():
            print(f"ERROR: File not found: {fixture_path}")
            all_valid = False
            continue

        valid = validate_fixture(fixture_path)
        all_valid = all_valid and valid
        print()

    if all_valid:
        print("✅ All fixtures valid")
        sys.exit(0)
    else:
        print("❌ Some fixtures invalid")
        sys.exit(1)


if __name__ == "__main__":
    main()
