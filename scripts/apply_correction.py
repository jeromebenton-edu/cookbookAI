#!/usr/bin/env python3
"""
Apply corrections from data/corrections/ to curated recipe files.

Usage:
    python scripts/apply_correction.py data/corrections/boston_page_0079_corrected.json
    python scripts/apply_correction.py data/corrections/  # Apply all corrections in folder
    python scripts/apply_correction.py --dry-run data/corrections/  # Preview without writing
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def find_recipe_file(page_num: int, recipes_dir: Path) -> Optional[Path]:
    """Find the curated recipe file for a given page number."""
    pattern = f"*-p{page_num:04d}.json"
    matches = list(recipes_dir.glob(pattern))
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"  Warning: Multiple matches for page {page_num}: {matches}")
        return matches[0]
    return None


def apply_correction(correction_path: Path, recipes_dir: Path, dry_run: bool = False) -> bool:
    """Apply a single correction file to its corresponding recipe."""
    print(f"\nProcessing: {correction_path.name}")

    try:
        with open(correction_path) as f:
            correction = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  Error: Invalid JSON - {e}")
        return False

    page_num = correction.get("page_num")
    corrected = correction.get("corrected", {})

    if not page_num:
        print("  Error: No page_num in correction file")
        return False

    if not corrected:
        print("  Error: No corrected data in correction file")
        return False

    # Find the recipe file
    recipe_path = find_recipe_file(page_num, recipes_dir)
    if not recipe_path:
        print(f"  Error: No recipe file found for page {page_num}")
        return False

    print(f"  Found recipe: {recipe_path.name}")

    # Load existing recipe
    with open(recipe_path) as f:
        recipe = json.load(f)

    # Track changes
    changes = []

    # Apply corrections
    if "title" in corrected and corrected["title"] != recipe.get("title"):
        changes.append(f"  title: '{recipe.get('title')}' -> '{corrected['title']}'")
        recipe["title"] = corrected["title"]

    if "ingredients" in corrected:
        old_count = len(recipe.get("ingredients", []))
        new_count = len(corrected["ingredients"])
        if recipe.get("ingredients") != corrected["ingredients"]:
            changes.append(f"  ingredients: {old_count} items -> {new_count} items")
            recipe["ingredients"] = corrected["ingredients"]

    if "instructions" in corrected:
        old_count = len(recipe.get("instructions", []))
        new_count = len(corrected["instructions"])
        if recipe.get("instructions") != corrected["instructions"]:
            changes.append(f"  instructions: {old_count} items -> {new_count} items")
            recipe["instructions"] = corrected["instructions"]

    if not changes:
        print("  No changes needed (already up to date)")
        return True

    print("  Changes:")
    for change in changes:
        print(f"    {change}")

    if dry_run:
        print("  [DRY RUN] Would write changes to file")
    else:
        with open(recipe_path, "w") as f:
            json.dump(recipe, f, indent=2)
        print(f"  Updated: {recipe_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Apply corrections to curated recipes")
    parser.add_argument("path", type=Path, help="Correction file or folder of corrections")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Preview changes without writing")
    parser.add_argument("--recipes-dir", type=Path,
                        default=Path("frontend/public/recipes/boston"),
                        help="Path to curated recipes folder")
    args = parser.parse_args()

    if not args.recipes_dir.exists():
        print(f"Error: Recipes directory not found: {args.recipes_dir}")
        sys.exit(1)

    # Collect correction files
    if args.path.is_file():
        correction_files = [args.path]
    elif args.path.is_dir():
        correction_files = sorted(args.path.glob("*_corrected.json"))
        if not correction_files:
            print(f"No correction files found in {args.path}")
            sys.exit(1)
    else:
        print(f"Error: Path not found: {args.path}")
        sys.exit(1)

    print(f"Found {len(correction_files)} correction file(s)")
    if args.dry_run:
        print("[DRY RUN MODE - no files will be modified]")

    success = 0
    failed = 0

    for correction_path in correction_files:
        if apply_correction(correction_path, args.recipes_dir, args.dry_run):
            success += 1
        else:
            failed += 1

    print(f"\nSummary: {success} succeeded, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
