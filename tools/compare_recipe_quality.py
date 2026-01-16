#!/usr/bin/env python3
"""Compare recipe quality before and after OCR improvements."""

import json
import sys
from pathlib import Path
from collections import Counter

def analyze_recipes(recipes_dir: Path):
    """Analyze recipe quality metrics."""
    if not recipes_dir.exists():
        print(f"Directory not found: {recipes_dir}")
        return None

    recipes = list(recipes_dir.glob("*.json"))
    if not recipes:
        print(f"No recipes found in {recipes_dir}")
        return None

    # Metrics
    total = len(recipes)
    with_titles = 0
    avg_ingredients = 0
    avg_instructions = 0
    title_lengths = []
    ingredient_counts = []
    instruction_counts = []

    for recipe_path in recipes:
        try:
            with open(recipe_path) as f:
                recipe = json.load(f)

            # Check title
            title = recipe.get("title", "")
            if title and not title.startswith("Recipe from page"):
                with_titles += 1
                title_lengths.append(len(title))

            # Count ingredients (check both field names)
            ingredients = recipe.get("ingredients_lines") or recipe.get("ingredients") or []
            ingredient_counts.append(len(ingredients))

            # Count instructions (check both field names)
            instructions = recipe.get("instruction_lines") or recipe.get("instructions") or []
            instruction_counts.append(len(instructions))

        except Exception as e:
            print(f"Error reading {recipe_path.name}: {e}")
            continue

    if ingredient_counts:
        avg_ingredients = sum(ingredient_counts) / len(ingredient_counts)
    if instruction_counts:
        avg_instructions = sum(instruction_counts) / len(instruction_counts)

    return {
        "total": total,
        "with_titles": with_titles,
        "title_rate": with_titles / total if total > 0 else 0,
        "avg_ingredients": avg_ingredients,
        "avg_instructions": avg_instructions,
        "avg_title_length": sum(title_lengths) / len(title_lengths) if title_lengths else 0,
    }


def main():
    recipes_dir = Path("frontend/public/recipes/boston")

    # Find most recent backup
    backup_dirs = sorted(Path("frontend/public/recipes").glob("boston_backup_*"))

    print("=" * 70)
    print("Recipe Quality Analysis")
    print("=" * 70)
    print()

    # Current recipes
    print("CURRENT RECIPES")
    print("-" * 70)
    current = analyze_recipes(recipes_dir)
    if current:
        print(f"Total recipes:        {current['total']}")
        print(f"With actual titles:   {current['with_titles']} ({current['title_rate']:.1%})")
        print(f"Avg title length:     {current['avg_title_length']:.1f} characters")
        print(f"Avg ingredients:      {current['avg_ingredients']:.1f}")
        print(f"Avg instructions:     {current['avg_instructions']:.1f}")
    print()

    # Backup (if exists)
    if backup_dirs:
        latest_backup = backup_dirs[-1]
        print(f"BACKUP (from {latest_backup.name})")
        print("-" * 70)
        backup = analyze_recipes(latest_backup)
        if backup:
            print(f"Total recipes:        {backup['total']}")
            print(f"With actual titles:   {backup['with_titles']} ({backup['title_rate']:.1%})")
            print(f"Avg title length:     {backup['avg_title_length']:.1f} characters")
            print(f"Avg ingredients:      {backup['avg_ingredients']:.1f}")
            print(f"Avg instructions:     {backup['avg_instructions']:.1f}")
        print()

        # Comparison
        if current and backup:
            print("IMPROVEMENT")
            print("-" * 70)
            title_diff = current['with_titles'] - backup['with_titles']
            title_rate_diff = current['title_rate'] - backup['title_rate']
            ingredient_diff = current['avg_ingredients'] - backup['avg_ingredients']
            instruction_diff = current['avg_instructions'] - backup['avg_instructions']

            def format_diff(val, suffix="", percent=False):
                if percent:
                    return f"{val:+.1%}" if val != 0 else "±0.0%"
                sign = "+" if val > 0 else ""
                return f"{sign}{val:.1f}{suffix}" if val != 0 else f"±0{suffix}"

            print(f"Titles extracted:     {format_diff(title_diff)} ({format_diff(title_rate_diff, percent=True)})")
            print(f"Avg ingredients:      {format_diff(ingredient_diff)}")
            print(f"Avg instructions:     {format_diff(instruction_diff)}")
            print()

            if title_diff > 0:
                print(f"✓ OCR improvements extracted {title_diff} more recipe titles!")
            elif title_diff < 0:
                print(f"! Warning: {abs(title_diff)} fewer titles extracted")
            else:
                print("= No change in title extraction")
    else:
        print("(No backup found for comparison)")

    print()
    print("=" * 70)

    # Sample some recipes
    if current:
        print()
        print("SAMPLE RECIPES (first 10 with titles)")
        print("-" * 70)
        count = 0
        for recipe_path in sorted(recipes_dir.glob("*.json"))[:50]:
            if count >= 10:
                break
            try:
                with open(recipe_path) as f:
                    recipe = json.load(f)
                title = recipe.get("title", "")
                if title and not title.startswith("Recipe from page"):
                    page = recipe.get("page_num", "?")
                    ingredients = recipe.get("ingredients_lines") or recipe.get("ingredients") or []
                    instructions = recipe.get("instruction_lines") or recipe.get("instructions") or []
                    print(f"  • {title} (page {page})")
                    print(f"    {len(ingredients)} ingredients, {len(instructions)} instructions")
                    count += 1
            except Exception:
                continue
        print()


if __name__ == "__main__":
    main()
