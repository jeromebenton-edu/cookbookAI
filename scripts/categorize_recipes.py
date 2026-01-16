#!/usr/bin/env python3
"""
Categorize recipes from the 1896 Boston Cooking-School Cook Book
Assigns categories and tags based on recipe titles and ingredients.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set

# Category mapping based on keywords in title/ingredients
CATEGORY_KEYWORDS = {
    "Beverages": [
        "coffee", "tea", "chocolate", "cocoa", "lemonade", "punch", "juice",
        "milk shake", "eggnog", "grape juice"
    ],
    "Breads & Biscuits": [
        "bread", "biscuit", "muffin", "roll", "bun", "pop-over", "scone",
        "johnny cake", "corn bread"
    ],
    "Cakes & Cookies": [
        "cake", "cookie", "macaroon", "gingerbread", "shortbread", "wafer",
        "lady finger", "sponge"
    ],
    "Desserts & Sweets": [
        "pie", "pudding", "ice cream", "sherbet", "custard", "jelly",
        "blanc mange", "mousse", "bavarian", "trifle", "cream", "charlotte",
        "fritter", "doughnut", "waffle", "pancake"
    ],
    "Fish & Seafood": [
        "fish", "salmon", "halibut", "cod", "haddock", "bluefish", "shad",
        "lobster", "oyster", "clam", "crab", "scallop", "shrimp"
    ],
    "Meats & Poultry": [
        "beef", "veal", "mutton", "lamb", "pork", "ham", "bacon",
        "chicken", "turkey", "duck", "goose", "fowl", "venison"
    ],
    "Soups & Broths": [
        "soup", "broth", "stock", "consomme", "bisque", "chowder", "bouillon"
    ],
    "Preserves & Canning": [
        "jelly", "jam", "marmalade", "preserve", "pickle", "canned", "bottled",
        "conserve"
    ],
    "Medicinal Cookery": [
        "invalid", "gruel", "albumeni", "beef essence", "barley water",
        "toast water", "junket", "creature food", "sick"
    ],
    "Vegetables & Sides": [
        "potato", "tomato", "asparagus", "peas", "beans", "corn", "squash",
        "cabbage", "cauliflower", "spinach", "onion", "carrot", "turnip",
        "beet", "celery", "rice", "macaroni"
    ],
    "Sauces & Condiments": [
        "sauce", "gravy", "dressing", "mayonnaise", "vinaigrette", "butter"
    ],
    "Salads": [
        "salad"
    ],
    "Eggs & Cheese": [
        "omelet", "omelette", "scrambled egg", "poached egg", "shirred egg",
        "cheese", "rarebit", "fondue"
    ]
}

# Tag keywords
TAG_KEYWORDS = {
    "baking": ["bake", "baked", "oven"],
    "frying": ["fry", "fried", "fritter"],
    "boiling": ["boil", "boiled", "poach"],
    "roasting": ["roast", "roasted"],
    "preserving": ["preserve", "pickle", "can", "bottle", "jelly", "jam"],
    "breakfast": ["pancake", "waffle", "muffin", "coffee", "tea", "egg"],
    "dessert": ["cake", "pie", "pudding", "ice cream", "cookie"],
    "beverage": ["coffee", "tea", "chocolate", "juice", "punch"],
    "medicinal": ["invalid", "gruel", "essence", "barley water", "albumeni"],
    "vegetarian": [],  # Will be determined by absence of meat keywords
}

MEAT_KEYWORDS = [
    "beef", "veal", "mutton", "lamb", "pork", "ham", "bacon",
    "chicken", "turkey", "duck", "goose", "fowl", "fish", "salmon",
    "lobster", "oyster", "clam", "crab", "shrimp", "meat"
]


def normalize_text(text: str) -> str:
    """Normalize text for matching."""
    return text.lower().strip()


def categorize_recipe(title: str, ingredients: List[str]) -> str:
    """Determine category based on title and ingredients."""
    title_lower = normalize_text(title)
    ingredients_text = " ".join([normalize_text(ing) for ing in ingredients])

    # Prioritize title matching over ingredients
    # This gives better results for most recipes

    # Check title first (more accurate)
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in title_lower:
                return category

    # Then check combined text
    combined_text = f"{title_lower} {ingredients_text}"
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in combined_text:
                return category

    # Default category
    return "Historical"


def generate_tags(title: str, ingredients: List[str], category: str) -> List[str]:
    """Generate tags based on content."""
    tags = ["1896"]

    title_lower = normalize_text(title)
    ingredients_text = " ".join([normalize_text(ing) for ing in ingredients])
    combined_text = f"{title_lower} {ingredients_text}"

    # Add technique tags
    for tag, keywords in TAG_KEYWORDS.items():
        if tag == "vegetarian":
            continue
        for keyword in keywords:
            if keyword in combined_text:
                if tag not in tags:
                    tags.append(tag)
                break

    # Check if vegetarian (no meat keywords)
    has_meat = any(meat in combined_text for meat in MEAT_KEYWORDS)
    if not has_meat and category not in ["Fish & Seafood", "Meats & Poultry"]:
        tags.append("vegetarian")

    # Add category-specific tags
    if category == "Medicinal Cookery":
        if "medicinal" not in tags:
            tags.append("medicinal")

    return tags


def main():
    """Categorize all recipes."""
    recipes_dir = Path(__file__).parent.parent / "frontend" / "public" / "recipes" / "boston"

    if not recipes_dir.exists():
        print(f"Error: Recipes directory not found: {recipes_dir}")
        return

    recipe_files = sorted(recipes_dir.glob("*.json"))
    updated_count = 0

    for recipe_file in recipe_files:
        if recipe_file.name == "index.json":
            continue

        try:
            with open(recipe_file, 'r', encoding='utf-8') as f:
                recipe = json.load(f)

            title = recipe.get("title", "")
            ingredients = recipe.get("ingredients", [])

            # Skip chapter markers
            if "chapter" in recipe.get("id", "").lower():
                continue

            # Determine category and tags
            new_category = categorize_recipe(title, ingredients)
            new_tags = generate_tags(title, ingredients, new_category)

            # Update recipe
            old_category = recipe.get("category", "")
            old_tags = recipe.get("tags", [])

            recipe["category"] = new_category
            recipe["tags"] = new_tags

            # Write back
            with open(recipe_file, 'w', encoding='utf-8') as f:
                json.dump(recipe, f, indent=2, ensure_ascii=False)

            if old_category != new_category or set(old_tags) != set(new_tags):
                updated_count += 1
                print(f"Updated: {recipe_file.name}")
                print(f"  Category: {old_category} -> {new_category}")
                print(f"  Tags: {old_tags} -> {new_tags}")

        except Exception as e:
            print(f"Error processing {recipe_file.name}: {e}")

    print(f"\nUpdated {updated_count} recipes")

    # Update index.json with new category/tag lists
    index_file = recipes_dir / "index.json"
    if index_file.exists():
        with open(index_file, 'r', encoding='utf-8') as f:
            index_data = json.load(f)

        # Collect all unique categories and tags
        all_categories: Set[str] = set()
        all_tags: Set[str] = set()

        for recipe_file in recipe_files:
            if recipe_file.name == "index.json":
                continue
            try:
                with open(recipe_file, 'r', encoding='utf-8') as f:
                    recipe = json.load(f)
                    all_categories.add(recipe.get("category", "Historical"))
                    all_tags.update(recipe.get("tags", []))
            except:
                pass

        index_data["categories"] = sorted(all_categories)
        index_data["tags"] = sorted(all_tags)

        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)

        print(f"\nUpdated index.json with {len(all_categories)} categories and {len(all_tags)} tags")


if __name__ == "__main__":
    main()
