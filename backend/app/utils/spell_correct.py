"""
Spell correction for recipe text.
Focuses on correcting common OCR errors in recipe titles, ingredients, and instructions.
"""
import re
from typing import Optional


# Common OCR misreadings for TITLES
OCR_CORRECTIONS_TITLE = {
    # B/E confusion
    "blection": "election",
    "Blection": "Election",

    # I/l confusion
    "Chlcken": "Chicken",
    "chlcken": "chicken",
    "Mlk": "Milk",
    "mlk": "milk",

    # O/0 confusion
    "C0ffee": "Coffee",
    "c0ffee": "coffee",

    # Common food words
    "chese": "cheese",
    "Chese": "Cheese",
    "tost": "toast",
    "Tost": "Toast",
    "bred": "bread",
    "Bred": "Bread",
    "sause": "sauce",
    "Sause": "Sauce",
    "peper": "pepper",
    "Peper": "Pepper",
}

# Common OCR misreadings for INGREDIENTS and INSTRUCTIONS
OCR_CORRECTIONS_INGREDIENT = {
    # Common measurement errors
    "Loup": "cup",  # "2 Loup" -> "2 cup"
    "loup": "cup",
    "Ibs": "lbs",  # "10 Ibs" -> "10 lbs" (capital I instead of lowercase l)
    "Ibs.": "lbs.",
    "Legg": "egg",  # "1 Legg" -> "1 egg"
    "legg": "egg",

    # Common ingredient words
    "flonr": "flour",
    "sngar": "sugar",
    "augar": "sugar",
    "milh": "milk",
    "bntter": "butter",
    "bnter": "butter",
    "hutter": "butter",
    "butier": "butter",
    "chese": "cheese",
    "chesse": "cheese",
    "bred": "bread",
    "hread": "bread",
    "sait": "salt",
    "aalt": "salt",
    "peper": "pepper",
    "pcpper": "pepper",
    "egga": "eggs",
    "egs": "eggs",
    "oggs": "eggs",
    "wafer": "water",
    "wator": "water",

    # Cooking terms
    "bakiug": "baking",
    "bonl": "boil",
    "boll": "boil",
    "etir": "stir",
    "atlr": "stir",
    "edd": "add",
    "ndd": "add",
    "hest": "heat",
    "hcat": "heat",
    "conk": "cook",
    "cooh": "cook",
    "ponr": "pour",
    "ponnd": "pound",
    "qnart": "quart",
    "onart": "quart",

    # Capitalized versions
    "Flonr": "Flour",
    "Sngar": "Sugar",
    "Augar": "Sugar",
    "Milh": "Milk",
    "Bntter": "Butter",
    "Bnter": "Butter",
    "Hutter": "Butter",
    "Butier": "Butter",
    "Chese": "Cheese",
    "Chesse": "Cheese",
    "Bred": "Bread",
    "Hread": "Bread",
    "Sait": "Salt",
    "Aalt": "Salt",
    "Peper": "Pepper",
    "Pcpper": "Pepper",
    "Wafer": "Water",
    "Wator": "Water",
}


def correct_title(title: str) -> str:
    """
    Apply spell correction to a recipe title.

    Args:
        title: The recipe title to correct

    Returns:
        Corrected title
    """
    if not title:
        return title

    # Split into words
    words = title.split()
    corrected_words = []

    for word in words:
        # Check if word (without punctuation) is in corrections
        word_clean = word.rstrip('.,;:!?')
        punct = word[len(word_clean):] if len(word) > len(word_clean) else ""

        if word_clean in OCR_CORRECTIONS_TITLE:
            corrected_words.append(OCR_CORRECTIONS_TITLE[word_clean] + punct)
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)


def correct_ingredient(text: str) -> str:
    """
    Apply spell correction to ingredient or instruction text.

    Args:
        text: The ingredient or instruction text to correct

    Returns:
        Corrected text
    """
    if not text:
        return text

    # Split into words
    words = text.split()
    corrected_words = []

    for word in words:
        # Check if word (without punctuation) is in corrections
        word_clean = word.rstrip('.,;:!?')
        punct = word[len(word_clean):] if len(word) > len(word_clean) else ""

        if word_clean in OCR_CORRECTIONS_INGREDIENT:
            corrected_words.append(OCR_CORRECTIONS_INGREDIENT[word_clean] + punct)
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)


def add_correction(incorrect: str, correct: str, correction_type: str = "title") -> None:
    """
    Add a new correction to the dictionary.

    Args:
        incorrect: The incorrect OCR text
        correct: The correct text
        correction_type: Either "title" or "ingredient"
    """
    if correction_type == "title":
        OCR_CORRECTIONS_TITLE[incorrect] = correct
    elif correction_type == "ingredient":
        OCR_CORRECTIONS_INGREDIENT[incorrect] = correct
