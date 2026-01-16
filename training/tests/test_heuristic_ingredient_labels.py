#!/usr/bin/env python3
"""Unit tests for add_heuristic_ingredient_labels.py."""

import unittest
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import functions from the script
import importlib.util
spec = importlib.util.spec_from_file_location(
    "add_heuristic_ingredient_labels",
    Path(__file__).parent.parent.parent / "scripts" / "add_heuristic_ingredient_labels.py"
)
ingredient_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingredient_module)

is_quantity_line = ingredient_module.is_quantity_line
has_quantity_unit = ingredient_module.has_quantity_unit
is_likely_instruction = ingredient_module.is_likely_instruction
add_heuristic_ingredient_labels = ingredient_module.add_heuristic_ingredient_labels


class TestIngredientHeuristics(unittest.TestCase):
    """Test ingredient detection heuristics."""

    def test_is_quantity_line_detects_integers(self):
        """Test that lines starting with integers are detected."""
        self.assertTrue(is_quantity_line("1 cup sugar"))
        self.assertTrue(is_quantity_line("2 eggs"))
        self.assertTrue(is_quantity_line("100 grams flour"))

    def test_is_quantity_line_detects_fractions(self):
        """Test that lines starting with fractions are detected."""
        self.assertTrue(is_quantity_line("1/2 cup milk"))
        self.assertTrue(is_quantity_line("3/4 teaspoon salt"))
        self.assertTrue(is_quantity_line("2/3 cup butter"))

    def test_is_quantity_line_detects_decimals(self):
        """Test that lines starting with decimals are detected."""
        self.assertTrue(is_quantity_line("0.5 cup water"))
        self.assertTrue(is_quantity_line("1.5 tablespoons vanilla"))
        self.assertTrue(is_quantity_line("2.25 cups flour"))

    def test_is_quantity_line_ignores_non_quantity(self):
        """Test that non-quantity lines are not detected."""
        self.assertFalse(is_quantity_line("Mix well and bake"))
        self.assertFalse(is_quantity_line("Chocolate Cake"))
        self.assertFalse(is_quantity_line("Add the ingredients"))

    def test_has_quantity_unit_detects_cup(self):
        """Test detection of quantity + cup unit."""
        self.assertTrue(has_quantity_unit("Add 1 cup sugar"))
        self.assertTrue(has_quantity_unit("2 cups flour"))
        self.assertTrue(has_quantity_unit("1/2 cup milk"))

    def test_has_quantity_unit_detects_tablespoon(self):
        """Test detection of quantity + tablespoon unit."""
        self.assertTrue(has_quantity_unit("2 tablespoons butter"))
        self.assertTrue(has_quantity_unit("1 tbsp vanilla"))
        self.assertTrue(has_quantity_unit("3 tablespoons oil"))

    def test_has_quantity_unit_detects_teaspoon(self):
        """Test detection of quantity + teaspoon unit."""
        self.assertTrue(has_quantity_unit("1 teaspoon salt"))
        self.assertTrue(has_quantity_unit("2 tsp baking powder"))
        self.assertTrue(has_quantity_unit("1/4 teaspoon nutmeg"))

    def test_has_quantity_unit_detects_ounces_pounds(self):
        """Test detection of quantity + weight units."""
        self.assertTrue(has_quantity_unit("8 ounces chocolate"))
        self.assertTrue(has_quantity_unit("2 oz cream cheese"))
        self.assertTrue(has_quantity_unit("1 pound butter"))
        self.assertTrue(has_quantity_unit("2 lbs flour"))

    def test_has_quantity_unit_detects_metric(self):
        """Test detection of quantity + metric units."""
        self.assertTrue(has_quantity_unit("500 grams sugar"))
        self.assertTrue(has_quantity_unit("2 kg flour"))
        self.assertTrue(has_quantity_unit("250 ml milk"))
        self.assertTrue(has_quantity_unit("1 liter water"))

    def test_has_quantity_unit_ignores_no_unit(self):
        """Test that lines without units are not detected."""
        self.assertFalse(has_quantity_unit("Mix 2 ingredients"))
        self.assertFalse(has_quantity_unit("Bake for 30 minutes"))

    def test_is_likely_instruction_detects_verbs(self):
        """Test that instruction verbs are detected."""
        self.assertTrue(is_likely_instruction("Mix well and bake"))
        self.assertTrue(is_likely_instruction("Add the sugar"))
        self.assertTrue(is_likely_instruction("Bake at 350°F"))
        self.assertTrue(is_likely_instruction("Stir until combined"))

    def test_is_likely_instruction_ignores_ingredients(self):
        """Test that ingredient lines are not detected as instructions."""
        self.assertFalse(is_likely_instruction("1 cup sugar"))
        self.assertFalse(is_likely_instruction("2 eggs, beaten"))
        self.assertFalse(is_likely_instruction("Vanilla extract"))


class TestAddHeuristicIngredientLabels(unittest.TestCase):
    """Test add_heuristic_ingredient_labels function."""

    def test_labels_quantity_line_as_ingredient(self):
        """Test that lines starting with quantities are labeled as INGREDIENT_LINE."""
        page = {
            "words": ["1", "cup", "sugar"],
            "bboxes": [[0, 300, 50, 320], [60, 300, 100, 320], [110, 300, 160, 320]],  # y=300 (30% down, not title area)
            "labels": ["O", "O", "O"],
            "height": 1000,
        }

        result = add_heuristic_ingredient_labels(page)

        # Should label all tokens in the ingredient line
        self.assertEqual(result["labels"], ["INGREDIENT_LINE", "INGREDIENT_LINE", "INGREDIENT_LINE"])

    def test_does_not_label_instruction_as_ingredient(self):
        """Test that instruction lines are not labeled as INGREDIENT_LINE."""
        page = {
            "words": ["Mix", "well", "and", "bake"],
            "bboxes": [[0, 300, 50, 320], [60, 300, 100, 320], [110, 300, 150, 320], [160, 300, 200, 320]],
            "labels": ["O", "O", "O", "O"],
            "height": 1000,
        }

        result = add_heuristic_ingredient_labels(page)

        # Should NOT label instruction as ingredient
        self.assertEqual(result["labels"], ["O", "O", "O", "O"])

    def test_preserves_existing_labels(self):
        """Test that existing non-O labels are preserved."""
        page = {
            "words": ["1", "cup", "sugar"],
            "bboxes": [[0, 300, 50, 320], [60, 300, 100, 320], [110, 300, 160, 320]],
            "labels": ["TITLE", "O", "O"],  # First token already labeled
            "height": 1000,
        }

        result = add_heuristic_ingredient_labels(page)

        # Should preserve TITLE label, add INGREDIENT_LINE to O tokens
        self.assertEqual(result["labels"], ["TITLE", "INGREDIENT_LINE", "INGREDIENT_LINE"])

    def test_ingredient_block_detection(self):
        """Test that ingredient block is detected and labeled."""
        # Simulate a page with ingredient block followed by instructions
        page = {
            "words": [
                # Line 1: quantity line (ingredient)
                "1", "cup", "sugar",
                # Line 2: quantity line (ingredient)
                "2", "eggs",
                # Line 3: instruction (ends block)
                "Mix", "well"
            ],
            "bboxes": [
                # Line 1 (y=300)
                [0, 300, 20, 320], [30, 300, 60, 320], [70, 300, 120, 320],
                # Line 2 (y=350)
                [0, 350, 20, 370], [30, 350, 60, 370],
                # Line 3 (y=400) - instruction
                [0, 400, 40, 420], [50, 400, 80, 420],
            ],
            "labels": ["O"] * 7,
            "height": 1000,
        }

        result = add_heuristic_ingredient_labels(page)

        # Lines 1 and 2 should be labeled as INGREDIENT_LINE
        # Line 3 (instruction) should remain O
        expected = [
            "INGREDIENT_LINE", "INGREDIENT_LINE", "INGREDIENT_LINE",  # Line 1
            "INGREDIENT_LINE", "INGREDIENT_LINE",  # Line 2
            "O", "O",  # Line 3 (instruction)
        ]
        self.assertEqual(result["labels"], expected)

    def test_dry_run_mode(self):
        """Test that dry_run mode computes stats without modifying labels."""
        page = {
            "words": ["1", "cup", "sugar"],
            "bboxes": [[0, 300, 50, 320], [60, 300, 100, 320], [110, 300, 160, 320]],
            "labels": ["O", "O", "O"],
            "height": 1000,
        }

        result = add_heuristic_ingredient_labels(page, dry_run=True)

        # Labels should NOT be modified in dry_run mode
        self.assertEqual(result["labels"], ["O", "O", "O"])

        # But stats should be computed
        self.assertGreater(result["heuristic_labeling"]["ingredient_labels_added"], 0)

    def test_tracks_stats(self):
        """Test that heuristic labeling stats are tracked."""
        page = {
            "words": ["1", "cup", "sugar"],
            "bboxes": [[0, 300, 50, 320], [60, 300, 100, 320], [110, 300, 160, 320]],
            "labels": ["O", "O", "O"],
            "height": 1000,
        }

        result = add_heuristic_ingredient_labels(page)

        # Should have stats metadata
        self.assertIn("heuristic_labeling", result)
        self.assertEqual(result["heuristic_labeling"]["ingredient_labels_added"], 3)
        self.assertEqual(result["heuristic_labeling"]["ingredient_lines_labeled"], 1)


if __name__ == "__main__":
    unittest.main()
