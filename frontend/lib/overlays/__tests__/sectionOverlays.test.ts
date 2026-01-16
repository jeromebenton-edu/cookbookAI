/**
 * Unit tests for section overlay aggregation logic
 */

import { describe, it, expect } from "@jest/globals";
import { buildSectionOverlays } from "../sectionOverlays";
import type { RecipeToken } from "../../types";

describe("buildSectionOverlays", () => {
  describe("multi-recipe page filtering", () => {
    it("should only include overlays for the selected recipe (Waffles)", () => {
      // Simulate a page with 3 recipes: Bread Griddle-Cakes, Buckwheat Cakes, and Waffles
      const tokens: RecipeToken[] = [
        // Recipe 1: Bread Griddle-Cakes (top of page)
        {
          id: "title_1",
          text: "Bread Griddle-Cakes",
          label: "TITLE",
          score: 0.96,
          bbox: [150, 50, 400, 90],
        },
        {
          id: "ing_1_1",
          text: "1.5 cups bread crumbs",
          label: "INGREDIENT_LINE",
          score: 0.92,
          bbox: [100, 110, 350, 140],
        },
        {
          id: "ing_1_2",
          text: "2 eggs",
          label: "INGREDIENT_LINE",
          score: 0.94,
          bbox: [400, 110, 550, 140],
        },

        // Recipe 2: Buckwheat Cakes (middle)
        {
          id: "title_2",
          text: "Buckwheat Cakes",
          label: "TITLE",
          score: 0.95,
          bbox: [150, 250, 380, 290],
        },
        {
          id: "ing_2_1",
          text: "0.5 cup bread crumbs",
          label: "INGREDIENT_LINE",
          score: 0.91,
          bbox: [100, 310, 350, 340],
        },
        {
          id: "inst_2_1",
          text: "Pour milk over crumbs",
          label: "INSTRUCTION_STEP",
          score: 0.89,
          bbox: [100, 380, 500, 420],
        },

        // Recipe 3: Waffles (bottom)
        {
          id: "title_3",
          text: "Waffles",
          label: "TITLE",
          score: 0.97,
          bbox: [150, 500, 320, 540],
        },
        {
          id: "ing_3_1",
          text: "2 cups flour",
          label: "INGREDIENT_LINE",
          score: 0.93,
          bbox: [100, 560, 350, 590],
        },
        {
          id: "ing_3_2",
          text: "3 tsp baking powder",
          label: "INGREDIENT_LINE",
          score: 0.95,
          bbox: [100, 600, 350, 630],
        },
        {
          id: "inst_3_1",
          text: "Mix and sift dry ingredients",
          label: "INSTRUCTION_STEP",
          score: 0.92,
          bbox: [100, 680, 500, 720],
        },
        {
          id: "inst_3_2",
          text: "Cook on a greased hot waffle iron",
          label: "INSTRUCTION_STEP",
          score: 0.94,
          bbox: [100, 730, 500, 770],
        },
      ];

      const result = buildSectionOverlays(tokens, "Waffles");

      // Title should match Waffles
      expect(result.titleBox).toBeDefined();
      expect(result.titleBox?.tokenIds).toEqual(["title_3"]);

      // Ingredients should only include Waffles ingredients (2 items)
      expect(result.ingredientBoxes).toHaveLength(1);
      expect(result.ingredientBoxes[0].tokenIds).toEqual(["ing_3_1", "ing_3_2"]);

      // Instructions should only include Waffles instructions (2 items)
      expect(result.instructionBox).toBeDefined();
      expect(result.instructionBox?.tokenIds).toEqual(["inst_3_1", "inst_3_2"]);

      // Debug info should confirm filtering
      expect(result.debug?.filteredIngredientCount).toBe(2);
      expect(result.debug?.filteredInstructionCount).toBe(2);
    });

    it("should exclude tokens from recipe below the selected one", () => {
      const tokens: RecipeToken[] = [
        {
          id: "title_1",
          text: "First Recipe",
          label: "TITLE",
          score: 0.95,
          bbox: [100, 50, 300, 90],
        },
        {
          id: "ing_1",
          text: "ingredient for first",
          label: "INGREDIENT_LINE",
          score: 0.92,
          bbox: [100, 110, 350, 140],
        },
        {
          id: "title_2",
          text: "Second Recipe",
          label: "TITLE",
          score: 0.94,
          bbox: [100, 200, 320, 240],
        },
        {
          id: "ing_2",
          text: "ingredient for second (should be excluded)",
          label: "INGREDIENT_LINE",
          score: 0.91,
          bbox: [100, 260, 350, 290],
        },
      ];

      const result = buildSectionOverlays(tokens, "First Recipe");

      // Should only include ingredients before the next title
      expect(result.ingredientBoxes).toHaveLength(1);
      expect(result.ingredientBoxes[0].tokenIds).toEqual(["ing_1"]);
      expect(result.debug?.nextTitleYPosition).toBe(200); // Top of "Second Recipe" title
    });
  });

  describe("two-column ingredient detection", () => {
    it("should split ingredients into 2 boxes when spread exceeds threshold", () => {
      const tokens: RecipeToken[] = [
        {
          id: "title",
          text: "Recipe",
          label: "TITLE",
          score: 0.95,
          bbox: [200, 50, 400, 90],
        },
        // Left column ingredients (x: 100-350)
        {
          id: "ing_1",
          text: "2 cups flour",
          label: "INGREDIENT_LINE",
          score: 0.92,
          bbox: [100, 110, 350, 140],
        },
        {
          id: "ing_2",
          text: "1 tsp salt",
          label: "INGREDIENT_LINE",
          score: 0.93,
          bbox: [100, 150, 350, 180],
        },
        // Right column ingredients (x: 450-700)
        {
          id: "ing_3",
          text: "2 eggs",
          label: "INGREDIENT_LINE",
          score: 0.91,
          bbox: [450, 110, 700, 140],
        },
        {
          id: "ing_4",
          text: "1 cup milk",
          label: "INGREDIENT_LINE",
          score: 0.94,
          bbox: [450, 150, 700, 180],
        },
      ];

      const result = buildSectionOverlays(tokens, "Recipe", {
        twoColumnThreshold: 250, // x spread is 600, which > 250
      });

      // Should detect two columns
      expect(result.ingredientBoxes).toHaveLength(2);

      // Left column should contain ing_1 and ing_2
      const leftBox = result.ingredientBoxes.find((box) => box.tokenIds.includes("ing_1"));
      expect(leftBox?.tokenIds).toEqual(["ing_1", "ing_2"]);

      // Right column should contain ing_3 and ing_4
      const rightBox = result.ingredientBoxes.find((box) => box.tokenIds.includes("ing_3"));
      expect(rightBox?.tokenIds).toEqual(["ing_3", "ing_4"]);
    });

    it("should use single box when spread is below threshold", () => {
      const tokens: RecipeToken[] = [
        {
          id: "title",
          text: "Recipe",
          label: "TITLE",
          score: 0.95,
          bbox: [200, 50, 400, 90],
        },
        {
          id: "ing_1",
          text: "2 cups flour",
          label: "INGREDIENT_LINE",
          score: 0.92,
          bbox: [100, 110, 350, 140],
        },
        {
          id: "ing_2",
          text: "1 tsp salt",
          label: "INGREDIENT_LINE",
          score: 0.93,
          bbox: [120, 150, 370, 180],
        },
      ];

      const result = buildSectionOverlays(tokens, "Recipe", {
        twoColumnThreshold: 250, // x spread is only ~20, which < 250
      });

      // Should use single column
      expect(result.ingredientBoxes).toHaveLength(1);
      expect(result.ingredientBoxes[0].tokenIds).toEqual(["ing_1", "ing_2"]);
    });
  });

  describe("missing title handling", () => {
    it("should return empty overlays when no titles exist", () => {
      const tokens: RecipeToken[] = [
        {
          id: "ing_1",
          text: "2 cups flour",
          label: "INGREDIENT_LINE",
          score: 0.92,
          bbox: [100, 110, 350, 140],
        },
      ];

      const result = buildSectionOverlays(tokens, "Nonexistent Recipe");

      expect(result.titleBox).toBeUndefined();
      expect(result.ingredientBoxes).toEqual([]);
      expect(result.instructionBox).toBeUndefined();
    });

    it("should use fallback title when exact match not found", () => {
      const tokens: RecipeToken[] = [
        {
          id: "title_1",
          text: "Different Recipe",
          label: "TITLE",
          score: 0.98,
          bbox: [100, 50, 300, 90],
        },
        {
          id: "title_2",
          text: "Another Recipe",
          label: "TITLE",
          score: 0.85,
          bbox: [100, 200, 300, 240],
        },
      ];

      const result = buildSectionOverlays(tokens, "Nonexistent Recipe");

      // Should fall back to highest confidence title (title_1 with 0.98)
      expect(result.titleBox).toBeDefined();
      expect(result.titleBox?.tokenIds).toEqual(["title_1"]);
    });
  });

  describe("bounding box union and padding", () => {
    it("should union multiple boxes with padding", () => {
      const tokens: RecipeToken[] = [
        {
          id: "title",
          text: "Recipe",
          label: "TITLE",
          score: 0.95,
          bbox: [200, 50, 400, 90],
        },
        {
          id: "ing_1",
          text: "ingredient 1",
          label: "INGREDIENT_LINE",
          score: 0.92,
          bbox: [100, 110, 300, 140],
        },
        {
          id: "ing_2",
          text: "ingredient 2",
          label: "INGREDIENT_LINE",
          score: 0.93,
          bbox: [120, 150, 350, 180],
        },
      ];

      const result = buildSectionOverlays(tokens, "Recipe", {
        padding: 10,
      });

      // Ingredient box should be union of [100,110,300,140] and [120,150,350,180]
      // Union: [100, 110, 350, 180]
      // With padding: [90, 100, 360, 190]
      expect(result.ingredientBoxes).toHaveLength(1);
      expect(result.ingredientBoxes[0].bbox).toEqual([90, 100, 360, 190]);
    });
  });

  describe("confidence aggregation", () => {
    it("should compute average confidence for section boxes", () => {
      const tokens: RecipeToken[] = [
        {
          id: "title",
          text: "Recipe",
          label: "TITLE",
          score: 0.97,
          bbox: [200, 50, 400, 90],
        },
        {
          id: "ing_1",
          text: "ingredient 1",
          label: "INGREDIENT_LINE",
          score: 0.9,
          bbox: [100, 110, 300, 140],
        },
        {
          id: "ing_2",
          text: "ingredient 2",
          label: "INGREDIENT_LINE",
          score: 0.8,
          bbox: [100, 150, 300, 180],
        },
        {
          id: "inst_1",
          text: "instruction 1",
          label: "INSTRUCTION_STEP",
          score: 0.85,
          bbox: [100, 200, 400, 240],
        },
      ];

      const result = buildSectionOverlays(tokens, "Recipe");

      // Title confidence = 0.97 (single token)
      expect(result.titleBox?.confidence).toBe(0.97);

      // Ingredients confidence = average of 0.9 and 0.8 = 0.85
      expect(result.ingredientBoxes[0].confidence).toBeCloseTo(0.85);

      // Instructions confidence = 0.85 (single token)
      expect(result.instructionBox?.confidence).toBe(0.85);
    });
  });
});
