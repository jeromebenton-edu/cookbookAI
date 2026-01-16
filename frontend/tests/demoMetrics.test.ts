import assert from "node:assert/strict";
import { test } from "node:test";
import { computeDemoMetrics, formatSectionStatus, getSectionStatusClass } from "../lib/demoMetrics";
import type { ExtractedRecipe } from "../lib/types";

test("computeDemoMetrics - should return all missing for null prediction", () => {
  const metrics = computeDemoMetrics(null);

  assert.equal(metrics.sectionStatus.title, "missing");
  assert.equal(metrics.sectionStatus.ingredients, "missing");
  assert.equal(metrics.sectionStatus.instructions, "missing");
  assert.equal(metrics.coverage.ingredientCount, 0);
  assert.equal(metrics.coverage.stepCount, 0);
  assert.equal(metrics.pageReadiness, "review");
});

test("computeDemoMetrics - should return high-confidence for perfect prediction", () => {
  const prediction: ExtractedRecipe = {
    page_num: 1,
    title: "Test Recipe",
    ingredients: ["1 cup flour", "2 eggs", "1 tsp salt", "1 cup milk"],
    instructions: ["Mix ingredients", "Bake at 350F", "Cool before serving"],
    confidence: {
      title: 0.95,
      ingredients: 0.92,
      instructions: 0.90,
      overall: 0.92,
    },
    is_recipe_page: true,
    recipe_confidence: 0.93,
    meta: {},
  };

  const metrics = computeDemoMetrics(prediction);

  assert.equal(metrics.sectionStatus.title, "high-confidence");
  assert.equal(metrics.sectionStatus.ingredients, "high-confidence");
  assert.equal(metrics.sectionStatus.instructions, "high-confidence");
  assert.equal(metrics.coverage.ingredientCount, 4);
  assert.equal(metrics.coverage.stepCount, 3);
  assert.equal(metrics.pageReadiness, "ready");
});

test("computeDemoMetrics - should return needs-review for low confidence", () => {
  const prediction: ExtractedRecipe = {
    page_num: 1,
    title: "Test Recipe",
    ingredients: ["1 cup flour", "2 eggs", "1 tsp salt"],
    instructions: ["Mix ingredients", "Bake at 350F"],
    confidence: {
      title: 0.75, // Below 0.85 threshold
      ingredients: 0.78, // Below 0.80 threshold
      instructions: 0.82, // Above 0.80 threshold
      overall: 0.78,
    },
    is_recipe_page: true,
    recipe_confidence: 0.80,
    meta: {},
  };

  const metrics = computeDemoMetrics(prediction);

  assert.equal(metrics.sectionStatus.title, "needs-review");
  assert.equal(metrics.sectionStatus.ingredients, "needs-review");
  assert.equal(metrics.sectionStatus.instructions, "high-confidence");
  assert.equal(metrics.pageReadiness, "review"); // Not all high-confidence
});

test("computeDemoMetrics - should return needs-review for insufficient ingredient count", () => {
  const prediction: ExtractedRecipe = {
    page_num: 1,
    title: "Test Recipe",
    ingredients: ["1 cup flour", "2 eggs"], // Only 2, needs 3+
    instructions: ["Mix ingredients", "Bake at 350F"],
    confidence: {
      title: 0.95,
      ingredients: 0.92, // High confidence but fails sanity check
      instructions: 0.90,
      overall: 0.92,
    },
    is_recipe_page: true,
    recipe_confidence: 0.93,
    meta: {},
  };

  const metrics = computeDemoMetrics(prediction);

  assert.equal(metrics.sectionStatus.title, "high-confidence");
  assert.equal(metrics.sectionStatus.ingredients, "needs-review"); // Fails sanity check
  assert.equal(metrics.sectionStatus.instructions, "high-confidence");
  assert.equal(metrics.coverage.ingredientCount, 2);
  assert.equal(metrics.pageReadiness, "review");
});

test("computeDemoMetrics - should return needs-review for insufficient step count", () => {
  const prediction: ExtractedRecipe = {
    page_num: 1,
    title: "Test Recipe",
    ingredients: ["1 cup flour", "2 eggs", "1 tsp salt"],
    instructions: ["Mix ingredients"], // Only 1, needs 2+
    confidence: {
      title: 0.95,
      ingredients: 0.92,
      instructions: 0.90, // High confidence but fails sanity check
      overall: 0.92,
    },
    is_recipe_page: true,
    recipe_confidence: 0.93,
    meta: {},
  };

  const metrics = computeDemoMetrics(prediction);

  assert.equal(metrics.sectionStatus.title, "high-confidence");
  assert.equal(metrics.sectionStatus.ingredients, "high-confidence");
  assert.equal(metrics.sectionStatus.instructions, "needs-review"); // Fails sanity check
  assert.equal(metrics.coverage.stepCount, 1);
  assert.equal(metrics.pageReadiness, "review");
});

test("computeDemoMetrics - should return missing for empty sections", () => {
  const prediction: ExtractedRecipe = {
    page_num: 1,
    title: "",
    ingredients: [],
    instructions: [],
    confidence: {
      title: 0,
      ingredients: 0,
      instructions: 0,
      overall: 0,
    },
    is_recipe_page: false,
    meta: {},
  };

  const metrics = computeDemoMetrics(prediction);

  assert.equal(metrics.sectionStatus.title, "missing");
  assert.equal(metrics.sectionStatus.ingredients, "missing");
  assert.equal(metrics.sectionStatus.instructions, "missing");
  assert.equal(metrics.pageReadiness, "review");
});

test("computeDemoMetrics - should handle zero confidence as low confidence", () => {
  const prediction: ExtractedRecipe = {
    page_num: 1,
    title: "Test Recipe",
    ingredients: ["1 cup flour", "2 eggs", "1 tsp salt"],
    instructions: ["Mix ingredients", "Bake at 350F"],
    confidence: {
      title: 0,
      ingredients: 0,
      instructions: 0,
      overall: 0,
    },
    is_recipe_page: true,
    meta: {},
  };

  const metrics = computeDemoMetrics(prediction);

  // Zero confidence should be treated as needs-review
  assert.equal(metrics.sectionStatus.title, "needs-review");
  assert.equal(metrics.sectionStatus.ingredients, "needs-review");
  assert.equal(metrics.sectionStatus.instructions, "needs-review");
  assert.equal(metrics.pageReadiness, "review");
});

test("formatSectionStatus - should format status labels correctly", () => {
  assert.equal(formatSectionStatus("high-confidence"), "High confidence");
  assert.equal(formatSectionStatus("needs-review"), "Needs review");
  assert.equal(formatSectionStatus("missing"), "Missing");
});

test("getSectionStatusClass - should return correct CSS classes", () => {
  assert.ok(getSectionStatusClass("high-confidence").includes("green"));
  assert.ok(getSectionStatusClass("needs-review").includes("yellow"));
  assert.ok(getSectionStatusClass("missing").includes("gray"));
});
