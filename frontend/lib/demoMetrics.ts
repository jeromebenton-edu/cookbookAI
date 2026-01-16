/**
 * Demo metrics computation
 *
 * Computes visible metrics for the demo product mode.
 * All heuristic thresholds and logic are centralized here.
 */

import type { ExtractedRecipe } from "./types";

export type SectionStatus = "high-confidence" | "needs-review" | "missing";

export type DemoMetrics = {
  sectionStatus: {
    title: SectionStatus;
    ingredients: SectionStatus;
    instructions: SectionStatus;
  };
  coverage: {
    ingredientCount: number;
    stepCount: number;
  };
  pageReadiness: "ready" | "review";
};

/**
 * Confidence thresholds for section status
 */
const THRESHOLDS = {
  title: 0.85,
  ingredients: 0.80,
  instructions: 0.80,
} as const;

/**
 * Sanity check thresholds for "ready" status
 */
const SANITY_CHECKS = {
  minIngredients: 3,
  minSteps: 2,
} as const;

/**
 * Determine section status based on presence, confidence, and sanity checks
 */
function computeSectionStatus(
  present: boolean,
  confidence: number | null | undefined,
  threshold: number,
  count?: number,
  minCount?: number
): SectionStatus {
  // Missing if not present or empty
  if (!present) {
    return "missing";
  }

  // If count-based sanity check fails, needs review
  if (minCount !== undefined && count !== undefined && count < minCount) {
    return "needs-review";
  }

  // If we have confidence, check threshold
  if (confidence !== null && confidence !== undefined) {
    if (confidence < threshold) {
      return "needs-review";
    }
    return "high-confidence";
  }

  // No confidence signal - base on presence and sanity checks only
  // If we got here, presence is true and sanity checks passed (or N/A)
  return "high-confidence";
}

/**
 * Compute all demo metrics from a prediction
 */
export function computeDemoMetrics(prediction: ExtractedRecipe | null): DemoMetrics {
  // Handle null/missing prediction
  if (!prediction) {
    return {
      sectionStatus: {
        title: "missing",
        ingredients: "missing",
        instructions: "missing",
      },
      coverage: {
        ingredientCount: 0,
        stepCount: 0,
      },
      pageReadiness: "review",
    };
  }

  // Extract coverage counts
  const ingredientCount = prediction.ingredients?.length ?? 0;
  const stepCount = prediction.instructions?.length ?? 0;

  // Compute section statuses
  const titlePresent = Boolean(prediction.title && prediction.title.trim().length > 0);
  const ingredientsPresent = ingredientCount > 0;
  const instructionsPresent = stepCount > 0;

  const titleStatus = computeSectionStatus(
    titlePresent,
    prediction.confidence?.title,
    THRESHOLDS.title
  );

  const ingredientsStatus = computeSectionStatus(
    ingredientsPresent,
    prediction.confidence?.ingredients,
    THRESHOLDS.ingredients,
    ingredientCount,
    SANITY_CHECKS.minIngredients
  );

  const instructionsStatus = computeSectionStatus(
    instructionsPresent,
    prediction.confidence?.instructions,
    THRESHOLDS.instructions,
    stepCount,
    SANITY_CHECKS.minSteps
  );

  // Page readiness: "ready" only if all sections are high-confidence
  const allHighConfidence =
    titleStatus === "high-confidence" &&
    ingredientsStatus === "high-confidence" &&
    instructionsStatus === "high-confidence";

  const pageReadiness: "ready" | "review" = allHighConfidence ? "ready" : "review";

  return {
    sectionStatus: {
      title: titleStatus,
      ingredients: ingredientsStatus,
      instructions: instructionsStatus,
    },
    coverage: {
      ingredientCount,
      stepCount,
    },
    pageReadiness,
  };
}

/**
 * Format section status as user-friendly label
 */
export function formatSectionStatus(status: SectionStatus): string {
  switch (status) {
    case "high-confidence":
      return "High confidence";
    case "needs-review":
      return "Needs review";
    case "missing":
      return "Missing";
  }
}

/**
 * Get CSS class for section status chip
 */
export function getSectionStatusClass(status: SectionStatus): string {
  switch (status) {
    case "high-confidence":
      return "bg-green-50 text-green-700 border-green-200";
    case "needs-review":
      return "bg-yellow-50 text-yellow-700 border-yellow-200";
    case "missing":
      return "bg-gray-50 text-gray-500 border-gray-200";
  }
}
