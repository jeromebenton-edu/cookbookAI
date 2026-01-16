/**
 * Bundled demo examples loader
 *
 * Provides offline-first demo examples that work without network calls.
 * Examples are versioned with the app and loaded from /src/demo_examples/
 *
 * TASK A: Strict interface with validation
 * TASK C: Mismatch guard to ensure scan matches prediction
 */

import type { ExtractedRecipe } from "./types";

/**
 * Metadata for a demo example with validation fields
 */
export type DemoExampleMeta = {
  title: string;
  difficulty: "easy" | "medium" | "hard";
  description?: string;
  tags?: string[];
  page_num?: number;
  // Validation fields
  expectedTitle: string; // What title we expect from the prediction
  expectedKeywords?: string[]; // Keywords to fuzzy match
};

/**
 * A complete demo example package - scan image + prediction must match
 */
export type DemoExample = {
  id: string;
  title: string; // Short display title
  difficulty: "easy" | "medium" | "hard";
  scanImageSrc: string; // Path to the scan image (public asset)
  prediction: ExtractedRecipe; // Prediction JSON from this scan
  meta: DemoExampleMeta; // Full metadata including validation
  pageImage: string; // Legacy field - same as scanImageSrc
};

// Import example data (these will be bundled with the app)
import example01Prediction from "../src/demo_examples/example_01/prediction.json";
import example01Meta from "../src/demo_examples/example_01/meta.json";
import example02Prediction from "../src/demo_examples/example_02/prediction.json";
import example02Meta from "../src/demo_examples/example_02/meta.json";

/**
 * Validation result for an example
 */
type ValidationResult = {
  valid: boolean;
  warnings: string[];
};

/**
 * Validate that a prediction matches the expected metadata
 */
function validateExample(
  id: string,
  prediction: ExtractedRecipe,
  meta: DemoExampleMeta
): ValidationResult {
  const warnings: string[] = [];
  const predictionTitle = prediction.title?.toLowerCase() || "";
  const expectedTitle = meta.expectedTitle.toLowerCase();
  const keywords = meta.expectedKeywords?.map((k) => k.toLowerCase()) || [];

  // Check if prediction title matches expected title
  const titleMatches = predictionTitle.includes(expectedTitle) ||
                       expectedTitle.includes(predictionTitle);

  // Check if any keywords match
  const keywordMatches = keywords.length === 0 ||
                         keywords.some((kw) => predictionTitle.includes(kw));

  if (!titleMatches && !keywordMatches) {
    warnings.push(
      `Demo example ${id} mismatch: ` +
      `prediction title "${prediction.title}" does not match ` +
      `expected "${meta.expectedTitle}" or keywords [${keywords.join(", ")}]. ` +
      `Scan may not match the extracted recipe.`
    );
  }

  // Check if page_num in prediction matches meta (if both exist)
  if (prediction.page_num && meta.page_num && prediction.page_num !== meta.page_num) {
    warnings.push(
      `Demo example ${id} page number mismatch: ` +
      `prediction has page ${prediction.page_num} but meta expects ${meta.page_num}`
    );
  }

  return {
    valid: warnings.length === 0,
    warnings,
  };
}

/**
 * Single source of truth for all demo examples
 */
const DEMO_EXAMPLES_RAW = [
  {
    id: "example_01",
    scanImageSrc: "/demo_examples/example_01/page.png?v=20260111g", // Cache bust (real OCR tokens!)
    prediction: example01Prediction as unknown as ExtractedRecipe,
    meta: example01Meta as DemoExampleMeta,
  },
  {
    id: "example_02",
    scanImageSrc: "/demo_examples/example_02/page.png?v=20260111g", // Cache bust (real OCR tokens!)
    prediction: example02Prediction as unknown as ExtractedRecipe,
    meta: example02Meta as DemoExampleMeta,
  },
];

/**
 * Validate all examples on load (dev-time check)
 */
function validateAllExamples() {
  const isDev = process.env.NODE_ENV === "development";

  for (const ex of DEMO_EXAMPLES_RAW) {
    const result = validateExample(ex.id, ex.prediction, ex.meta);

    if (!result.valid) {
      if (isDev) {
        // In development, log warnings to console
        result.warnings.forEach((warning) => console.warn(`⚠️  ${warning}`));
      } else {
        // In production, silently log but don't break
        result.warnings.forEach((warning) => console.info(`[Demo validation] ${warning}`));
      }
    }
  }
}

// Run validation on module load
if (typeof window !== "undefined") {
  validateAllExamples();
}

/**
 * Get all bundled demo examples
 */
export function getBundledExamples(): DemoExample[] {
  return DEMO_EXAMPLES_RAW.map((ex) => ({
    id: ex.id,
    title: ex.meta.title,
    difficulty: ex.meta.difficulty,
    scanImageSrc: ex.scanImageSrc,
    prediction: ex.prediction,
    meta: ex.meta,
    // Legacy field for backward compatibility
    pageImage: ex.scanImageSrc,
  })) as DemoExample[];
}

/**
 * Get the default demo example (first one)
 */
export function getDefaultExample(): DemoExample {
  const examples = getBundledExamples();
  if (examples.length === 0) {
    throw new Error("No bundled demo examples available");
  }
  return examples[0];
}

/**
 * Get a specific example by ID
 */
export function getExampleById(id: string): DemoExample | null {
  const examples = getBundledExamples();
  return examples.find((ex) => ex.id === id) || null;
}

/**
 * Check if examples are valid (for UI display)
 */
export function areExamplesValid(): boolean {
  let allValid = true;

  for (const ex of DEMO_EXAMPLES_RAW) {
    const result = validateExample(ex.id, ex.prediction, ex.meta);
    if (!result.valid) {
      allValid = false;
    }
  }

  return allValid;
}
