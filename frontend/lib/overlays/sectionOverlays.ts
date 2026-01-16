/**
 * Section-level overlay aggregation for demo Inspector mode
 *
 * This module groups token-level predictions into semantic section boxes
 * (Title, Ingredients, Instructions) for a cleaner Label Studio-style UX.
 */

import { RecipeToken } from "../types";

export type BBox = [number, number, number, number]; // [x1, y1, x2, y2]

export interface SectionBox {
  bbox: BBox;
  label: "TITLE" | "INGREDIENTS" | "INSTRUCTIONS";
  confidence: number;
  tokenIds: string[]; // For debugging
}

export interface SectionOverlays {
  titleBox?: SectionBox;
  ingredientBoxes: SectionBox[];
  instructionBox?: SectionBox;
  debug?: {
    selectedTitleMatch?: RecipeToken;
    titleYPosition?: number;
    nextTitleYPosition?: number;
    filteredIngredientCount: number;
    filteredInstructionCount: number;
  };
}

/**
 * Build section-level overlays for a selected recipe from multi-recipe page predictions
 *
 * @param tokens - All tokens from the page (may contain multiple recipes)
 * @param selectedTitle - The title of the recipe to isolate (e.g., "Waffles")
 * @param options - Configuration options
 * @returns Section-level bounding boxes for the selected recipe only
 */
export function buildSectionOverlays(
  tokens: RecipeToken[],
  selectedTitle: string,
  options: {
    /** Add padding (in pixels) around union boxes */
    padding?: number;
    /** Two-column detection threshold (if ingredients spread > this, split into 2 boxes) */
    twoColumnThreshold?: number;
  } = {}
): SectionOverlays {
  const { padding = 8, twoColumnThreshold = 250 } = options;

  // Step 1: Find the title anchor for the selected recipe
  const titleTokens = tokens.filter((t) => t.label === "TITLE");

  if (titleTokens.length === 0) {
    return {
      ingredientBoxes: [],
      debug: { filteredIngredientCount: 0, filteredInstructionCount: 0 },
    };
  }

  // Normalize text for matching: lowercase, trim, remove punctuation, collapse spaces
  const normalize = (text: string) =>
    text
      .toLowerCase()
      .trim()
      .replace(/[.,;:!?]/g, "")
      .replace(/\s+/g, " ");

  // Find best matching title (robust normalization)
  const normalizedTarget = normalize(selectedTitle);

  console.log("[sectionOverlays] Title matching:", {
    selectedTitle,
    normalizedTarget,
    candidates: titleTokens.map((t) => ({
      text: t.text,
      normalized: normalize(t.text),
      score: t.score,
    })),
  });

  const titleMatch = titleTokens.find((t) => {
    const normalizedToken = normalize(t.text);
    return (
      normalizedToken.includes(normalizedTarget) ||
      normalizedTarget.includes(normalizedToken)
    );
  });

  if (!titleMatch) {
    // Fallback: use highest confidence title (but this is risky on multi-recipe pages)
    const fallbackTitle = titleTokens.reduce((best, t) =>
      t.score > best.score ? t : best
    );
    console.warn(
      `[sectionOverlays] No exact title match for "${selectedTitle}", using fallback:`,
      fallbackTitle.text
    );
  }

  const selectedTitleToken = titleMatch || titleTokens[0];
  const titleYCenter = (selectedTitleToken.bbox[1] + selectedTitleToken.bbox[3]) / 2;

  // Step 2: Find the next TITLE below this one (cutoff boundary)
  const titlesBelow = titleTokens
    .filter((t) => t.bbox[1] > selectedTitleToken.bbox[3]) // Top edge below selected title's bottom
    .sort((a, b) => a.bbox[1] - b.bbox[1]); // Sort by top Y position

  const nextTitleYPosition = titlesBelow.length > 0 ? titlesBelow[0].bbox[1] : Infinity;

  // Step 3: Filter ingredient/instruction tokens belonging to this recipe
  const ingredientCandidates = tokens.filter(
    (t) =>
      t.label === "INGREDIENT_LINE" &&
      t.bbox[1] > selectedTitleToken.bbox[3] && // Below the title
      t.bbox[1] < nextTitleYPosition // Above the next title
  );

  const instructionCandidates = tokens.filter(
    (t) =>
      t.label === "INSTRUCTION_STEP" &&
      t.bbox[1] > selectedTitleToken.bbox[3] &&
      t.bbox[1] < nextTitleYPosition
  );

  // Step 4: Build title box
  const titleBox: SectionBox = {
    bbox: padBox(selectedTitleToken.bbox, padding),
    label: "TITLE",
    confidence: selectedTitleToken.score,
    tokenIds: [selectedTitleToken.id],
  };

  // Step 5: Build ingredient boxes (detect two-column layout)
  let ingredientBoxes: SectionBox[] = [];
  if (ingredientCandidates.length > 0) {
    const xPositions = ingredientCandidates.map((t) => (t.bbox[0] + t.bbox[2]) / 2);
    const xMin = Math.min(...xPositions);
    const xMax = Math.max(...xPositions);
    const xSpread = xMax - xMin;

    if (xSpread > twoColumnThreshold && ingredientCandidates.length >= 4) {
      // Two-column layout detected: cluster by centerX
      const xMedian = (xMin + xMax) / 2;
      const leftColumn = ingredientCandidates.filter((t) => (t.bbox[0] + t.bbox[2]) / 2 < xMedian);
      const rightColumn = ingredientCandidates.filter((t) => (t.bbox[0] + t.bbox[2]) / 2 >= xMedian);

      if (leftColumn.length > 0) {
        ingredientBoxes.push({
          bbox: padBox(unionBoxes(leftColumn.map((t) => t.bbox)), padding),
          label: "INGREDIENTS",
          confidence: average(leftColumn.map((t) => t.score)),
          tokenIds: leftColumn.map((t) => t.id),
        });
      }
      if (rightColumn.length > 0) {
        ingredientBoxes.push({
          bbox: padBox(unionBoxes(rightColumn.map((t) => t.bbox)), padding),
          label: "INGREDIENTS",
          confidence: average(rightColumn.map((t) => t.score)),
          tokenIds: rightColumn.map((t) => t.id),
        });
      }
    } else {
      // Single column
      ingredientBoxes.push({
        bbox: padBox(unionBoxes(ingredientCandidates.map((t) => t.bbox)), padding),
        label: "INGREDIENTS",
        confidence: average(ingredientCandidates.map((t) => t.score)),
        tokenIds: ingredientCandidates.map((t) => t.id),
      });
    }
  }

  // Step 6: Build instruction box
  let instructionBox: SectionBox | undefined;
  if (instructionCandidates.length > 0) {
    instructionBox = {
      bbox: padBox(unionBoxes(instructionCandidates.map((t) => t.bbox)), padding),
      label: "INSTRUCTIONS",
      confidence: average(instructionCandidates.map((t) => t.score)),
      tokenIds: instructionCandidates.map((t) => t.id),
    };
  }

  return {
    titleBox,
    ingredientBoxes,
    instructionBox,
    debug: {
      selectedTitleMatch: selectedTitleToken,
      titleYPosition: titleYCenter,
      nextTitleYPosition,
      filteredIngredientCount: ingredientCandidates.length,
      filteredInstructionCount: instructionCandidates.length,
    },
  };
}

/**
 * Compute the union (bounding box that contains all input boxes)
 */
function unionBoxes(boxes: BBox[]): BBox {
  if (boxes.length === 0) return [0, 0, 0, 0];

  const x1 = Math.min(...boxes.map((b) => b[0]));
  const y1 = Math.min(...boxes.map((b) => b[1]));
  const x2 = Math.max(...boxes.map((b) => b[2]));
  const y2 = Math.max(...boxes.map((b) => b[3]));

  return [x1, y1, x2, y2];
}

/**
 * Add padding to a bounding box
 */
function padBox(bbox: BBox, padding: number): BBox {
  return [
    bbox[0] - padding,
    bbox[1] - padding,
    bbox[2] + padding,
    bbox[3] + padding,
  ];
}

/**
 * Compute average of an array
 */
function average(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}
