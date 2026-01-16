/**
 * Demo Page - Redesigned for zero-setup, polished extraction
 *
 * Features:
 * - Bundled demo examples (no network required for initial render)
 * - Product Mode (default): Polished recipe view
 * - Inspector Mode (advanced): ML internals with overlays
 * - Visible metrics: section status, coverage, page readiness
 */

"use client";

import { Suspense, useEffect, useState, useMemo } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  getBundledExamples,
  getDefaultExample,
  type DemoExample,
} from "../../lib/bundledExamples";
import { computeDemoMetrics } from "../../lib/demoMetrics";
import type { RecipeToken } from "../../lib/types";
import type { DemoPrediction } from "../../lib/demoPredictionTypes";
import ProductModeView from "../../components/demo/ProductModeView";
import InspectorModeView from "../../components/demo/InspectorModeView";
import ValidationBanner from "../../components/demo/ValidationBanner";

// Import NEW real token data (245+ OCR tokens per example)
import realTokenData01 from "../../src/demo_examples/example_01/prediction.json";
import realTokenData02 from "../../src/demo_examples/example_02/prediction.json";

// Map example IDs to real token data
const REAL_TOKEN_DATA: Record<string, DemoPrediction> = {
  example_01: realTokenData01 as unknown as DemoPrediction,
  example_02: realTokenData02 as unknown as DemoPrediction,
};

/**
 * Convert DemoPrediction tokens to RecipeToken format for overlay rendering
 */
function convertRealTokens(prediction: DemoPrediction): RecipeToken[] {
  return prediction.tokens.map((t) => ({
    id: `tok-${t.id}`,
    text: t.text,
    label: t.label as RecipeToken["label"],
    score: t.conf,
    bbox: t.bbox,
  }));
}

type ViewMode = "product" | "inspector";

function DemoPageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // State
  const [mode, setMode] = useState<ViewMode>("product");
  const [examples] = useState<DemoExample[]>(getBundledExamples());
  const [selectedExampleId, setSelectedExampleId] = useState<string>(() => {
    const idFromUrl = searchParams?.get("example");
    return idFromUrl || getDefaultExample().id;
  });

  // Inspector mode state
  const [selectedLabels, setSelectedLabels] = useState<Set<string>>(
    new Set(["TITLE", "INGREDIENT_LINE", "INSTRUCTION_STEP"])
  );
  const [confidence, setConfidence] = useState(0.5);
  const [showBoxes, setShowBoxes] = useState(true);

  // Derived state
  const currentExample = useMemo(
    () => examples.find((ex) => ex.id === selectedExampleId) || examples[0],
    [examples, selectedExampleId]
  );

  // Extract the recipe data for ProductModeView (handles both old and new formats)
  const recipeForDisplay = useMemo(() => {
    const realData = REAL_TOKEN_DATA[currentExample.id];
    if (realData) {
      // NEW format: Use extractedRecipe from DemoPrediction with defaults for missing fields
      return {
        ...realData.extractedRecipe,
        is_recipe_page: true,
        recipe_confidence: realData.extractedRecipe.confidence.overall,
        page_num: realData.meta.cookbookPage || 0,
        meta: {},
      };
    }
    // OLD format: Use prediction directly
    return currentExample.prediction;
  }, [currentExample.id, currentExample.prediction]);

  const metrics = useMemo(
    () => computeDemoMetrics(recipeForDisplay),
    [recipeForDisplay]
  );

  // Extract imageSize from prediction data for correct bbox rendering
  const imageSize = useMemo(() => {
    const realData = REAL_TOKEN_DATA[currentExample.id];
    if (realData?.page) {
      return { width: realData.page.width, height: realData.page.height };
    }
    return undefined;
  }, [currentExample.id]);

  // Generate overlay tokens from REAL OCR data (245+ tokens per example!)
  const overlayTokens = useMemo<RecipeToken[]>(() => {
    // Use real token data if available
    const realData = REAL_TOKEN_DATA[currentExample.id];
    if (realData) {
      console.log(`[Demo] Using REAL OCR tokens for ${currentExample.id}:`, {
        tokenCount: realData.tokens.length,
        imageSize: `${realData.page.width}x${realData.page.height}`,
        coordSpace: realData.page.coordSpace,
      });
      return convertRealTokens(realData);
    }

    // Fallback to legacy format (should not happen with new fixtures)
    console.warn(`[Demo] No real token data for ${currentExample.id}, using legacy format`);
    const tokens: RecipeToken[] = [];
    const pred = currentExample.prediction;

    // Title tokens
    if (pred.title_obj) {
      tokens.push({
        id: pred.title_obj.id,
        text: pred.title_obj.text,
        label: "TITLE",
        score: pred.title_obj.confidence,
        bbox: pred.title_obj.bbox,
      });
    }

    // Ingredient tokens
    pred.ingredients_lines?.forEach((line) => {
      tokens.push({
        id: line.id,
        text: line.text,
        label: "INGREDIENT_LINE",
        score: line.confidence,
        bbox: line.bbox,
      });
    });

    // Instruction tokens
    pred.instruction_lines?.forEach((line) => {
      tokens.push({
        id: line.id,
        text: line.text,
        label: "INSTRUCTION_STEP",
        score: line.confidence,
        bbox: line.bbox,
      });
    });

    return tokens;
  }, [currentExample.id, currentExample.prediction]);

  // Update URL when example changes
  useEffect(() => {
    const currentParam = searchParams?.get("example");
    if (currentParam !== selectedExampleId) {
      router.replace(`/demo?example=${selectedExampleId}`, { scroll: false });
    }
  }, [selectedExampleId, searchParams, router]);

  // Initialize selected labels when tokens change (exclude "O" label by default)
  useEffect(() => {
    const labels = Array.from(new Set(overlayTokens.map((t) => t.label))).filter(l => l !== "O");
    setSelectedLabels(new Set(labels));
  }, [overlayTokens]);

  const handleEnterInspector = () => setMode("inspector");
  const handleExitInspector = () => setMode("product");

  const handleToggleLabel = (label: string) => {
    setSelectedLabels((prev) => {
      const next = new Set(prev);
      if (next.has(label)) {
        next.delete(label);
      } else {
        next.add(label);
      }
      return next;
    });
  };

  const handleSelectAll = () => {
    const allLabels = new Set(overlayTokens.map((t) => t.label));
    setSelectedLabels(allLabels);
  };

  const handleClear = () => {
    setSelectedLabels(new Set());
  };

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-10">
      {/* Header */}
      <div className="flex flex-col gap-2">
        <p className="text-xs uppercase tracking-[0.28em] text-[#6b8b6f]">AI Recipe Extraction Demo</p>
        <h1 className="display-font text-4xl font-semibold">
          Turn cookbook scans into structured recipes.
        </h1>
        <p className="max-w-3xl text-sm leading-relaxed text-[#4b4237]">
          See how LayoutLMv3 extracts titles, ingredients, and instructions from scanned cookbook pages.
          {mode === "product"
            ? " Click 'Show how the AI found this' to see the ML internals."
            : " You're in Inspector Mode - viewing token-level predictions."}
        </p>
      </div>

      {/* Validation Banner */}
      <ValidationBanner />

      {/* Example Selector */}
      {examples.length > 1 && (
        <div className="glass-panel flex flex-col gap-3 p-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">Demo Examples</p>
              <p className="text-sm text-[#4b4237]">
                Choose from {examples.length} bundled examples. No network required.
              </p>
            </div>
            <span className="rounded-full bg-[#2c2620]/5 px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-[#2c2620]">
              {currentExample.meta.difficulty}
            </span>
          </div>
          <div className="flex gap-2 overflow-x-auto pb-1">
            {examples.map((ex) => (
              <button
                key={ex.id}
                onClick={() => setSelectedExampleId(ex.id)}
                className={`min-w-[160px] rounded-xl border px-3 py-2 text-left text-xs transition ${
                  selectedExampleId === ex.id
                    ? "border-[#2c2620] bg-[#f7efe3]"
                    : "border-[#2c2620]/10 bg-white hover:border-[#2c2620]/30"
                }`}
              >
                <p className="text-[10px] uppercase tracking-[0.2em] text-[#6b8b6f]">
                  {ex.meta.difficulty}
                </p>
                <p className="font-semibold text-[#2c2620]">{ex.meta.title}</p>
                <p className="text-[10px] text-[#4b4237]">{ex.meta.description}</p>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Mode-specific content */}
      {mode === "product" ? (
        <ProductModeView
          imageUrl={currentExample.pageImage}
          recipe={recipeForDisplay}
          metrics={metrics}
          onEnterInspector={handleEnterInspector}
          loading={false}
        />
      ) : (
        <InspectorModeView
          imageUrl={currentExample.pageImage}
          imageSize={imageSize}
          tokens={overlayTokens}
          selectedTitle={recipeForDisplay?.title || ""}
          selectedLabels={selectedLabels}
          onToggleLabel={handleToggleLabel}
          onSelectAll={handleSelectAll}
          onClear={handleClear}
          showBoxes={showBoxes}
          onToggleBoxes={setShowBoxes}
          confidence={confidence}
          onConfidenceChange={setConfidence}
          onExitInspector={handleExitInspector}
        />
      )}
    </div>
  );
}

export default function DemoPage() {
  return (
    <Suspense
      fallback={
        <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-10">
          <p className="text-sm text-[#4b4237]">Loading demo...</p>
        </div>
      }
    >
      <DemoPageInner />
    </Suspense>
  );
}
