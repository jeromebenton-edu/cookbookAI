/**
 * Product Mode View - Polished recipe display without ML internals
 *
 * Shows:
 * - Scan image (left)
 * - Formatted recipe card (right) with title, ingredients, instructions
 * - Extraction status metrics (per-section chips, coverage, page readiness)
 * - Action buttons (Copy recipe, Export JSON, Show how AI found this)
 */

"use client";

import { useState } from "react";
import Image from "next/image";
import type { ExtractedRecipe } from "../../lib/types";
import type { DemoMetrics, SectionStatus } from "../../lib/demoMetrics";
import { formatSectionStatus, getSectionStatusClass } from "../../lib/demoMetrics";
import JsonViewerModal from "../ai/JsonViewerModal";
import { recipeToMarkdown } from "../../lib/recipeMarkdown";

type ProductModeViewProps = {
  imageUrl: string;
  recipe: ExtractedRecipe | null;
  metrics: DemoMetrics;
  onEnterInspector: () => void;
  loading?: boolean;
};

function SectionStatusChip({ status }: { status: SectionStatus }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider ${getSectionStatusClass(status)}`}
    >
      {status === "high-confidence" && <span className="text-green-600">✓</span>}
      {status === "needs-review" && <span className="text-yellow-600">⚠</span>}
      {status === "missing" && <span className="text-gray-400">○</span>}
      {formatSectionStatus(status)}
    </span>
  );
}

function PageReadinessLabel({ readiness }: { readiness: "ready" | "review" }) {
  if (readiness === "ready") {
    return (
      <div className="inline-flex items-center gap-2 rounded-full bg-green-50 px-3 py-1.5 text-sm font-medium text-green-700">
        <span className="text-base">✅</span>
        Ready to use
      </div>
    );
  }

  return (
    <div className="inline-flex items-center gap-2 rounded-full bg-yellow-50 px-3 py-1.5 text-sm font-medium text-yellow-700">
      <span className="text-base">👀</span>
      Review recommended
    </div>
  );
}

export default function ProductModeView({
  imageUrl,
  recipe,
  metrics,
  onEnterInspector,
  loading = false,
}: ProductModeViewProps) {
  const [copyStatus, setCopyStatus] = useState<"copied" | null>(null);

  const handleCopy = async () => {
    if (!recipe) return;
    await navigator.clipboard.writeText(recipeToMarkdown(recipe));
    setCopyStatus("copied");
    setTimeout(() => setCopyStatus(null), 1600);
  };

  const hasRecipe = recipe && recipe.is_recipe_page !== false;

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {/* Left: Scan Image */}
      <div className="flex flex-col gap-4">
        <div className="paper-card overflow-hidden p-4">
          <div className="mb-3">
            <p className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">Source page for this recipe</p>
            <p className="text-sm text-[#4b4237]">Original cookbook scan processed by LayoutLMv3</p>
          </div>

          {loading || !imageUrl ? (
            <div className="flex h-[600px] items-center justify-center rounded-xl border border-dashed border-[#2c2620]/20 bg-white/60">
              <div className="h-full w-full animate-pulse rounded-xl bg-gradient-to-br from-[#f7efe3] to-white/70" />
            </div>
          ) : (
            <div className="relative h-[600px] w-full overflow-hidden rounded-xl border border-[#2c2620]/10 bg-white shadow-inner">
              <Image
                src={imageUrl}
                alt="Scanned recipe page"
                fill
                className="object-contain"
                sizes="(max-width: 1024px) 100vw, 50vw"
                priority
              />
            </div>
          )}
        </div>

        <button
          onClick={onEnterInspector}
          className="rounded-full border border-[#2c2620]/15 bg-white px-4 py-2.5 text-sm font-medium uppercase tracking-[0.18em] text-[#2c2620] shadow-sm transition hover:-translate-y-[1px] hover:shadow-md"
        >
          Show how the AI found this →
        </button>
      </div>

      {/* Right: Recipe Card */}
      <aside className="flex flex-col gap-4">
        <div className="paper-card flex flex-col gap-4 p-5">
          {/* Page Readiness */}
          <div className="flex items-start justify-between gap-3">
            <PageReadinessLabel readiness={metrics.pageReadiness} />
            {recipe?.recipe_confidence !== undefined && (
              <span className="rounded-full bg-[#2c2620]/5 px-3 py-1 text-[11px] uppercase tracking-[0.18em] text-[#2c2620]">
                {Math.round(recipe.recipe_confidence * 100)}% confidence
              </span>
            )}
          </div>

          {/* Title Section */}
          <div>
            <div className="mb-1.5 flex items-center gap-2">
              <p className="text-xs uppercase tracking-[0.22em] text-[#6b8b6f]">Title</p>
              <SectionStatusChip status={metrics.sectionStatus.title} />
            </div>
            <h3 className="display-font text-2xl font-semibold text-[#2c2620]">
              {hasRecipe ? recipe.title || "Untitled Recipe" : loading ? "Loading..." : "No recipe detected"}
            </h3>
          </div>

          {hasRecipe ? (
            <>
              {/* Ingredients Section */}
              <div>
                <div className="mb-2 flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <p className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">Ingredients</p>
                    <SectionStatusChip status={metrics.sectionStatus.ingredients} />
                  </div>
                  <span className="text-xs text-[#4b4237]">
                    {metrics.coverage.ingredientCount} captured
                  </span>
                </div>
                <ul className="list-disc space-y-1 pl-5 text-sm leading-relaxed text-[#2c2620]">
                  {recipe.ingredients.length > 0 ? (
                    recipe.ingredients.map((ing, idx) => <li key={idx}>{ing}</li>)
                  ) : (
                    <li className="text-[#4b4237]">No ingredients detected</li>
                  )}
                </ul>
              </div>

              {/* Instructions Section */}
              <div>
                <div className="mb-2 flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <p className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">Instructions</p>
                    <SectionStatusChip status={metrics.sectionStatus.instructions} />
                  </div>
                  <span className="text-xs text-[#4b4237]">
                    {metrics.coverage.stepCount} steps
                  </span>
                </div>
                <ol className="list-decimal space-y-1.5 pl-5 text-sm leading-relaxed text-[#2c2620]">
                  {recipe.instructions.length > 0 ? (
                    recipe.instructions.map((step, idx) => <li key={idx}>{step}</li>)
                  ) : (
                    <li className="text-[#4b4237]">No instructions detected</li>
                  )}
                </ol>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-wrap items-center gap-3 border-t border-[#2c2620]/10 pt-4">
                <button
                  onClick={handleCopy}
                  disabled={!recipe}
                  className="rounded-full bg-[#2c2620] px-5 py-2.5 text-xs font-medium uppercase tracking-[0.2em] text-white shadow-sm transition hover:-translate-y-[1px] hover:shadow-md disabled:opacity-60 disabled:hover:translate-y-0"
                >
                  Copy recipe
                </button>
                <JsonViewerModal data={recipe ?? {}} label="Export JSON" />
                {copyStatus && (
                  <span className="text-xs text-[#6b8b6f]">✓ Recipe copied to clipboard</span>
                )}
              </div>
            </>
          ) : (
            <div className="rounded-xl border border-[#2c2620]/10 bg-[#fff8ed] px-4 py-3 text-sm text-[#5b3d22]">
              {loading ? "Loading recipe extraction..." : recipe?.message ?? "No recipe detected on this page."}
              {!loading && (
                <p className="mt-1 text-xs text-[#4b4237]">
                  Click "Show how the AI found this" to inspect the model's output.
                </p>
              )}
            </div>
          )}
        </div>

        {/* Upload CTA */}
        <div className="rounded-2xl border border-[#2c2620]/10 bg-white/80 p-4 text-sm text-[#4b4237]">
          <p className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">Try your own page</p>
          <p className="mt-1">Upload your cookbook scan to extract recipes with LayoutLMv3.</p>
          <a
            href="/upload"
            className="mt-3 inline-flex rounded-full bg-[#2c2620] px-4 py-2 text-xs uppercase tracking-[0.2em] text-white shadow transition hover:-translate-y-[1px] hover:shadow-md"
          >
            Upload a page
          </a>
        </div>
      </aside>
    </div>
  );
}
