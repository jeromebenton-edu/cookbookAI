import { useEffect, useMemo, useState } from "react";
import { Recipe, RecipeToken } from "../lib/types";
import { useParseOverlay } from "../hooks/useParseOverlay";
import AiControlsPanel from "./ai/AiControlsPanel";
import AiLegend from "./ai/AiLegend";
import AiNarrative from "./ai/AiNarrative";
import ConfidenceBar from "./ai/ConfidenceBar";
import DocOverlayViewer from "./ai/DocOverlayViewer";
import { LABEL_COLORS } from "./ai/constants";
import JsonViewerModal from "./ai/JsonViewerModal";

export default function RecipeDetailAiView({ recipe, enabled = true }: { recipe: Recipe; enabled?: boolean }) {
  const pageNum = recipe.source.page;
  const { data: overlayData, loading, error, refresh, unavailable, hint } = useParseOverlay(pageNum, enabled);

  const tokens: RecipeToken[] = useMemo(() => {
    if (!overlayData?.tokens) return [];
    return overlayData.tokens
      .filter((t) => t.pred_label !== "O")
      .map((t, idx) => ({
        id: `tok-${pageNum}-${idx}`,
        text: t.text,
        label: (t.pred_label as RecipeToken["label"]) ?? "OTHER",
        score: t.confidence,
        bbox: t.bbox as [number, number, number, number]
      }));
  }, [overlayData, pageNum]);

  const availableLabels = useMemo(() => {
    const labels = Array.from(new Set(tokens.map((token) => token.label).filter((l) => l !== "O")));
    const canonical = Object.keys(LABEL_COLORS).filter((l) => l !== "O");
    return labels.length > 0 ? labels : canonical;
  }, [tokens]);

  const [selectedLabels, setSelectedLabels] = useState<Set<string>>(
    new Set(availableLabels)
  );
  const [showBoxes, setShowBoxes] = useState(true);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);

  useEffect(() => {
    setSelectedLabels(new Set(availableLabels));
  }, [availableLabels]);

  const filteredTokens = useMemo(() => {
    return tokens.filter(
      (token) =>
        selectedLabels.has(token.label) && token.score >= confidenceThreshold
    );
  }, [tokens, selectedLabels, confidenceThreshold]);

  const legendEntries = useMemo(() => {
    const counts: Record<string, number> = {};
    filteredTokens.forEach((token) => {
      counts[token.label] = (counts[token.label] ?? 0) + 1;
    });
    return Object.entries(counts).map(([label, count]) => ({
      label,
      count
    }));
  }, [filteredTokens]);

  const fieldConfidence = recipe.ai.fieldConfidence ?? {};
  const overallConfidence = computeOverallConfidence(recipe, tokens);

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

  return (
    <div className="flex flex-col gap-6">
      <AiNarrative recipe={recipe} overallConfidence={overallConfidence} />

      <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="flex flex-col gap-4">
          <AiControlsPanel
            labels={availableLabels}
            selected={selectedLabels}
            onToggleLabel={handleToggleLabel}
            onSelectAll={() => setSelectedLabels(new Set(availableLabels))}
            onClear={() => setSelectedLabels(new Set())}
            showBoxes={showBoxes}
            onToggleBoxes={setShowBoxes}
            confidence={confidenceThreshold}
            onConfidenceChange={setConfidenceThreshold}
            disabled={tokens.length === 0}
          />
          <div className="paper-card flex flex-col gap-4 p-4">
            {error ? (
              <div className="rounded-xl border border-[#d34b4b]/30 bg-[#fff5f5] px-4 py-3 text-sm text-[#7a1d1d]">
                {unavailable ? (
                  <div className="flex flex-col gap-1">
                    <span>AI overlay not available for this page (demo subset only).</span>
                    <span className="text-xs text-[#a04141]">{error}</span>
                    <div className="flex flex-wrap gap-2 text-xs">
                      <a
                        href="/demo"
                        className="inline-flex items-center rounded-full border border-[#7a1d1d]/30 bg-white px-3 py-1 uppercase tracking-[0.18em] text-[#7a1d1d] transition hover:-translate-y-[1px] hover:shadow-sm"
                      >
                        Go to Demo
                      </a>
                      {hint ? <span className="text-xs text-[#7a1d1d]">{hint}</span> : null}
                    </div>
                  </div>
                ) : (
                  <>
                    AI parsing service is offline or unreachable.
                    <div className="text-xs text-[#a04141]">{error}</div>
                  </>
                )}
              </div>
            ) : null}
            {loading ? (
              <div className="flex items-center justify-center rounded-xl border border-[#2c2620]/10 bg-white/70 px-4 py-6 text-sm text-[#4b4237]">
                Loading AI overlay...
              </div>
            ) : (
              <DocOverlayViewer
                imageUrl={overlayData?.image_url_resolved ?? recipe.ai.pageImage}
                imageSize={recipe.ai.pageSize}
                tokens={tokens}
                visibleLabels={selectedLabels}
                showBoxes={showBoxes}
                confidenceThreshold={confidenceThreshold}
              />
            )}
            <p className="text-xs text-[#4b4237]">
              Hover a box to see the predicted label, token text, and
              confidence.
            </p>
            <AiLegend entries={legendEntries} />
          </div>
        </div>

        <aside className="flex flex-col gap-4">
          <div className="paper-card flex flex-col gap-4 p-6">
            <h4 className="display-font text-2xl">Extracted Fields</h4>
            <div className="flex flex-col gap-4 text-sm leading-relaxed text-[#2c2620]">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">
                  Title
                </p>
                <p className="text-base font-semibold">{recipe.title}</p>
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">
                  Ingredients
                </p>
                <ul className="list-disc pl-5">
                  {recipe.ingredients.map((ingredient, idx) => (
                    <li key={idx}>{ingredient}</li>
                  ))}
                </ul>
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">
                  Instructions
                </p>
                <ol className="list-decimal pl-5">
                  {recipe.instructions.map((step, idx) => (
                    <li key={idx}>{step}</li>
                  ))}
                </ol>
              </div>
              <div className="grid grid-cols-2 gap-3 text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">
                <span>Servings</span>
                <span className="text-right text-[#2c2620]">{recipe.servings}</span>
                <span>Time</span>
                <span className="text-right text-[#2c2620]">
                  {recipe.time.label ?? `${recipe.time.totalMinutes} min`}
                </span>
              </div>
            </div>
          </div>
          <div className="paper-card flex flex-col gap-4 p-6">
            <h4 className="display-font text-2xl">Field Confidence</h4>
            <ConfidenceBar label="Title" value={fieldConfidence.title} />
            <ConfidenceBar label="Ingredients" value={fieldConfidence.ingredients} />
            <ConfidenceBar label="Instructions" value={fieldConfidence.instructions} />
            <ConfidenceBar label="Servings" value={fieldConfidence.servings} />
            <ConfidenceBar label="Time" value={fieldConfidence.time} />
          </div>
          <JsonViewerModal data={overlayData ?? recipe.ai.raw} />
        </aside>
      </div>
    </div>
  );
}

function computeOverallConfidence(recipe: Recipe, tokens: RecipeToken[]): number {
  const fc = recipe.ai.fieldConfidence;
  if (fc) {
    const values = Object.values(fc).filter((v): v is number => typeof v === "number");
    if (values.length) {
      return values.reduce((sum, value) => sum + value, 0) / values.length;
    }
  }
  if (tokens?.length) {
    const scores = tokens.map((t) => t.score ?? 0);
    return scores.reduce((sum, value) => sum + value, 0) / scores.length;
  }
  return 0.85; // TODO: replace with model-derived confidence when available
}
