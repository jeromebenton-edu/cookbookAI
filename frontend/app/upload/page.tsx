"use client";

import { useEffect, useMemo, useState } from "react";
import DocOverlayViewer from "../../components/ai/DocOverlayViewer";
import AiLegend from "../../components/ai/AiLegend";
import AiControlsPanel from "../../components/ai/AiControlsPanel";
import { getLabelColor } from "../../components/ai/constants";
import { RecipeToken, ExtractedRecipe } from "../../lib/types";
import {
  uploadPage,
  getUploadImageUrl,
  getUploadPred,
  getUploadRecipe,
  UploadSessionSummary,
} from "../../lib/api/upload";
import { useRecipeCorrection } from "../../hooks/useRecipeCorrection";
import { downloadJson } from "../../lib/export/downloadJson";

type PredResponse = {
  tokens: Array<{ text: string; bbox: [number, number, number, number]; pred_label: string; confidence: number }>;
  image_url?: string;
};

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [session, setSession] = useState<UploadSessionSummary | null>(null);
  const [pred, setPred] = useState<PredResponse | null>(null);
  const [recipe, setRecipe] = useState<ExtractedRecipe | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const tokens: RecipeToken[] = useMemo(() => {
    if (!pred?.tokens) return [];
    return pred.tokens
      .filter((t) => t.pred_label !== "O")
      .map((t, idx) => ({
        id: `tok-${idx}`,
        text: t.text,
        label: t.pred_label as RecipeToken["label"],
        score: t.confidence,
        bbox: t.bbox,
      }));
  }, [pred]);

  const [selectedLabels, setSelectedLabels] = useState<Set<string>>(new Set());
  const [confidence, setConfidence] = useState(0.5);
  const [showBoxes, setShowBoxes] = useState(true);
  useEffect(() => {
    const labels = new Set(tokens.map((t) => t.label));
    setSelectedLabels(labels);
  }, [tokens]);

  const filtered = useMemo(
    () => tokens.filter((t) => selectedLabels.has(t.label) && t.score >= confidence),
    [tokens, selectedLabels, confidence]
  );
  const legendEntries = useMemo(() => {
    const counts: Record<string, number> = {};
    filtered.forEach((t) => {
      counts[t.label] = (counts[t.label] ?? 0) + 1;
    });
    return Object.entries(counts).map(([label, count]) => ({ label, count }));
  }, [filtered]);

  const correction = useRecipeCorrection(recipe);

  const handleFile = (f: File | null) => {
    if (!f) return;
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setSession(null);
    setPred(null);
    setRecipe(null);
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const summary = await uploadPage(file);
      setSession(summary);
      const predResp = await getUploadPred(summary.session_id);
      setPred(predResp as PredResponse);
      const recipeResp = await getUploadRecipe(summary.session_id);
      setRecipe(recipeResp as ExtractedRecipe);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-6 px-4 py-10">
      <div className="flex flex-col gap-2">
        <p className="text-xs uppercase tracking-[0.28em] text-[#6b8b6f]">Upload</p>
        <h1 className="display-font text-4xl font-semibold">Parse your own recipe page</h1>
        <p className="max-w-3xl text-sm leading-relaxed text-[#4b4237]">
          Upload a single recipe page (PNG/JPG). We will run OCR + LayoutLMv3, draw overlays, extract a recipe, and let
          you correct/export the result.
        </p>
      </div>

      <div className="rounded-2xl border border-dashed border-[#2c2620]/20 bg-white/80 p-6 text-sm text-[#2c2620]">
        <input
          type="file"
          accept="image/png,image/jpeg"
          onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
        />
        {previewUrl ? (
          <div className="mt-3 flex items-center gap-3 text-xs text-[#4b4237]">
            <span className="rounded-full bg-[#2c2620]/5 px-3 py-1">{file?.name}</span>
            <button
              className="rounded-full bg-[#2c2620] px-4 py-2 text-xs uppercase tracking-[0.2em] text-white shadow"
              onClick={handleUpload}
              disabled={loading}
            >
              {loading ? "Processing..." : "Upload and Parse"}
            </button>
          </div>
        ) : (
          <p className="mt-2 text-xs text-[#6b8b6f]">Max size 10MB. PNG/JPG.</p>
        )}
        {error ? <p className="mt-2 text-sm text-[#d34b4b]">{error}</p> : null}
      </div>

      {session && (
        <div className="rounded-2xl border border-[#2c2620]/10 bg-white/70 p-3 text-xs text-[#4b4237]">
          <p>Session: {session.session_id}</p>
          <p>Status: {session.status}</p>
        </div>
      )}

      {pred && (
        <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="flex flex-col gap-4">
            <AiControlsPanel
              labels={Array.from(new Set([...selectedLabels]))}
              selected={selectedLabels}
              onToggleLabel={(lbl) =>
                setSelectedLabels((prev) => {
                  const next = new Set(prev);
                  if (next.has(lbl)) next.delete(lbl);
                  else next.add(lbl);
                  return next;
                })
              }
              onSelectAll={() => setSelectedLabels(new Set(tokens.map((t) => t.label)))}
              onClear={() => setSelectedLabels(new Set())}
              showBoxes={showBoxes}
              onToggleBoxes={setShowBoxes}
              confidence={confidence}
              onConfidenceChange={setConfidence}
              disabled={tokens.length === 0}
            />
            <div className="paper-card flex flex-col gap-4 p-4">
              <DocOverlayViewer
                imageUrl={session?.session_id ? getUploadImageUrl(session.session_id) : previewUrl ?? ""}
                imageSize={undefined}
                tokens={filtered}
                visibleLabels={selectedLabels}
                showBoxes={showBoxes}
                confidenceThreshold={confidence}
              />
              <AiLegend entries={legendEntries} />
            </div>
          </div>
          <aside className="flex flex-col gap-3">
            <div className="paper-card flex flex-col gap-3 p-5">
              <p className="text-xs uppercase tracking-[0.22em] text-[#6b8b6f]">AI Extracted Recipe</p>
              <input
                className="rounded border border-[#2c2620]/20 bg-white px-3 py-2 text-sm"
                value={correction.state.title}
                onChange={(e) => correction.updateTitle(e.target.value)}
                placeholder="Title"
              />
              <div>
                <div className="flex items-center justify-between">
                  <p className="text-[11px] uppercase tracking-[0.18em] text-[#6b8b6f]">Ingredients</p>
                  <button className="text-[11px] text-[#2c2620]" onClick={correction.addIngredient}>
                    + Add
                  </button>
                </div>
                <ul className="mt-2 space-y-2">
                  {correction.state.ingredients.map((ing) => (
                    <li key={ing.id} className="flex items-center gap-2">
                      <input
                        className="flex-1 rounded border border-[#2c2620]/20 bg-white px-3 py-2 text-sm"
                        value={ing.text}
                        onChange={(e) => correction.updateIngredient(ing.id, e.target.value)}
                      />
                      <span className="rounded-full bg-[#2c2620]/5 px-2 py-1 text-[10px]">
                        {confidenceLabel(ing.confidence ?? 0)}
                      </span>
                      <button className="text-[11px] text-[#d34b4b]" onClick={() => correction.removeIngredient(ing.id)}>
                        Delete
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <div className="flex items-center justify-between">
                  <p className="text-[11px] uppercase tracking-[0.18em] text-[#6b8b6f]">Instructions</p>
                  <button className="text-[11px] text-[#2c2620]" onClick={correction.addInstruction}>
                    + Add
                  </button>
                </div>
                <ol className="mt-2 space-y-2">
                  {correction.state.instructions.map((step) => (
                    <li key={step.id} className="flex items-start gap-2">
                      <textarea
                        className="flex-1 rounded border border-[#2c2620]/20 bg-white px-3 py-2 text-sm"
                        value={step.text}
                        onChange={(e) => correction.updateInstruction(step.id, e.target.value)}
                      />
                      <span className="rounded-full bg-[#2c2620]/5 px-2 py-1 text-[10px]">
                        {confidenceLabel(step.confidence ?? 0)}
                      </span>
                      <button className="text-[11px] text-[#d34b4b]" onClick={() => correction.removeInstruction(step.id)}>
                        Delete
                      </button>
                    </li>
                  ))}
                </ol>
              </div>
              <div className="flex flex-wrap gap-2">
                <button
                  className="rounded-full border border-[#2c2620]/15 bg-white px-4 py-2 text-xs uppercase tracking-[0.2em] text-[#2c2620] shadow-sm transition hover:-translate-y-[1px] hover:shadow-md"
                  onClick={() =>
                    downloadJson(
                      correction.exportJson({
                        model: "layoutlmv3",
                        session_id: session?.session_id,
                      }),
                      `upload_${session?.session_id ?? "session"}_corrected.json`
                    )
                  }
                >
                  Download corrected JSON
                </button>
                <button
                  className="rounded-full border border-[#2c2620]/15 bg-white px-4 py-2 text-xs uppercase tracking-[0.2em] text-[#2c2620] shadow-sm transition hover:-translate-y-[1px] hover:shadow-md"
                  onClick={() =>
                    navigator.clipboard.writeText(
                      JSON.stringify(
                        correction.exportJson({
                          model: "layoutlmv3",
                          session_id: session?.session_id,
                        }),
                        null,
                        2
                      )
                    )
                  }
                >
                  Copy corrected JSON
                </button>
              </div>
            </div>
          </aside>
        </div>
      )}
    </div>
  );
}

function confidenceLabel(v: number): string {
  if (v >= 0.75) return `High ${(v * 100).toFixed(0)}%`;
  if (v >= 0.55) return `Med ${(v * 100).toFixed(0)}%`;
  return `Low ${(v * 100).toFixed(0)}%`;
}
