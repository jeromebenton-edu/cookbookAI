/**
 * Inspector Mode View - Advanced ML inspection interface
 *
 * Shows:
 * - Overlay bounding boxes grouped by section
 * - Toggle chips for label filtering
 * - Confidence slider
 * - Optional raw JSON view
 */

"use client";

import { useMemo, useState } from "react";
import type { RecipeToken } from "../../lib/types";
import DocOverlayViewer from "../ai/DocOverlayViewer";
import SectionOverlayViewer from "../ai/SectionOverlayViewer";
import AiControlsPanel from "../ai/AiControlsPanel";
import AiLegend from "../ai/AiLegend";

export type OverlayMode = "sections" | "tokens";

type InspectorModeViewProps = {
  imageUrl: string;
  imageSize?: { width: number; height: number };
  tokens: RecipeToken[];
  selectedTitle: string;
  selectedLabels: Set<string>;
  onToggleLabel: (label: string) => void;
  onSelectAll: () => void;
  onClear: () => void;
  showBoxes: boolean;
  onToggleBoxes: (show: boolean) => void;
  confidence: number;
  onConfidenceChange: (conf: number) => void;
  onExitInspector: () => void;
};

export default function InspectorModeView({
  imageUrl,
  imageSize,
  tokens,
  selectedTitle,
  selectedLabels,
  onToggleLabel,
  onSelectAll,
  onClear,
  showBoxes,
  onToggleBoxes,
  confidence,
  onConfidenceChange,
  onExitInspector,
}: InspectorModeViewProps) {
  const [overlayMode, setOverlayMode] = useState<OverlayMode>("sections");

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

  const overlayLabels = useMemo(
    () => Array.from(new Set(tokens.map((t) => t.label))),
    [tokens]
  );

  return (
    <div className="flex flex-col gap-4">
      {/* Exit Inspector Button */}
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">
            Inspector Mode{" "}
            <span className="rounded-full bg-blue-100 px-2 py-0.5 text-[10px] font-semibold text-blue-700">
              {overlayMode === "sections" ? "Sections" : "Tokens"}
            </span>
          </p>
          <p className="text-sm text-[#4b4237]">
            {overlayMode === "sections"
              ? "Viewing section-level overlays (Title, Ingredients, Instructions) for the selected recipe."
              : "Viewing token-level ML predictions with label and confidence filtering."}
          </p>
        </div>
        <button
          onClick={onExitInspector}
          className="rounded-full border border-[#2c2620]/15 bg-white px-4 py-2 text-xs uppercase tracking-[0.18em] text-[#2c2620] shadow-sm transition hover:-translate-y-[1px] hover:shadow-md"
        >
          ← Back to recipe
        </button>
      </div>

      {/* Controls Panel - Only show in tokens mode */}
      {overlayMode === "tokens" && (
        <AiControlsPanel
          labels={overlayLabels}
          selected={selectedLabels}
          onToggleLabel={onToggleLabel}
          onSelectAll={onSelectAll}
          onClear={onClear}
          showBoxes={showBoxes}
          onToggleBoxes={onToggleBoxes}
          confidence={confidence}
          onConfidenceChange={onConfidenceChange}
          disabled={tokens.length === 0}
        />
      )}

      {/* Show boxes toggle for sections mode */}
      {overlayMode === "sections" && (
        <div className="glass-panel flex items-center justify-between p-4">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">Section Overlays</p>
            <p className="text-sm text-[#4b4237]">
              Showing semantic regions for "{selectedTitle}"
            </p>
          </div>
          <label className="flex cursor-pointer items-center gap-2">
            <span className="text-xs uppercase tracking-[0.2em] text-[#4b4237]">Show boxes</span>
            <input
              type="checkbox"
              checked={showBoxes}
              onChange={(e) => onToggleBoxes(e.target.checked)}
              className="h-4 w-4 cursor-pointer rounded border-[#2c2620]/30"
            />
          </label>
        </div>
      )}

      {/* Overlay Mode Toggle */}
      <div className="glass-panel p-4">
        <details className="group">
          <summary className="cursor-pointer text-xs uppercase tracking-[0.2em] text-[#6b8b6f] transition hover:text-[#5a7a5e]">
            Advanced ▾
          </summary>
          <div className="mt-3 flex items-center gap-3">
            <span className="text-sm text-[#4b4237]">Overlay detail:</span>
            <div className="flex gap-2">
              <button
                onClick={() => setOverlayMode("sections")}
                className={`rounded-full px-3 py-1.5 text-xs font-medium uppercase tracking-wider transition ${
                  overlayMode === "sections"
                    ? "bg-[#6b8b6f] text-white shadow-sm"
                    : "border border-[#2c2620]/15 bg-white text-[#2c2620] hover:border-[#6b8b6f]/30"
                }`}
              >
                Sections
              </button>
              <button
                onClick={() => setOverlayMode("tokens")}
                className={`rounded-full px-3 py-1.5 text-xs font-medium uppercase tracking-wider transition ${
                  overlayMode === "tokens"
                    ? "bg-[#6b8b6f] text-white shadow-sm"
                    : "border border-[#2c2620]/15 bg-white text-[#2c2620] hover:border-[#6b8b6f]/30"
                }`}
              >
                Tokens
              </button>
            </div>
          </div>
          <p className="mt-2 text-xs text-[#6b8b6f]">
            {overlayMode === "sections"
              ? "Shows 3 clean boxes (Title, Ingredients, Instructions) for the selected recipe only."
              : "Shows all raw token-level boxes from the ML model output."}
          </p>
        </details>
      </div>

      {/* Overlay Viewer */}
      <div className="paper-card flex flex-col gap-4 p-4">
        {overlayMode === "sections" ? (
          <SectionOverlayViewer
            imageUrl={imageUrl}
            imageSize={imageSize}
            tokens={tokens}
            selectedTitle={selectedTitle}
            showBoxes={showBoxes}
          />
        ) : (
          <DocOverlayViewer
            imageUrl={imageUrl}
            imageSize={imageSize}
            tokens={filtered}
            visibleLabels={selectedLabels}
            showBoxes={showBoxes}
            confidenceThreshold={confidence}
          />
        )}
        {overlayMode === "tokens" && <AiLegend entries={legendEntries} />}
      </div>

      {/* Stats Summary */}
      <div className="glass-panel p-4">
        <p className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">Token Statistics</p>
        <div className="mt-2 grid gap-2 text-sm text-[#2c2620] md:grid-cols-3">
          <div>
            <span className="font-semibold">{tokens.length}</span> total tokens
          </div>
          <div>
            <span className="font-semibold">{filtered.length}</span> visible after filters
          </div>
          <div>
            <span className="font-semibold">{overlayLabels.length}</span> unique labels
          </div>
        </div>
        <div className="mt-3 border-t border-[#2c2620]/10 pt-3">
          <p className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">Labels Detected</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {overlayLabels.map((label) => (
              <span
                key={label}
                className="rounded-full bg-[#2c2620]/5 px-3 py-1 text-xs font-medium text-[#2c2620]"
              >
                {label}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
