"use client";

import { useMemo } from "react";
import { getLabelColor } from "./constants";

type AiControlsPanelProps = {
  labels: string[];
  selected: Set<string>;
  onToggleLabel: (label: string) => void;
  onSelectAll: () => void;
  onClear: () => void;
  showBoxes: boolean;
  onToggleBoxes: (value: boolean) => void;
  confidence: number;
  onConfidenceChange: (value: number) => void;
  disabled?: boolean;
};

export default function AiControlsPanel({
  labels,
  selected,
  onToggleLabel,
  onSelectAll,
  onClear,
  showBoxes,
  onToggleBoxes,
  confidence,
  onConfidenceChange,
  disabled
}: AiControlsPanelProps) {
  const allSelected = useMemo(
    () => labels.every((label) => selected.has(label)) && labels.length > 0,
    [labels, selected]
  );

  return (
    <div className="glass-panel flex flex-col gap-4 p-5 text-sm text-[#2c2620]">
      <div className="flex items-center justify-between gap-3">
        <p className="text-xs uppercase tracking-[0.22em] text-[#6b8b6f]">Overlay Controls</p>
        <label className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-[#2c2620]">
          <input
            type="checkbox"
            className="h-4 w-4 rounded border-[#2c2620]/30 text-[#b8793b]"
            checked={showBoxes}
            onChange={(event) => onToggleBoxes(event.target.checked)}
            disabled={disabled}
          />
          Show boxes
        </label>
      </div>

      <div className="flex flex-wrap gap-2 text-xs">
        <button
          type="button"
          className="rounded-full border border-[#2c2620]/15 bg-white px-3 py-1 shadow-sm transition hover:-translate-y-[1px] hover:shadow-md disabled:opacity-50"
          onClick={onSelectAll}
          disabled={disabled}
        >
          Select all
        </button>
        <button
          type="button"
          className="rounded-full border border-[#2c2620]/15 bg-white px-3 py-1 shadow-sm transition hover:-translate-y-[1px] hover:shadow-md disabled:opacity-50"
          onClick={onClear}
          disabled={disabled}
        >
          Clear all
        </button>
      </div>

      <div className="flex flex-col gap-2">
        <p className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">Labels</p>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {labels.map((label) => {
            const color = getLabelColor(label);
            return (
              <label
                key={label}
                className="flex items-center gap-2 rounded-xl border border-[#2c2620]/10 bg-white/70 px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-[#2c2620] shadow-sm"
              >
                <span className="h-3 w-3 rounded-full" style={{ backgroundColor: color }} />
                <input
                  type="checkbox"
                  className="h-4 w-4 rounded border-[#2c2620]/30 text-[#b8793b]"
                  checked={selected.has(label)}
                  onChange={() => onToggleLabel(label)}
                  disabled={disabled}
                />
                <span>{label.replace(/_/g, " ")}</span>
              </label>
            );
          })}
        </div>
        {labels.length === 0 ? (
          <p className="text-xs text-[#6b8b6f]">No labels available yet.</p>
        ) : null}
        <div className="flex items-center gap-3 pt-2">
          <span className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">Confidence</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={confidence}
            onChange={(event) => onConfidenceChange(Number(event.target.value))}
            className="flex-1 accent-[#b8793b]"
            disabled={disabled}
          />
          <span className="text-xs font-semibold text-[#2c2620]">
            ≥ {Math.round(confidence * 100)}%
          </span>
        </div>
        {allSelected ? null : (
          <p className="text-[10px] uppercase tracking-[0.18em] text-[#6b8b6f]">
            Showing selected labels only.
          </p>
        )}
      </div>
    </div>
  );
}
