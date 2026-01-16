import { getLabelColor } from "./constants";

export type LegendEntry = {
  label: string;
  count: number;
};

type AiLegendProps = {
  entries: LegendEntry[];
};

export default function AiLegend({ entries }: AiLegendProps) {
  if (entries.length === 0) {
    return (
      <div className="flex items-center justify-between rounded-2xl border border-[#2c2620]/10 bg-white/70 px-4 py-3 text-xs text-[#6b8b6f]">
        <span>Legend</span>
        <span>No tokens yet</span>
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-[#2c2620]/10 bg-white/80 p-4 text-xs text-[#2c2620] shadow-sm">
      <p className="mb-3 text-[11px] uppercase tracking-[0.2em] text-[#6b8b6f]">Legend</p>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {entries.map((entry) => {
          const color = getLabelColor(entry.label);
          return (
            <div key={entry.label} className="flex items-center gap-2 rounded-xl bg-white/60 px-3 py-2 shadow-sm">
              <span className="h-3 w-3 rounded-full" style={{ backgroundColor: color }} />
              <span className="truncate uppercase tracking-[0.14em] text-[#2c2620]">{entry.label.replace(/_/g, " ")}</span>
              <span className="ml-auto text-[11px] text-[#6b8b6f]">{entry.count}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
