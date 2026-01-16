import { cn } from "../../lib/utils";

type ConfidenceBarProps = {
  label: string;
  value?: number;
};

export default function ConfidenceBar({ label, value }: ConfidenceBarProps) {
  const safeValue = typeof value === "number" ? value : 0;
  const percent = Math.round(safeValue * 100);
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">
        <span>{label}</span>
        <span>{percent}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-[#e7dccb]">
        <div
          className={cn("h-full rounded-full bg-[#b8793b] transition-all")}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}
