import { cn } from "../../lib/utils";

type ConfidenceBadgeProps = {
  value: number;
};

export default function ConfidenceBadge({ value }: ConfidenceBadgeProps) {
  const label = value >= 0.9 ? "High" : value >= 0.75 ? "Medium" : "Low";
  const color = value >= 0.9 ? "bg-[#d9b07e]" : value >= 0.75 ? "bg-[#b9c9b1]" : "bg-[#f7efe3]";

  return (
    <span
      className={cn(
        "rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-[#2c2620]",
        color
      )}
    >
      {label} {Math.round(value * 100)}%
    </span>
  );
}
