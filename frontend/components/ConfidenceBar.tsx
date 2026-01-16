export default function ConfidenceBar({
  label,
  value
}: {
  label: string;
  value: number;
}) {
  const percent = Math.round(value * 100);
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">
        <span>{label}</span>
        <span>{percent}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-[#e7dccb]">
        <div
          className="h-full rounded-full bg-[#b8793b]"
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}
