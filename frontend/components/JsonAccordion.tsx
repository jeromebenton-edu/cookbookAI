export default function JsonAccordion({
  title,
  data
}: {
  title: string;
  data: Record<string, unknown>;
}) {
  return (
    <details className="rounded-2xl border border-[#2c2620]/10 bg-white/70 p-4 text-sm">
      <summary className="cursor-pointer text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">
        {title}
      </summary>
      <pre className="mt-3 max-h-80 overflow-auto whitespace-pre-wrap rounded-xl bg-[#f7efe3] p-4 text-xs text-[#2c2620]">
        {JSON.stringify(data, null, 2)}
      </pre>
    </details>
  );
}
