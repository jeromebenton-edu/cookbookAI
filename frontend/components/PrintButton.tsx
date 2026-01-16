"use client";

export default function PrintButton() {
  return (
    <button
      type="button"
      onClick={() => window.print()}
      className="print-hidden inline-flex items-center gap-2 rounded-full border border-[#2c2620]/20 bg-white px-4 py-2 text-xs uppercase tracking-[0.2em] text-[#2c2620] shadow-sm transition hover:-translate-y-[1px] hover:shadow-md"
    >
      Print Recipe
    </button>
  );
}
