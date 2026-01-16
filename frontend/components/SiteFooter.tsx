import Link from "next/link";

export default function SiteFooter() {
  return (
    <footer className="no-print mx-auto w-full max-w-6xl px-6 pb-12">
      <div className="flex flex-col gap-6 rounded-3xl border border-[#2c2620]/10 bg-white/60 px-8 py-8 text-sm text-[#4b4237] shadow-[0_18px_40px_rgba(31,27,22,0.1)] md:flex-row md:items-center md:justify-between">
        <div>
          <p className="display-font text-lg font-semibold text-[#1f1b16]">
            CookbookAI Lab
          </p>
          <p className="max-w-md text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">
            From archived recipes to structured data with LayoutLMv3.
          </p>
        </div>
        <div className="flex gap-4 text-xs uppercase tracking-[0.22em]">
          <Link className="transition hover:text-[#1f1b16]" href="/recipes">
            Browse
          </Link>
          <Link className="transition hover:text-[#1f1b16]" href="/try">
            Try AI
          </Link>
          <Link className="transition hover:text-[#1f1b16]" href="/about">
            About
          </Link>
        </div>
      </div>
      <p className="mt-6 text-center text-xs text-[#6b8b6f]">
        Built for the CookbookAI portfolio project. Boston Cooking-School Cook Book (1918).
      </p>
    </footer>
  );
}
