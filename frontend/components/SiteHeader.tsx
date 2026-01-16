import Link from "next/link";

const navItems = [
  { label: "Recipes", href: "/recipes" },
  { label: "Try the AI", href: "/demo" },
  { label: "About", href: "/about" }
];

export default function SiteHeader() {
  return (
    <header className="no-print mx-auto flex w-full max-w-6xl items-center justify-between px-6 pt-10">
      <Link href="/" className="group flex items-center gap-3">
        <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[#2c2620] text-xs uppercase tracking-[0.3em] text-[#f7efe3] shadow-lg">
          AI
        </div>
        <div>
          <p className="display-font text-xl font-semibold">CookbookAI</p>
          <p className="text-xs uppercase tracking-[0.25em] text-[#6b8b6f]">1918 Archive</p>
        </div>
      </Link>
      <nav className="hidden items-center gap-6 text-sm font-medium text-[#4b4237] md:flex">
        {navItems.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className="transition hover:text-[#1f1b16]"
          >
            {item.label}
          </Link>
        ))}
      </nav>
      <div className="md:hidden">
        <Link
          href="/recipes"
          className="rounded-full border border-[#2c2620]/20 px-4 py-2 text-xs uppercase tracking-[0.2em]"
        >
          Explore
        </Link>
      </div>
    </header>
  );
}
