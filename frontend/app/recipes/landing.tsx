import Link from "next/link";
import RecipeSearchUI from "../../components/RecipeSearchUI";
import { loadRecipes } from "../../lib/recipes";
import { getDemoBundle } from "../../lib/api/parse";

export default async function LandingPage({
  searchParams
}: {
  searchParams?: { query?: string };
}) {
  const recipes = await loadRecipes();
  const demo = await getDemoBundle({ timeoutMs: 8000 }).catch(() => null);
  const defaultDemoPage = demo?.default_page_id ?? demo?.default_page ?? null;
  const demoHref = defaultDemoPage ? `/demo?page=${defaultDemoPage}` : "/demo";

  return (
    <div className="flex flex-col gap-14">
      <section className="grid gap-10 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="flex flex-col gap-6">
          <div className="flex flex-col gap-4">
            <p className="text-xs uppercase tracking-[0.3em] text-[#6b8b6f]">
              Boston Cooking-School Cook Book
            </p>
            <h1 className="display-font text-5xl font-semibold text-balance">
              A polished cookbook with an AI parse layer for 1896 recipes.
            </h1>
            <p className="max-w-xl text-sm text-[#4b4237]">
              Browse cleaned, searchable recipes and jump into the AI overlay to see how LayoutLMv3 labels and groups
              each page.
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <Link
              href={demoHref}
              className="rounded-full bg-[#2c2620] px-6 py-3 text-xs uppercase tracking-[0.25em] text-[#f7efe3] shadow transition hover:-translate-y-[1px] hover:shadow-md"
            >
              SEE AI PARSE DEMO
            </Link>
            <Link
              href="/about"
              className="rounded-full border border-[#2c2620]/20 px-6 py-3 text-xs uppercase tracking-[0.25em] text-[#2c2620]"
            >
              About the project
            </Link>
          </div>
        </div>
        <div className="glass-panel flex flex-col gap-6 p-8">
          <p className="text-xs uppercase tracking-[0.25em] text-[#6b8b6f]">
            AI Parse Highlights
          </p>
          <div className="flex flex-col gap-4">
            <div className="rounded-2xl border border-[#2c2620]/10 bg-white/80 p-4">
              <p className="display-font text-2xl">Token overlays</p>
              <p className="text-sm text-[#4b4237]">
                Visualize extracted titles, ingredients, and instructions directly on the scanned page.
              </p>
            </div>
            <div className="rounded-2xl border border-[#2c2620]/10 bg-white/80 p-4">
              <p className="display-font text-2xl">Confidence bars</p>
              <p className="text-sm text-[#4b4237]">
                See where the model is strong and where the archive needs more fine-tuning.
              </p>
            </div>
            <div className="rounded-2xl border border-[#2c2620]/10 bg-white/80 p-4">
              <p className="display-font text-2xl">Raw JSON</p>
              <p className="text-sm text-[#4b4237]">
                Inspect the structured output used to render cook-friendly views.
              </p>
            </div>
          </div>
        </div>
      </section>

      <div className="rounded-2xl border border-[#d4a574]/30 bg-[#f9f3eb] p-6">
        <p className="text-sm text-[#4b4237]">
          <span className="font-semibold">Educational purposes only:</span> These recipes are digitized from historical cookbooks for educational and research purposes. They have not been tested in modern kitchens. Cook at your own risk!
        </p>
      </div>

      <RecipeSearchUI recipes={recipes} initialQuery={searchParams?.query ?? ""} showBrowseCTA={true} />

      <section className="paper-card flex flex-col gap-6 p-10 md:flex-row md:items-center md:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.25em] text-[#6b8b6f]">
            Featured AI Pages
          </p>
          <h2 className="display-font text-3xl font-semibold">
            Jump straight to the AI overlay experience.
          </h2>
          <p className="text-sm text-[#4b4237]">
            Open the demo on the first recipe-heavy page to see tokens, bounding boxes, and extracted fields together.
          </p>
        </div>
        <Link
          href={demoHref}
          className="rounded-full bg-[#2c2620] px-6 py-3 text-xs uppercase tracking-[0.25em] text-[#f7efe3] shadow transition hover:-translate-y-[1px] hover:shadow-md"
        >
          View Featured Pages
        </Link>
      </section>
    </div>
  );
}
