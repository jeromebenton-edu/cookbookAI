import RecipeBrowser from "../../components/RecipeBrowser";
import { loadRecipes } from "../../lib/recipes";

export default async function RecipesPage({
  searchParams
}: {
  searchParams?: {
    q?: string;
    category?: string;
    tag?: string;
    ingredient?: string;
    time?: string;
    sort?: string;
  };
}) {
  const recipes = await loadRecipes();

  // Map URL params to initial filter state
  const initialFilters = {
    category: searchParams?.category ?? "",
    tag: searchParams?.tag ?? "",
    ingredient: searchParams?.ingredient ?? "",
    timeRange: searchParams?.time ?? "",
    sort: (searchParams?.sort as "title" | "category" | "confidence") ?? "title"
  };

  return (
    <div className="flex flex-col gap-6">
      <RecipeBrowser
        recipes={recipes}
        initialQuery={searchParams?.q ?? ""}
        initialFilters={initialFilters}
      />
    </div>
  );
}
