"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { Recipe } from "../lib/types";
import RecipeFilters, { RecipeFilterState } from "./RecipeFilters";
import SearchBar from "./SearchBar";
import Link from "next/link";

const emptyFilters: RecipeFilterState = {
  category: "",
  tag: "",
  ingredient: "",
  timeRange: "",
  sort: "title"
};

type RecipeSearchUIProps = {
  recipes: Recipe[];
  initialQuery?: string;
  showBrowseCTA?: boolean;
};

export default function RecipeSearchUI({
  recipes,
  initialQuery = "",
  showBrowseCTA = false
}: RecipeSearchUIProps) {
  const router = useRouter();
  const [filters, setFilters] = useState<RecipeFilterState>(emptyFilters);
  const [query, setQuery] = useState(initialQuery);

  const categories = useMemo(() => {
    return Array.from(new Set(recipes.map((recipe) => recipe.category))).sort();
  }, [recipes]);

  const tags = useMemo(() => {
    return Array.from(
      new Set(recipes.flatMap((recipe) => recipe.tags))
    ).sort();
  }, [recipes]);

  const handleSearch = (newQuery: string) => {
    setQuery(newQuery);
    // Navigate to recipes page with query params
    const params = new URLSearchParams();
    if (newQuery.trim()) params.set("q", newQuery);
    if (filters.category) params.set("category", filters.category);
    if (filters.tag) params.set("tag", filters.tag);
    if (filters.ingredient) params.set("ingredient", filters.ingredient);
    if (filters.timeRange) params.set("time", filters.timeRange);
    if (filters.sort !== "title") params.set("sort", filters.sort);

    const queryString = params.toString();
    router.push(queryString ? `/recipes?${queryString}` : "/recipes");
  };

  return (
    <section className="flex flex-col gap-10">
      <div className="flex flex-col gap-4">
        <h1 className="display-font text-4xl font-semibold text-balance">
          Browse the Boston Kitchen
        </h1>
        <p className="max-w-2xl text-sm text-[#4b4237]">
          Search by ingredient, filter by category, and compare AI confidence
          across the 1918 recipe archive.
        </p>
      </div>

      <SearchBar
        initialValue={query}
        onSearch={handleSearch}
        placeholder="Search titles or ingredients"
      />

      <RecipeFilters
        filters={filters}
        categories={categories}
        tags={tags}
        onChange={setFilters}
      />

      {showBrowseCTA && (
        <div className="paper-card flex flex-col items-center gap-4 p-8 text-center">
          <p className="display-font text-2xl">Ready to explore?</p>
          <p className="text-sm text-[#4b4237]">
            Browse all recipes with filters and search.
          </p>
          <Link
            href="/recipes"
            className="rounded-full bg-[#2c2620] px-6 py-3 text-xs uppercase tracking-[0.25em] text-[#f7efe3] shadow transition hover:-translate-y-[1px] hover:shadow-md"
          >
            Browse all recipes →
          </Link>
        </div>
      )}
    </section>
  );
}
