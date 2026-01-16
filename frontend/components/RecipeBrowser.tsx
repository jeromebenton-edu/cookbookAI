"use client";

import { useMemo, useState } from "react";
import { Recipe } from "../lib/types";
import RecipeCard from "./RecipeCard";
import RecipeFilters, { RecipeFilterState } from "./RecipeFilters";
import SearchBar from "./SearchBar";

const emptyFilters: RecipeFilterState = {
  category: "",
  tag: "",
  ingredient: "",
  timeRange: "",
  sort: "title"
};

type RecipeBrowserProps = {
  recipes: Recipe[];
  initialQuery?: string;
  initialFilters?: Partial<RecipeFilterState>;
};

export default function RecipeBrowser({
  recipes,
  initialQuery = "",
  initialFilters
}: RecipeBrowserProps) {
  const [filters, setFilters] = useState<RecipeFilterState>({
    ...emptyFilters,
    ...initialFilters
  });
  const [query, setQuery] = useState(initialQuery);

  const categories = useMemo(() => {
    return Array.from(new Set(recipes.map((recipe) => recipe.category))).sort();
  }, [recipes]);

  const tags = useMemo(() => {
    return Array.from(
      new Set(recipes.flatMap((recipe) => recipe.tags))
    ).sort();
  }, [recipes]);

  const filteredRecipes = useMemo(() => {
    return recipes
      .filter((recipe) => {
        if (filters.category && recipe.category !== filters.category) {
          return false;
        }
        if (filters.tag && !recipe.tags.includes(filters.tag)) {
          return false;
        }
        if (filters.ingredient) {
          const ingredientQuery = filters.ingredient.toLowerCase();
          const matches = recipe.ingredients.some((ingredient) =>
            ingredient.toLowerCase().includes(ingredientQuery)
          );
          if (!matches) {
            return false;
          }
        }
        if (filters.timeRange) {
          const minutes = recipe.time.totalMinutes;
          if (filters.timeRange === "quick" && minutes > 30) {
            return false;
          }
          if (filters.timeRange === "mid" && (minutes < 30 || minutes > 60)) {
            return false;
          }
          if (filters.timeRange === "slow" && minutes <= 60) {
            return false;
          }
        }
        if (query.trim()) {
          const queryValue = query.toLowerCase();
          const inTitle = recipe.title.toLowerCase().includes(queryValue);
          const inIngredients = recipe.ingredients.some((ingredient) =>
            ingredient.toLowerCase().includes(queryValue)
          );
          if (!inTitle && !inIngredients) {
            return false;
          }
        }
        return true;
      })
      .sort((a, b) => {
        if (filters.sort === "category") {
          return a.category.localeCompare(b.category);
        }
        if (filters.sort === "confidence") {
          return b.confidence - a.confidence;
        }
        return a.title.localeCompare(b.title);
      });
  }, [filters, query, recipes]);

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
        onSearch={setQuery}
        placeholder="Search titles or ingredients"
      />

      <RecipeFilters
        filters={filters}
        categories={categories}
        tags={tags}
        onChange={setFilters}
      />

      <div className="grid gap-6 md:grid-cols-2">
        {filteredRecipes.map((recipe) => (
          <RecipeCard key={recipe.id} recipe={recipe} />
        ))}
      </div>

      {filteredRecipes.length === 0 ? (
        <div className="paper-card flex flex-col items-center gap-2 p-8 text-center text-sm text-[#4b4237]">
          <p className="display-font text-2xl">No matches yet</p>
          <p>Try clearing a filter or search for another ingredient.</p>
        </div>
      ) : null}
    </section>
  );
}
