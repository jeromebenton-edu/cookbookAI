import { ChangeEvent } from "react";

export type RecipeFilterState = {
  category: string;
  tag: string;
  ingredient: string;
  timeRange: string;
  sort: string;
};

type RecipeFiltersProps = {
  filters: RecipeFilterState;
  categories: string[];
  tags: string[];
  onChange: (next: RecipeFilterState) => void;
};

export default function RecipeFilters({
  filters,
  categories,
  tags,
  onChange
}: RecipeFiltersProps) {
  const handleChange = (key: keyof RecipeFilterState) =>
    (event: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
      onChange({ ...filters, [key]: event.target.value });
    };

  return (
    <div className="glass-panel grid gap-4 p-6 md:grid-cols-2 lg:grid-cols-5">
      <div className="flex flex-col gap-2">
        <label className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">
          Category
        </label>
        <select
          className="rounded-xl border border-[#2c2620]/10 bg-white/80 px-3 py-2 text-sm"
          value={filters.category}
          onChange={handleChange("category")}
        >
          <option value="">All</option>
          {categories.map((category) => (
            <option key={category} value={category}>
              {category}
            </option>
          ))}
        </select>
      </div>
      <div className="flex flex-col gap-2">
        <label className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">
          Tag
        </label>
        <select
          className="rounded-xl border border-[#2c2620]/10 bg-white/80 px-3 py-2 text-sm"
          value={filters.tag}
          onChange={handleChange("tag")}
        >
          <option value="">All</option>
          {tags.map((tag) => (
            <option key={tag} value={tag}>
              {tag}
            </option>
          ))}
        </select>
      </div>
      <div className="flex flex-col gap-2">
        <label className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">
          Ingredient
        </label>
        <input
          className="rounded-xl border border-[#2c2620]/10 bg-white/80 px-3 py-2 text-sm"
          placeholder="e.g. butter"
          value={filters.ingredient}
          onChange={handleChange("ingredient")}
        />
      </div>
      <div className="flex flex-col gap-2">
        <label className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">
          Time Range
        </label>
        <select
          className="rounded-xl border border-[#2c2620]/10 bg-white/80 px-3 py-2 text-sm"
          value={filters.timeRange}
          onChange={handleChange("timeRange")}
        >
          <option value="">Any</option>
          <option value="quick">Under 30 min</option>
          <option value="mid">30 - 60 min</option>
          <option value="slow">Over 60 min</option>
        </select>
      </div>
      <div className="flex flex-col gap-2">
        <label className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">
          Sort
        </label>
        <select
          className="rounded-xl border border-[#2c2620]/10 bg-white/80 px-3 py-2 text-sm"
          value={filters.sort}
          onChange={handleChange("sort")}
        >
          <option value="title">Title</option>
          <option value="category">Category</option>
          <option value="confidence">AI Confidence</option>
        </select>
      </div>
    </div>
  );
}
