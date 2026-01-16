import { Recipe } from "../lib/types";
import { cn } from "../lib/utils";

type RecipeMetaProps = {
  recipe: Recipe;
  className?: string;
};

export default function RecipeMeta({ recipe, className }: RecipeMetaProps) {
  const timeLabel = recipe.time.label ?? `${recipe.time.totalMinutes} min`;
  return (
    <div
      className={cn(
        "flex flex-col gap-3 rounded-2xl border border-[#2c2620]/10 bg-white/80 p-4 text-sm text-[#4b4237]",
        className
      )}
    >
      <div className="flex flex-wrap items-center gap-3 text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">
        <span>{recipe.category}</span>
        <span className="h-3 w-px bg-[#2c2620]/15" aria-hidden />
        <span>{timeLabel}</span>
        <span className="h-3 w-px bg-[#2c2620]/15" aria-hidden />
        <span>{recipe.servings}</span>
      </div>
      <div className="flex flex-wrap gap-2">
        {recipe.tags.map((tag) => (
          <span
            key={tag}
            className="rounded-full border border-[#2c2620]/15 bg-[#f7efe3] px-3 py-1 text-xs font-semibold uppercase tracking-[0.15em] text-[#4b4237]"
          >
            {tag}
          </span>
        ))}
      </div>
      <p className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">
        Source: {recipe.source.book} — Page {recipe.source.page}
      </p>
    </div>
  );
}
