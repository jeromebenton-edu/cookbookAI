import Link from "next/link";
import { Recipe } from "../lib/types";

export default function RecipeCard({ recipe }: { recipe: Recipe }) {
  const timeLabel = recipe.time.label ?? `${recipe.time.totalMinutes} min`;
  return (
    <Link
      href={`/recipes/${recipe.id}`}
      className="paper-card group flex h-full flex-col justify-between gap-6 p-6 transition hover:-translate-y-1"
    >
      <div className="flex flex-col gap-3">
        <div className="flex flex-wrap gap-2">
          <span className="tag-pill">{recipe.category}</span>
          {recipe.tags.slice(0, 2).map((tag) => (
            <span key={tag} className="tag-pill">
              {tag}
            </span>
          ))}
        </div>
        <div>
          <h3 className="display-font text-2xl font-semibold text-balance">
            {recipe.title}
          </h3>
          {recipe.description ? (
            <p className="mt-2 text-sm text-[#4b4237]">
              {recipe.description}
            </p>
          ) : null}
        </div>
      </div>
      <div className="flex items-center justify-between text-xs uppercase tracking-[0.22em] text-[#6b8b6f]">
        <span>{timeLabel}</span>
        <span>{Math.round(recipe.confidence * 100)}% AI</span>
      </div>
    </Link>
  );
}
