import type { ExtractedRecipe } from "./types";

export function recipeToMarkdown(recipe: ExtractedRecipe): string {
  if (recipe.is_recipe_page === false) {
    const msg = recipe.message || "No recipe detected on this page.";
    return `# Demo Recipe\n\n${msg}`;
  }

  const lines: string[] = [];
  const title = recipe.title?.trim() || "Recipe";
  lines.push(`# ${title}`);
  lines.push("");
  lines.push("## Ingredients");
  const ingredients = recipe.ingredients?.length ? recipe.ingredients : ["(not detected)"];
  ingredients.forEach((ing) => lines.push(`- ${ing}`));
  lines.push("");
  lines.push("## Instructions");
  const steps = recipe.instructions?.length ? recipe.instructions : ["(not detected)"];
  steps.forEach((step, idx) => lines.push(`${idx + 1}. ${step}`));

  return lines.join("\n");
}
