import { notFound } from "next/navigation";
import RecipeDetail from "../../../components/RecipeDetail";
import { loadRecipeById, loadRecipeIds } from "../../../lib/recipes";

export async function generateStaticParams() {
  const ids = await loadRecipeIds();
  return ids.map((id) => ({ id }));
}

export default async function RecipeDetailPage({
  params
}: {
  params: { id: string };
}) {
  const recipe = await loadRecipeById(params.id);
  if (!recipe) {
    notFound();
  }

  return <RecipeDetail recipe={recipe} />;
}
