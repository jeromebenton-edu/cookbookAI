import path from "path";
import { promises as fs } from "fs";
import { Recipe, RecipeAI, RecipeToken } from "./types";

type BostonRecipe = {
  id: string;
  book: string;
  year: number;
  title: string;
  category: string;
  tags: string[];
  time: { prep: string; cook: string; total: string };
  servings: string;
  ingredients: string[];
  instructions: string[];
  notes?: string[];
  source: { page: number; pdf_url: string };
  ai: {
    page_image: string;
    tokens?: RecipeToken[];
    field_confidence?: {
      title?: number;
      ingredients?: number;
      instructions?: number;
    };
  };
};

type RecipeIndexEntry = {
  id: string;
  title: string;
  category: string;
  tags: string[];
};

type RecipeIndex = {
  collection: string;
  year: number;
  recipes: RecipeIndexEntry[];
};

const recipesDir = path.join(process.cwd(), "public", "recipes", "boston");
const indexPath = path.join(recipesDir, "index.json");
const defaultPageSize = { width: 900, height: 1200 };

async function loadIndex(): Promise<RecipeIndex> {
  const raw = await fs.readFile(indexPath, "utf-8");
  return JSON.parse(raw) as RecipeIndex;
}

function parseTimeToMinutes(label: string): number {
  const hoursMatch = label.match(/(\\d+)\\s*hr/);
  const minutesMatch = label.match(/(\\d+)\\s*min/);
  const soakMatch = label.match(/(\\d+)\\s*hr\\s*soak/);

  if (soakMatch && !hoursMatch) {
    return parseInt(soakMatch[1], 10) * 60;
  }

  const hours = hoursMatch ? parseInt(hoursMatch[1], 10) : 0;
  const minutes = minutesMatch ? parseInt(minutesMatch[1], 10) : 0;

  if (hours === 0 && minutes === 0) {
    return 45;
  }

  return hours * 60 + minutes;
}

function mapBostonRecipe(data: BostonRecipe): Recipe {
  const fieldConfidence = {
    title: data.ai.field_confidence?.title ?? 1.0,
    ingredients: data.ai.field_confidence?.ingredients ?? 1.0,
    instructions: data.ai.field_confidence?.instructions ?? 1.0,
    servings: 1.0,
    time: 1.0
  };
  const confidenceValues = Object.values(fieldConfidence);
  const confidence =
    confidenceValues.reduce((sum, value) => sum + value, 0) /
    confidenceValues.length;

  const totalMinutes = parseTimeToMinutes(data.time.total);

  const ai: RecipeAI = {
    pageImage: data.ai.page_image,
    pageSize: defaultPageSize,
    tokens: data.ai.tokens ?? [],
    fieldConfidence,
    raw: data as unknown as Record<string, unknown>
  };

  return {
    id: data.id,
    book: data.book,
    year: data.year,
    title: data.title,
    category: data.category,
    tags: data.tags,
    servings: data.servings,
    time: {
      totalMinutes,
      label: data.time.total,
      breakdown: data.time
    },
    ingredients: data.ingredients,
    instructions: data.instructions,
    notes: data.notes,
    confidence,
    source: {
      book: data.book,
      page: data.source.page
    },
    ai
  };
}

async function loadRecipeFile(id: string): Promise<Recipe | null> {
  const filePath = path.join(recipesDir, `${id}.json`);
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    const data = JSON.parse(raw) as BostonRecipe;
    return mapBostonRecipe(data);
  } catch (error) {
    return null;
  }
}

export async function loadRecipes(): Promise<Recipe[]> {
  const index = await loadIndex();
  const recipes = await Promise.all(
    index.recipes.map(async (entry) => {
      const recipe = await loadRecipeFile(entry.id);
      if (!recipe) {
        throw new Error(`Recipe file missing for id ${entry.id}`);
      }
      return recipe;
    })
  );
  return recipes;
}

export async function loadRecipeById(id: string): Promise<Recipe | null> {
  return loadRecipeFile(id);
}

export async function loadRecipeIds(): Promise<string[]> {
  const index = await loadIndex();
  return index.recipes.map((entry) => entry.id);
}
