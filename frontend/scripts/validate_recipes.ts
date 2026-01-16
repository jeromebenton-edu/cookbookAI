import { promises as fs } from "fs";
import path from "path";

type ValidationIssue = { file: string; message: string };

type RecipeIndex = {
  collection: string;
  year: number;
  recipes: Array<{
    id: string;
    title: string;
    category: string;
    tags: string[];
  }>;
};

const recipesDir = path.join(process.cwd(), "public", "recipes", "boston");
const indexPath = path.join(recipesDir, "index.json");

async function loadJson<T>(filePath: string): Promise<T> {
  const raw = await fs.readFile(filePath, "utf-8");
  return JSON.parse(raw) as T;
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function validateRecipeShape(data: any, file: string): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  const requiredStringFields = [
    "id",
    "book",
    "title",
    "category",
    "servings"
  ];

  for (const field of requiredStringFields) {
    if (typeof data?.[field] !== "string" && field !== "year") {
      issues.push({ file, message: `Missing or invalid field '${field}'` });
    }
  }

  if (typeof data?.year !== "number") {
    issues.push({ file, message: "Missing or invalid field 'year'" });
  }

  if (!isStringArray(data?.tags)) {
    issues.push({ file, message: "Missing or invalid field 'tags'" });
  }

  if (!isStringArray(data?.ingredients) || data.ingredients.length === 0) {
    issues.push({ file, message: "Missing ingredients array" });
  }

  if (!isStringArray(data?.instructions) || data.instructions.length === 0) {
    issues.push({ file, message: "Missing instructions array" });
  }

  if (!data?.time || typeof data.time.total !== "string") {
    issues.push({ file, message: "Missing time.total" });
  }

  if (typeof data?.source?.page !== "number") {
    issues.push({ file, message: "Missing source.page" });
  }

  if (typeof data?.source?.pdf_url !== "string") {
    issues.push({ file, message: "Missing source.pdf_url" });
  }

  if (typeof data?.ai?.page_image !== "string") {
    issues.push({ file, message: "Missing ai.page_image" });
  }

  if (!Array.isArray(data?.ai?.tokens)) {
    issues.push({ file, message: "ai.tokens must be an array (can be empty)" });
  }

  const fc = data?.ai?.field_confidence;
  if (
    !fc ||
    typeof fc.title !== "number" ||
    typeof fc.ingredients !== "number" ||
    typeof fc.instructions !== "number"
  ) {
    issues.push({ file, message: "ai.field_confidence must include title, ingredients, instructions" });
  }

  return issues;
}

async function main() {
  const files = (await fs.readdir(recipesDir))
    .filter((file) => file.startsWith("bcsb_") && file.endsWith(".json"));

  const index = await loadJson<RecipeIndex>(indexPath);
  const indexIds = new Set(index.recipes.map((entry) => entry.id));
  const issues: ValidationIssue[] = [];
  const ids = new Set<string>();

  for (const file of files) {
    const fullPath = path.join(recipesDir, file);
    let data: any;
    try {
      data = await loadJson(fullPath);
    } catch (error) {
      issues.push({ file, message: "JSON parse error" });
      continue;
    }

    const recipeIssues = validateRecipeShape(data, file);
    issues.push(...recipeIssues);

    if (ids.has(data.id)) {
      issues.push({ file, message: `Duplicate id '${data.id}'` });
    } else {
      ids.add(data.id);
    }

    if (!indexIds.has(data.id)) {
      issues.push({ file, message: "Missing from index.json" });
    }
  }

  for (const entry of index.recipes) {
    if (!ids.has(entry.id)) {
      issues.push({ file: "index.json", message: `Index references missing id '${entry.id}'` });
    }
  }

  if (issues.length === 0) {
    console.log(`✅ Recipes validated: ${files.length} files, no issues.`);
    return;
  }

  console.log(`⚠️ Found ${issues.length} issue(s) across ${files.length} file(s):`);
  for (const issue of issues) {
    console.log(`- ${issue.file}: ${issue.message}`);
  }
  process.exitCode = 1;
}

main().catch((error) => {
  console.error("Validation failed:", error);
  process.exitCode = 1;
});
