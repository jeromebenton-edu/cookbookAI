"use client";

import { useMemo, useState } from "react";
import { Recipe, RecipeToken } from "../lib/types";
import RecipeDetail from "./RecipeDetail";

const mockRecipePath = "/recipes/boston/oatmeal-cookies.json";

type AnalyzeResponse = {
  recipe: {
    title?: string;
    ingredients?: string[];
    instructions?: string[];
    notes?: string[];
    servings?: string;
    time?: { totalMinutes?: number; activeMinutes?: number };
    field_confidence?: Record<string, number>;
  };
  tokens?: RecipeToken[];
  image?: { base64: string; width: number; height: number };
  meta?: { mock?: boolean };
};

export default function UploadDropzone() {
  const [fileName, setFileName] = useState<string | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "done" | "error">(
    "idle"
  );
  const [recipe, setRecipe] = useState<Recipe | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL;
  const apiEnabled = Boolean(apiBase);

  const handleFile = async (file: File) => {
    setFileName(file.name);
    setStatus("loading");
    setMessage(null);

    if (!apiEnabled) {
      try {
        setMessage(
          "Backend API not configured. Showing a mock parse result from the archive."
        );
        const mockRecipe = await fetch(mockRecipePath).then((res) => res.json());
        setRecipe(mockRecipe);
        setStatus("done");
        return;
      } catch (error) {
        setStatus("error");
        setMessage("Mock recipe fetch failed.");
        return;
      }
    }

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${apiBase}/analyze-recipe`, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error("Failed to analyze recipe");
      }

      const data = (await response.json()) as AnalyzeResponse;
      const mapped = mapAnalyzeResponseToRecipe(data);
      setRecipe(mapped);
      if (data.meta?.mock) {
        setMessage("Backend is running in MOCK_INFERENCE mode.");
      }
      setStatus("done");
    } catch (error) {
      setStatus("error");
      setMessage("Upload failed. Try again or enable mock mode.");
    }
  };

  const helperText = useMemo(() => {
    if (!apiEnabled) {
      return "Set NEXT_PUBLIC_API_BASE_URL to connect the backend.";
    }
    return "Upload a PDF or image to run LayoutLMv3 parsing.";
  }, [apiEnabled]);

  return (
    <div className="flex flex-col gap-8">
      <div className="paper-card flex flex-col gap-4 p-6">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">
            {apiEnabled ? "Live Inference" : "Mock Mode"}
          </p>
          <h2 className="display-font text-3xl font-semibold">Upload a Recipe</h2>
          <p className="text-sm text-[#4b4237]">{helperText}</p>
        </div>
        <label className="flex cursor-pointer flex-col items-center gap-3 rounded-2xl border border-dashed border-[#2c2620]/30 bg-white/70 px-6 py-10 text-center text-sm">
          <input
            type="file"
            className="hidden"
            accept=".pdf,image/*"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) {
                void handleFile(file);
              }
            }}
          />
          <span className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">
            Drop or browse
          </span>
          <span className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">
            Click to upload
          </span>
          {fileName ? (
            <span className="text-sm text-[#2c2620]">{fileName}</span>
          ) : null}
        </label>
        {status === "loading" ? (
          <p className="text-sm text-[#4b4237]">Analyzing...</p>
        ) : null}
        {message ? (
          <p className="rounded-xl border border-[#2c2620]/10 bg-[#f7efe3] px-4 py-3 text-xs text-[#4b4237]">
            {message}
          </p>
        ) : null}
        {status === "error" ? (
          <p className="text-sm text-[#b8793b]">
            Something went wrong. Check the backend logs.
          </p>
        ) : null}
      </div>

      {recipe ? <RecipeDetail recipe={recipe} /> : null}
    </div>
  );
}

function mapAnalyzeResponseToRecipe(data: AnalyzeResponse): Recipe {
  const fieldConfidence = {
    title: data.recipe.field_confidence?.title ?? 0.72,
    ingredients: data.recipe.field_confidence?.ingredients ?? 0.68,
    instructions: data.recipe.field_confidence?.instructions ?? 0.64,
    servings: data.recipe.field_confidence?.servings ?? 0.55,
    time: data.recipe.field_confidence?.time ?? 0.5
  };
  const confidenceValues = Object.values(fieldConfidence);
  const averageConfidence =
    confidenceValues.reduce((sum, value) => sum + value, 0) /
    confidenceValues.length;

  return {
    id: "uploaded",
    title: data.recipe.title ?? "Untitled Recipe",
    category: "Uploads",
    tags: ["user", "upload"],
    servings: data.recipe.servings ?? "Serves 2-4",
    time: {
      totalMinutes: data.recipe.time?.totalMinutes ?? 45,
      activeMinutes: data.recipe.time?.activeMinutes
    },
    ingredients: data.recipe.ingredients ?? [],
    instructions: data.recipe.instructions ?? [],
    notes: data.recipe.notes ?? [],
    confidence: averageConfidence,
    source: { book: "User Upload", page: 1 },
    ai: {
      pageImage: data.image?.base64
        ? `data:image/png;base64,${data.image.base64}`
        : "/recipes/boston/pages/placeholder-1.png",
      pageSize: {
        width: data.image?.width ?? 900,
        height: data.image?.height ?? 1200
      },
      tokens: data.tokens ?? [],
      fieldConfidence,
      raw: data as unknown as Record<string, unknown>
    }
  };
}
