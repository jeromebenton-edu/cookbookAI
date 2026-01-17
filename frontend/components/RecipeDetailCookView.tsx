import IngredientChecklist from "./IngredientChecklist";
import CorrectionExport from "./CorrectionExport";
import { Recipe, ExtractedRecipe } from "../lib/types";
import { useEffect, useMemo, useState } from "react";
import { getExtractedRecipe } from "../lib/api/parse";
import { diffLines } from "../lib/diff/recipeDiff";
import { useRecipeCorrection } from "../hooks/useRecipeCorrection";

export default function RecipeDetailCookView({ recipe }: { recipe: Recipe }) {
  const [extracted, setExtracted] = useState<ExtractedRecipe | null>(null);
  const [mode, setMode] = useState<"curated" | "compare">("curated");
  useEffect(() => {
    // best-effort; ignore errors
    getExtractedRecipe(recipe.source.page)
      .then(setExtracted)
      .catch(() => setExtracted(null));
  }, [recipe.source.page]);
  const correction = useRecipeCorrection(extracted);
  const ingDiff = useMemo(
    () => diffLines(recipe.ingredients, correction.state.ingredients.map((i) => i.text), 0.75),
    [recipe.ingredients, correction.state.ingredients]
  );
  const insDiff = useMemo(
    () => diffLines(recipe.instructions, correction.state.instructions.map((i) => i.text), 0.7),
    [recipe.instructions, correction.state.instructions]
  );

  const modeToggle = (
    <div className="print-hidden flex items-center gap-2">
      <div className="flex items-center rounded-full border border-[#2c2620]/15 bg-white/80 p-1 text-xs uppercase tracking-[0.2em] text-[#2c2620]">
        {(["curated", "compare"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`rounded-full px-3 py-1 transition ${mode === m ? "bg-[#2c2620] text-[#f7efe3]" : ""}`}
          >
            {m === "curated" ? "Curated" : "Compare and Correct"}
          </button>
        ))}
      </div>
    </div>
  );

  return (
    <div className="recipe-cook-grid grid gap-8">
      <div className="flex flex-wrap items-center justify-between gap-3">{modeToggle}</div>

      {mode === "curated" && (
        <div className="grid gap-8 lg:grid-cols-[1.05fr_1fr]">
          <section
            id="ingredients"
            className="paper-card flex flex-col gap-6 p-8 shadow-[0_20px_60px_rgba(31,27,22,0.08)]"
          >
            <div className="flex items-center justify-between gap-4">
              <div>
                <h2 className="display-font text-3xl font-semibold">Ingredients</h2>
                <p className="text-sm text-[#4b4237]">Check off as you prep.</p>
              </div>
            </div>
            <IngredientChecklist recipeId={recipe.id} ingredients={recipe.ingredients} />
          </section>

          <section
            id="steps"
            className="paper-card flex flex-col gap-6 p-8 shadow-[0_20px_60px_rgba(31,27,22,0.08)]"
          >
            <div className="flex items-center justify-between gap-4">
              <div>
                <h2 className="display-font text-3xl font-semibold">Instructions</h2>
                <p className="text-sm text-[#4b4237]">Step-by-step for a clean cook-through.</p>
              </div>
            </div>
            <ol className="list-decimal pl-5 text-base leading-relaxed text-[#2c2620]">
              {recipe.instructions.map((step, index) => (
                <li key={index} className="mb-3 last:mb-0 marker:text-[#b8793b]">
                  {step}
                </li>
              ))}
            </ol>
            {recipe.notes?.length ? (
              <div className="rounded-2xl border border-[#2c2620]/10 bg-[#f7efe3] p-4 text-sm leading-relaxed print:border-0 print:bg-transparent">
                <p className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">Notes</p>
                <ul className="mt-2 list-disc pl-4">
                  {recipe.notes.map((note, index) => (
                    <li key={index}>{note}</li>
                  ))}
                </ul>
              </div>
            ) : null}
            <p className="hidden text-xs text-[#4b4237] print:block">
              Source: {recipe.source.book} — Page {recipe.source.page}
            </p>
          </section>
        </div>
      )}

      {mode === "compare" && extracted && (
        <section className="paper-card flex flex-col gap-4 p-6 text-sm text-[#2c2620] print:border-0 print:bg-transparent">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">Compare AI vs Curated</p>
              <p className="text-[12px] text-[#4b4237]">
                Green = AI extra, Red = missing, Yellow = partial match. Click AI lines to edit.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                className="rounded-full border border-[#2c2620]/15 bg-white px-3 py-2 text-[11px] uppercase tracking-[0.18em] text-[#2c2620]"
                onClick={correction.addIngredient}
              >
                + Ingredient
              </button>
              <button
                className="rounded-full border border-[#2c2620]/15 bg-white px-3 py-2 text-[11px] uppercase tracking-[0.18em] text-[#2c2620]"
                onClick={correction.addInstruction}
              >
                + Instruction
              </button>
            </div>
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            <div>
              <h4 className="display-font text-xl">Curated</h4>
              <p className="text-xs text-[#6b8b6f]">Ground truth from recipe file.</p>
              <div className="mt-2 space-y-2">
                <p className="font-semibold">{recipe.title}</p>
                <p className="text-[11px] uppercase tracking-[0.18em] text-[#6b8b6f]">Ingredients</p>
                <ul className="space-y-1">
                  {recipe.ingredients.map((line, idx) => {
                    const diff = ingDiff.find((d) => d.curated === line);
                    const cls =
                      diff?.status === "missing"
                        ? "bg-[#ffecec]"
                        : diff?.status === "mismatch"
                        ? "bg-[#fff8e1]"
                        : "";
                    return (
                      <li key={idx} className={`rounded px-2 py-1 ${cls}`}>
                        {line}
                      </li>
                    );
                  })}
                </ul>
                <p className="text-[11px] uppercase tracking-[0.18em] text-[#6b8b6f]">Instructions</p>
                <ol className="space-y-1">
                  {recipe.instructions.map((line, idx) => {
                    const diff = insDiff.find((d) => d.curated === line);
                    const cls =
                      diff?.status === "missing"
                        ? "bg-[#ffecec]"
                        : diff?.status === "mismatch"
                        ? "bg-[#fff8e1]"
                        : "";
                    return (
                      <li key={idx} className={`rounded px-2 py-1 ${cls}`}>
                        {line}
                      </li>
                    );
                  })}
                </ol>
              </div>
            </div>
            <div>
              <h4 className="display-font text-xl">AI</h4>
              <p className="text-xs text-[#6b8b6f]">Editable corrections with confidence chips.</p>
              <div className="space-y-2">
                <label className="flex flex-col gap-1">
                  <span className="text-[11px] uppercase tracking-[0.18em] text-[#6b8b6f]">Title</span>
                  <input
                    className="rounded border border-[#2c2620]/20 bg-white px-3 py-2"
                    value={correction.state.title}
                    onChange={(e) => correction.updateTitle(e.target.value)}
                  />
                </label>
                <p className="text-[11px] uppercase tracking-[0.18em] text-[#6b8b6f]">Ingredients</p>
                <ul className="space-y-2">
                  {correction.state.ingredients.map((ing, idx) => {
                    const diff = ingDiff[idx];
                    const status = diff?.status;
                    const cls =
                      status === "extra"
                        ? "bg-[#e6fff4]"
                        : status === "mismatch"
                        ? "bg-[#fff8e1]"
                        : "";
                    return (
                      <li key={ing.id} className={`rounded border border-[#2c2620]/10 px-2 py-1 ${cls}`}>
                        <div className="flex items-center gap-2">
                          <input
                            className="flex-1 bg-transparent outline-none"
                            value={ing.text}
                            onChange={(e) => correction.updateIngredient(ing.id, e.target.value)}
                          />
                          {typeof ing.confidence === "number" ? (
                            <span
                              className="rounded-full bg-[#2c2620]/5 px-2 py-[2px] text-[10px]"
                              title={`Confidence ${(ing.confidence * 100).toFixed(1)}%`}
                            >
                              {confidenceLabel(ing.confidence)}
                            </span>
                          ) : null}
                          <button className="text-[11px] text-[#d34b4b]" onClick={() => correction.removeIngredient(ing.id)}>
                            Delete
                          </button>
                        </div>
                      </li>
                    );
                  })}
                </ul>
                <p className="text-[11px] uppercase tracking-[0.18em] text-[#6b8b6f]">Instructions</p>
                <ol className="space-y-2">
                  {correction.state.instructions.map((step, idx) => {
                    const diff = insDiff[idx];
                    const status = diff?.status;
                    const cls =
                      status === "extra"
                        ? "bg-[#e6fff4]"
                        : status === "mismatch"
                        ? "bg-[#fff8e1]"
                        : "";
                    return (
                      <li key={step.id} className={`rounded border border-[#2c2620]/10 px-2 py-1 ${cls}`}>
                        <div className="flex items-center gap-2">
                          <textarea
                            className="flex-1 bg-transparent outline-none"
                            value={step.text}
                            onChange={(e) => correction.updateInstruction(step.id, e.target.value)}
                          />
                          {typeof step.confidence === "number" ? (
                            <span
                              className="rounded-full bg-[#2c2620]/5 px-2 py-[2px] text-[10px]"
                              title={`Confidence ${(step.confidence * 100).toFixed(1)}%`}
                            >
                              {confidenceLabel(step.confidence)}
                            </span>
                          ) : null}
                          <button className="text-[11px] text-[#d34b4b]" onClick={() => correction.removeInstruction(step.id)}>
                            Delete
                          </button>
                        </div>
                      </li>
                    );
                  })}
                </ol>
              </div>
              <div className="mt-3 flex flex-wrap items-center gap-3">
                <CorrectionExport
                  exportJson={correction.exportJson}
                  pageNum={recipe.source.page}
                  overallConf={extracted.confidence.overall}
                />
                <button className="text-xs text-[#6b8b6f] hover:text-[#4b7050]" onClick={() => correction.reset()}>
                  Reset
                </button>
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}

function confidenceLabel(v: number): string {
  if (v >= 0.75) return `High ${(v * 100).toFixed(0)}%`;
  if (v >= 0.55) return `Med ${(v * 100).toFixed(0)}%`;
  return `Low ${(v * 100).toFixed(0)}%`;
}
