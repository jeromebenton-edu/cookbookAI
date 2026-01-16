"use client";

import { useEffect, useMemo, useState } from "react";

const STORAGE_PREFIX = "ingredients_checked_";

type IngredientChecklistProps = {
  recipeId: string;
  ingredients: string[];
};

type ChecklistState = boolean[];

export default function IngredientChecklist({
  recipeId,
  ingredients
}: IngredientChecklistProps) {
  const [checked, setChecked] = useState<ChecklistState>([]);

  const storageKey = useMemo(() => `${STORAGE_PREFIX}${recipeId}`, [recipeId]);

  useEffect(() => {
    const saved = typeof window !== "undefined" ? localStorage.getItem(storageKey) : null;
    if (saved) {
      try {
        const parsed = JSON.parse(saved) as ChecklistState;
        if (Array.isArray(parsed)) {
          setChecked(padState(parsed, ingredients.length));
          return;
        }
      } catch (error) {
        // ignore parse errors
      }
    }
    setChecked(Array(ingredients.length).fill(false));
  }, [storageKey, ingredients.length]);

  useEffect(() => {
    if (checked.length === ingredients.length && typeof window !== "undefined") {
      localStorage.setItem(storageKey, JSON.stringify(checked));
    }
  }, [checked, ingredients.length, storageKey]);

  const checkedCount = checked.filter(Boolean).length;

  const toggleIndex = (index: number) => {
    setChecked((prev) => {
      const next = [...padState(prev, ingredients.length)];
      next[index] = !next[index];
      return next;
    });
  };

  const setAll = (value: boolean) => {
    setChecked(Array(ingredients.length).fill(value));
  };

  const reset = () => {
    setChecked(Array(ingredients.length).fill(false));
    if (typeof window !== "undefined") {
      localStorage.removeItem(storageKey);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-3 text-sm text-[#4b4237]">
        <p>
          <span className="font-semibold text-[#2c2620]">{checkedCount}</span> / {ingredients.length} ingredients checked
        </p>
        <div className="flex flex-wrap gap-2 text-xs">
          <button
            type="button"
            onClick={() => setAll(true)}
            className="rounded-full border border-[#2c2620]/15 bg-white px-3 py-1 text-[#2c2620] shadow-sm transition hover:-translate-y-[1px] hover:shadow-md print:hidden"
          >
            Check all
          </button>
          <button
            type="button"
            onClick={() => setAll(false)}
            className="rounded-full border border-[#2c2620]/15 bg-white px-3 py-1 text-[#2c2620] shadow-sm transition hover:-translate-y-[1px] hover:shadow-md print:hidden"
          >
            Uncheck all
          </button>
          <button
            type="button"
            onClick={reset}
            className="rounded-full border border-[#b8793b]/20 bg-[#fff5ea] px-3 py-1 text-[#b8793b] shadow-sm transition hover:-translate-y-[1px] hover:shadow-md print:hidden"
          >
            Reset checklist
          </button>
        </div>
      </div>
      <ul className="flex flex-col gap-3 text-base leading-relaxed">
        {ingredients.map((ingredient, index) => {
          const id = `${recipeId}-ingredient-${index}`;
          return (
            <li key={id} className="flex items-start gap-3">
              <input
                id={id}
                type="checkbox"
                className="checklist-checkbox mt-[6px] h-4 w-4 rounded border-[#2c2620]/30 text-[#b8793b] print:h-4 print:w-4"
                checked={checked[index] ?? false}
                onChange={() => toggleIndex(index)}
              />
              <label htmlFor={id} className="checklist-label cursor-pointer text-[#2c2620]">
                {ingredient}
              </label>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function padState(state: ChecklistState, length: number): ChecklistState {
  if (state.length === length) return state;
  const next = Array(length).fill(false) as boolean[];
  for (let i = 0; i < length && i < state.length; i += 1) {
    next[i] = state[i];
  }
  return next;
}
