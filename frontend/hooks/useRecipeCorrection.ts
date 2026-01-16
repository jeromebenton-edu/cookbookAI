import { useState } from "react";
import { ExtractedRecipe } from "../lib/types";

export type CorrectedRecipeState = {
  title: string;
  ingredients: { id: string; text: string; confidence?: number }[];
  instructions: { id: string; text: string; confidence?: number }[];
};

export function useRecipeCorrection(ai: ExtractedRecipe | null) {
  const [state, setState] = useState<CorrectedRecipeState>(() => initState(ai));

  const reset = () => setState(initState(ai));

  const updateTitle = (text: string) => setState((s) => ({ ...s, title: text }));
  const updateIngredient = (id: string, text: string) =>
    setState((s) => ({
      ...s,
      ingredients: s.ingredients.map((i) => (i.id === id ? { ...i, text } : i))
    }));
  const updateInstruction = (id: string, text: string) =>
    setState((s) => ({
      ...s,
      instructions: s.instructions.map((i) => (i.id === id ? { ...i, text } : i))
    }));
  const addIngredient = () =>
    setState((s) => ({
      ...s,
      ingredients: [...s.ingredients, { id: `ing_new_${s.ingredients.length + 1}`, text: "", confidence: 0 }]
    }));
  const addInstruction = () =>
    setState((s) => ({
      ...s,
      instructions: [...s.instructions, { id: `ins_new_${s.instructions.length + 1}`, text: "", confidence: 0 }]
    }));
  const removeIngredient = (id: string) =>
    setState((s) => ({ ...s, ingredients: s.ingredients.filter((i) => i.id !== id) }));
  const removeInstruction = (id: string) =>
    setState((s) => ({ ...s, instructions: s.instructions.filter((i) => i.id !== id) }));

  const exportJson = (meta?: Record<string, unknown>) => ({
    page_num: ai?.page_num,
    corrected: {
      title: state.title,
      ingredients: state.ingredients.map((i) => i.text).filter(Boolean),
      instructions: state.instructions.map((i) => i.text).filter(Boolean)
    },
    ai_original: ai,
    meta: {
      ...meta,
      created_at: new Date().toISOString()
    }
  });

  return {
    state,
    updateTitle,
    updateIngredient,
    updateInstruction,
    addIngredient,
    addInstruction,
    removeIngredient,
    removeInstruction,
    reset,
    exportJson
  };
}

function initState(ai: ExtractedRecipe | null): CorrectedRecipeState {
  return {
    title: ai?.title || "",
    ingredients:
      ai?.ingredients_lines?.map((l) => ({ id: l.id, text: l.text, confidence: l.confidence })) ||
      ai?.ingredients?.map((txt, idx) => ({ id: `ing_${idx}`, text: txt })) ||
      [],
    instructions:
      ai?.instruction_lines?.map((l) => ({ id: l.id, text: l.text, confidence: l.confidence })) ||
      ai?.instructions?.map((txt, idx) => ({ id: `ins_${idx}`, text: txt })) ||
      []
  };
}
