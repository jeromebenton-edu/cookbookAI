export const LABEL_COLORS: Record<string, string> = {
  TITLE: "#7a5ea6",
  INGREDIENT_LINE: "#5b9c75",
  INSTRUCTION_STEP: "#2b67b2",
  TIME: "#f0a037",
  TEMP: "#d34b4b",
  SERVINGS: "#34a1a3",
  NOTE: "#9c7a5b",
  O: "#d9d9d9",
  OTHER: "#7c6f64"
};

export function getLabelColor(label: string): string {
  return LABEL_COLORS[label] ?? "#7c6f64";
}
