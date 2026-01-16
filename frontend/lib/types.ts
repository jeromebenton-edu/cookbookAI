export type RecipeToken = {
  id: string;
  text: string;
  label:
    | "TITLE"
    | "INGREDIENT_LINE"
    | "INSTRUCTION_STEP"
    | "TIME"
    | "TEMP"
    | "SERVINGS"
    | "NOTE"
    | "O"
    | "OTHER";
  score: number;
  bbox: [number, number, number, number];
};

export type RecipeAI = {
  pageImage: string;
  pageSize: { width: number; height: number };
  tokens: RecipeToken[];
  fieldConfidence: {
    title?: number;
    ingredients?: number;
    instructions?: number;
    servings?: number;
    time?: number;
    notes?: number;
  };
  raw: Record<string, unknown>;
};

export type ExtractedRecipe = {
  page_num: number;
  title: string;
  title_obj?: {
    id: string;
    text: string;
    confidence: number;
    bbox: [number, number, number, number];
    token_count: number;
  } | null;
  ingredients_lines?: {
    id: string;
    text: string;
    confidence: number;
    bbox: [number, number, number, number];
    token_count: number;
  }[] | null;
  instruction_lines?: {
    id: string;
    text: string;
    confidence: number;
    bbox: [number, number, number, number];
    token_count: number;
  }[] | null;
  ingredients: string[];
  instructions: string[];
  confidence: {
    title: number;
    ingredients: number;
    instructions: number;
    overall: number;
  };
  is_recipe_page?: boolean;
  recipe_confidence?: number;
  message?: string;
  meta: Record<string, unknown>;
  raw?: Record<string, unknown>;
};

export type UploadSessionSummary = {
  session_id: string;
  status: string;
  image_url?: string;
  ocr_url?: string | null;
  pred_url?: string | null;
  recipe_url?: string | null;
};

export type RecipeTime = {
  totalMinutes: number;
  activeMinutes?: number;
  label?: string;
  breakdown?: {
    prep?: string;
    cook?: string;
    total?: string;
  };
};

export type RecipeSource = {
  book: string;
  page: number;
};

export type Recipe = {
  id: string;
  book?: string;
  year?: number;
  title: string;
  description?: string;
  category: string;
  tags: string[];
  servings: string;
  time: RecipeTime;
  ingredients: string[];
  instructions: string[];
  notes?: string[];
  confidence: number;
  source: RecipeSource;
  ai: RecipeAI;
};
