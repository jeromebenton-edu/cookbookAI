import type { ExtractedRecipe } from "../types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ||
  "http://localhost:8000";

type FetchOpts = RequestInit & { timeoutMs?: number; retry?: number };

async function fetchJson<T>(url: string, opts: FetchOpts = {}): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), opts.timeoutMs ?? 20000);
  const request: RequestInit = {
    ...opts,
    signal: controller.signal
  };
  try {
    const res = await fetch(url, request);
    if (!res.ok) {
      throw new Error(`Request failed (${res.status})`);
    }
    return (await res.json()) as T;
  } catch (err) {
    if ((opts.retry ?? 0) > 0) {
      return fetchJson(url, { ...opts, retry: (opts.retry ?? 0) - 1 });
    }
    throw err;
  } finally {
    clearTimeout(timeout);
  }
}

export type ParseToken = {
  text: string;
  bbox: [number, number, number, number];
  pred_label: string;
  pred_id: number;
  confidence: number;
};

export type ParseResponse = {
  page_num: number;
  page_id?: number;
  image_path: string;
  image_url?: string;
  is_recipe_page?: boolean;
  recipe_confidence?: number;
  label_counts?: Record<string, number>;
  recipe_token_ratio?: number;
  avg_token_confidence?: number;
  title?: string;
  tokens: ParseToken[];
  grouped?: Record<string, ParseToken[]>;
  label_map: { id2label: Record<string, string>; label2id: Record<string, number> };
  meta: Record<string, unknown>;
};

export type DemoFeaturedPage = {
  page_id: number;
  png_id: string;
  page_num?: number;
  recipe_confidence: number;
  is_recipe_page: boolean;
  label_counts?: Record<string, number>;
  image_url?: string;
  title?: string;
};

export type DemoBundle = {
  featured?: { count?: number; pages?: DemoFeaturedPage[] };
  featured_pages?: DemoFeaturedPage[];
  health: unknown;
  default_page?: number | null;
  default_page_id?: number | null;
  status?: string | null;
  message?: string | null;
  pages_total?: number;
  pages_with_images?: number;
  featured_mode?: "curated_recipe" | "heuristic_recipe" | "fallback_any" | string;
};

export async function getParseHealth() {
  return fetchJson<Record<string, unknown>>(`${API_BASE}/api/parse/health`, { retry: 1 });
}

export async function getBostonPages() {
  return fetchJson<{ pages: number[]; count: number }>(`${API_BASE}/api/parse/boston/pages`, {
    retry: 1
  });
}

export async function getBostonPagePrediction(
  pageNum: number,
  opts?: { refresh?: boolean; minConf?: number; grouped?: boolean }
): Promise<ParseResponse & { image_url_resolved: string }> {
  const params = new URLSearchParams();
  if (opts?.refresh) params.set("refresh", "true");
  if (opts?.grouped === false) params.set("grouped", "false");
  if (typeof opts?.minConf === "number") params.set("min_conf", String(opts.minConf));
  const url = `${API_BASE}/api/parse/boston/${pageNum}${params.toString() ? `?${params.toString()}` : ""}`;
  const resRaw = await fetch(url, { method: "GET" });
  let body: any = undefined;
  if (!resRaw.ok) {
    try {
      body = await resRaw.json();
    } catch {
      body = await resRaw.text();
    }
    const err: any = new Error(
      typeof body?.message === "string" ? body.message : `Request failed (${resRaw.status})`
    );
    err.status = resRaw.status;
    err.body = body;
    throw err;
  }
  const res = (await resRaw.json()) as ParseResponse;
  const imageUrl =
    res.image_url && res.image_url.startsWith("http")
      ? res.image_url
      : `${API_BASE}${res.image_url ?? `/api/parse/boston/${pageNum}/image`}`;
  return { ...res, image_url_resolved: imageUrl };
}

export type AvailablePagesResponse = {
  num_pages: number;
  available_page_ids: number[];
  available_png_ids: string[];
};

export async function getAvailableOverlayPages() {
  return fetchJson<AvailablePagesResponse>(`${API_BASE}/api/parse/boston/available`, { retry: 1 });
}

export async function getFeaturedPages(limit = 10, refresh = false) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (refresh) params.set("refresh", "true");
  return fetchJson<{ count: number; pages: DemoFeaturedPage[] }>(`${API_BASE}/api/parse/boston/featured?${params.toString()}`, {
    retry: 1
  });
}

export async function getExtractedRecipe(
  pageNum: number,
  opts?: { refresh?: boolean; includeRaw?: boolean; includeLines?: boolean }
) {
  const params = new URLSearchParams();
  if (opts?.refresh) params.set("refresh", "true");
  if (opts?.includeRaw) params.set("include_raw", "true");
  if (opts?.includeLines === false) params.set("include_lines", "false");
  const url = `${API_BASE}/api/parse/boston/${pageNum}/recipe${params.toString() ? `?${params.toString()}` : ""}`;
  return fetchJson<ExtractedRecipe>(url, { retry: 1 });
}

export async function getDemoBundle(opts?: { timeoutMs?: number; retry?: number }) {
  return fetchJson<DemoBundle>(`${API_BASE}/api/parse/boston/demo`, {
    retry: 1,
    ...(opts ?? {})
  });
}

export { API_BASE };
