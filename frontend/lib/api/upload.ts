const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "";

export type UploadSessionSummary = {
  session_id: string;
  status: string;
  image_url?: string;
  ocr_url?: string | null;
  pred_url?: string | null;
  recipe_url?: string | null;
};

async function fetchJson<T>(url: string, opts: RequestInit = {}): Promise<T> {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed (${res.status})`);
  }
  return (await res.json()) as T;
}

export async function uploadPage(file: File): Promise<UploadSessionSummary> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/api/upload/page`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Upload failed (${res.status})`);
  }
  return (await res.json()) as UploadSessionSummary;
}

export function getUploadImageUrl(sessionId: string) {
  return `${API_BASE}/api/upload/${sessionId}/image`;
}

export async function getUploadPred(sessionId: string) {
  return fetchJson(`${API_BASE}/api/upload/${sessionId}/pred`);
}

export async function getUploadRecipe(sessionId: string) {
  return fetchJson(`${API_BASE}/api/upload/${sessionId}/recipe`);
}

export async function getUploadSession(sessionId: string) {
  return fetchJson(`${API_BASE}/api/upload/${sessionId}`);
}
