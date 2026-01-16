export type DemoFeatured = { page_id: number; is_recipe_page?: boolean | null };

type SelectionResult = {
  initialPageId: number | null;
  canonicalPageId: number | null;
  redirectPageId: number | null;
  isFrontMatter: boolean;
  isOutOfFeatured: boolean;
  attemptedPageId: number | null;
};

/**
 * Centralized demo page selection logic.
 * Do not hardcode page ids; default comes from backend demo endpoint.
 */
export function evaluateDemoPageSelection(opts: {
  queryPageId?: number | null;
  defaultPageId?: number | null;
  featuredPages?: DemoFeatured[];
}): SelectionResult {
  const queryPageId = Number.isFinite(opts.queryPageId ?? NaN) ? Number(opts.queryPageId) : null;
  const defaultPageId = Number.isFinite(opts.defaultPageId ?? NaN) ? Number(opts.defaultPageId) : null;
  const featured = opts.featuredPages ?? [];

  const findEntry = (pid: number | null) => featured.find((p) => p.page_id === pid);
  const pickFallback = () => defaultPageId ?? (featured[0]?.page_id ?? null);

  if (queryPageId !== null) {
    const entry = findEntry(queryPageId);
    const isFrontMatter = entry?.is_recipe_page === false;
    const inFeatured = Boolean(entry);
    const fallback = pickFallback();
    const shouldRedirect = isFrontMatter || !inFeatured;
    return {
      initialPageId: shouldRedirect ? fallback : queryPageId,
      canonicalPageId: null,
      redirectPageId: shouldRedirect ? fallback : null,
      isFrontMatter,
      isOutOfFeatured: !inFeatured,
      attemptedPageId: queryPageId,
    };
  }

  const initial = pickFallback();
  return {
    initialPageId: initial,
    canonicalPageId: initial,
    redirectPageId: null,
    isFrontMatter: false,
    isOutOfFeatured: false,
    attemptedPageId: null,
  };
}
