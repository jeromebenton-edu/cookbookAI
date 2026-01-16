"use client";

import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  DemoFeaturedPage,
  getBostonPagePrediction,
  getDemoBundle,
  getExtractedRecipe,
  getFeaturedPages,
  getAvailableOverlayPages
} from "../../lib/api/parse";
import { ExtractedRecipe, RecipeToken } from "../../lib/types";
import DocOverlayViewer from "../../components/ai/DocOverlayViewer";
import AiLegend from "../../components/ai/AiLegend";
import AiControlsPanel from "../../components/ai/AiControlsPanel";
import JsonViewerModal from "../../components/ai/JsonViewerModal";
import { recipeToMarkdown } from "../../lib/recipeMarkdown";
import { evaluateDemoPageSelection } from "../../lib/demoSelection";

type LoadError = { message: string; variant: "offline" | "not_found" | "server" };
type FeaturedEntry = DemoFeaturedPage & { page_id: number; png_id: string; page_num?: number };

function normalizeFeatured(entry: Partial<DemoFeaturedPage>): FeaturedEntry | null {
  if (typeof entry.page_id !== "number") return null;
  return {
    page_id: entry.page_id,
    png_id: entry.png_id ?? String(entry.page_id).padStart(4, "0"),
    page_num: entry.page_num ?? entry.page_id,
    recipe_confidence: entry.recipe_confidence ?? 0,
    is_recipe_page: entry.is_recipe_page ?? true,
    label_counts: entry.label_counts,
    image_url: entry.image_url,
    title: entry.title
  };
}

function DemoPageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [featured, setFeatured] = useState<FeaturedEntry[]>([]);
  const [selectedPage, setSelectedPage] = useState<number | null>(null);
  const [defaultPageId, setDefaultPageId] = useState<number | null>(null);
  const [overlayTokens, setOverlayTokens] = useState<RecipeToken[]>([]);
  const [imageUrl, setImageUrl] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<LoadError | null>(null);
  const [recipe, setRecipe] = useState<ExtractedRecipe | null>(null);
  const [selectedLabels, setSelectedLabels] = useState<Set<string>>(new Set());
  const [confidence, setConfidence] = useState(0.5);
  const [showBoxes, setShowBoxes] = useState(true);
  const [copyStatus, setCopyStatus] = useState<"copied" | null>(null);
  const [frontMatterBanner, setFrontMatterBanner] = useState(false);
  const [outOfFeaturedBanner, setOutOfFeaturedBanner] = useState(false);
  const [degradedMessage, setDegradedMessage] = useState<string | null>(null);
  const [featuredMode, setFeaturedMode] = useState<string | null>(null);
  const [attemptedPageId, setAttemptedPageId] = useState<number | null>(null);
  const [availablePages, setAvailablePages] = useState<number[]>([]);
  const [recipeOnlyNav, setRecipeOnlyNav] = useState(true);
  const [pageInput, setPageInput] = useState<string>("");
  const canonicalizedRef = useRef(false);
  const redirectedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    async function loadDemoBundle() {
      try {
        const demo = await getDemoBundle({ timeoutMs: 12000 }).catch(() => null);
        getAvailableOverlayPages()
          .then((resp) => {
            if (!cancelled && resp?.available_page_ids?.length) {
              setAvailablePages(resp.available_page_ids.sort((a, b) => a - b));
            }
          })
          .catch(() => {});
        const needsFallback =
          !(demo?.featured_pages && demo.featured_pages.length) &&
          !((demo?.featured as any)?.pages && (demo?.featured as any)?.pages.length);
        const fallbackFeatured = needsFallback ? await getFeaturedPages().catch(() => ({ pages: [] })) : null;
        const rawPages = (demo?.featured_pages ??
          (demo?.featured as any)?.pages ??
          fallbackFeatured?.pages ??
          []) as DemoFeaturedPage[];
        let normalized = rawPages.map((p) => normalizeFeatured(p)).filter(Boolean) as FeaturedEntry[];
        normalized = normalized.sort((a, b) => {
          if (a.is_recipe_page !== b.is_recipe_page) return a.is_recipe_page ? -1 : 1;
          return (b.recipe_confidence ?? 0) - (a.recipe_confidence ?? 0);
        });
        const requested = Number(searchParams?.get("page") ?? NaN);
        const defaultPage = (demo?.default_page_id ?? demo?.default_page ?? null) as number | null;
        const selection = evaluateDemoPageSelection({
          queryPageId: requested,
          defaultPageId: defaultPage,
          featuredPages: normalized,
        });
        const initialSelection = selection.initialPageId;
        const canonicalPageId = selection.canonicalPageId;
        if (canonicalPageId && !Number.isFinite(requested) && !canonicalizedRef.current) {
          canonicalizedRef.current = true;
          router.replace(`/demo?page=${canonicalPageId}`);
        }
        if (!cancelled) {
          setFeaturedMode(demo?.featured_mode ?? (demo?.featured as any)?.source ?? null);
          if (demo?.status && demo.status !== "ok") {
            setDegradedMessage(
              typeof demo.message === "string"
                ? demo.message
                : "Featured recipe pages are unavailable. Rebuild the dataset to refresh the demo."
            );
          } else if (!normalized.length) {
            setDegradedMessage("Featured recipe pages are unavailable. Rebuild the dataset to refresh the demo.");
          } else {
            setDegradedMessage(null);
          }
          // if backend returned a default page but no featured list, inject it so nav works
          if (!normalized.length && defaultPage) {
            const injected = normalizeFeatured({
              page_id: defaultPage,
              png_id: String(defaultPage).padStart(4, "0"),
              page_num: defaultPage,
              recipe_confidence: 0,
              is_recipe_page: true
            });
            if (injected) {
              normalized = [injected];
            }
          }
          setFeatured(normalized);
          setDefaultPageId(defaultPage ?? null);
          setSelectedPage(initialSelection ?? defaultPage ?? null);
          if (initialSelection !== null) {
            setPageInput(String(initialSelection));
          }
          setAttemptedPageId(selection.attemptedPageId ?? null);
          setFrontMatterBanner(selection.isFrontMatter && Boolean(selection.redirectPageId));
          setOutOfFeaturedBanner(selection.isOutOfFeatured && Boolean(selection.redirectPageId));
          if (selection.redirectPageId && !redirectedRef.current) {
            redirectedRef.current = true;
            if (selection.redirectPageId !== null) {
              setSelectedPage(selection.redirectPageId);
              router.replace(`/demo?page=${selection.redirectPageId}`);
            }
          }
        }
      } catch (err) {
        if (!cancelled) {
          setError({ variant: "offline", message: "Unable to load demo metadata." });
        }
      }
    }
    void loadDemoBundle();
    return () => {
      cancelled = true;
    };
  }, [searchParams]);

  useEffect(() => {
    let cancelled = false;
    if (selectedPage == null) return undefined;
    setFrontMatterBanner(false);
    setOutOfFeaturedBanner(false);
    async function loadOverlay(pageId: number) {
      setLoading(true);
      setError(null);
      setRecipe(null);
      setOverlayTokens([]);
      setSelectedLabels(new Set());
      setCopyStatus(null);
      try {
        const pred = await getBostonPagePrediction(pageId);
        if (cancelled) return;
        const toks: RecipeToken[] = pred.tokens
          .filter((t) => t.pred_label !== "O")
          .map((t, idx) => ({
            id: `tok-${pageId}-${idx}`,
            text: t.text,
            label: t.pred_label as RecipeToken["label"],
            score: t.confidence,
            bbox: t.bbox as [number, number, number, number]
          }));
        setOverlayTokens(toks);
        setImageUrl(pred.image_url_resolved);
        setSelectedLabels(new Set(toks.map((t) => t.label)));
        const rec = await getExtractedRecipe(pageId).catch(() => null);
        if (!cancelled) {
          setRecipe(rec);
          const queryProvided = searchParams?.has("page") ?? false;
          if (
            queryProvided &&
            rec &&
            rec.is_recipe_page === false &&
            defaultPageId !== null &&
            defaultPageId !== pageId &&
            !redirectedRef.current
          ) {
            setFrontMatterBanner(true);
            redirectedRef.current = true;
            setSelectedPage(defaultPageId);
            router.replace(`/demo?page=${defaultPageId}`);
          }
        }
      } catch (err: any) {
        if (cancelled) return;
        const status = err?.status;
        const body = err?.body;
        if (typeof status !== "number") {
          setError({ variant: "offline", message: "AI parsing service offline." });
        } else if (status === 404) {
          if (body?.error === "overlay_not_available") {
            setError({ variant: "not_found", message: "AI overlay not available for this page." });
          } else if (body?.error === "page_not_found") {
            setError({ variant: "not_found", message: "Page not found in dataset." });
          } else {
            const message = typeof body?.message === "string" ? body.message : "Page not found.";
            setError({ variant: "not_found", message });
          }
        } else if (status >= 500) {
          const message =
            typeof body?.message === "string"
              ? body.message
              : err instanceof Error
                ? err.message
                : "Inference failed.";
          setError({ variant: "server", message });
        } else {
          const message =
            typeof body?.message === "string"
              ? body.message
              : err instanceof Error
                ? err.message
                : "Failed to load overlay.";
          setError({ variant: "server", message });
        }
        setOverlayTokens([]);
        setImageUrl("");
        setRecipe(null);
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }
    void loadOverlay(selectedPage);
    return () => {
      cancelled = true;
    };
  }, [selectedPage]);

  useEffect(() => {
    if (!copyStatus) return;
    const timeout = setTimeout(() => setCopyStatus(null), 1600);
    return () => clearTimeout(timeout);
  }, [copyStatus]);

  const filtered = useMemo(
    () => overlayTokens.filter((t) => selectedLabels.has(t.label) && t.score >= confidence),
    [overlayTokens, selectedLabels, confidence]
  );

  const legendEntries = useMemo(() => {
    const counts: Record<string, number> = {};
    filtered.forEach((t) => {
      counts[t.label] = (counts[t.label] ?? 0) + 1;
    });
    return Object.entries(counts).map(([label, count]) => ({ label, count }));
  }, [filtered]);

  const overlayLabels = useMemo(
    () => Array.from(new Set(overlayTokens.map((t) => t.label))),
    [overlayTokens]
  );

  const selectedFeatured = useMemo(
    () => featured.find((f) => f.page_id === selectedPage),
    [featured, selectedPage]
  );

  const sortedFeaturedIds = useMemo(() => featured.map((f) => f.page_id), [featured]);
  const datasetMin = useMemo(() => (availablePages.length ? Math.min(...availablePages) : 1), [availablePages]);
  const datasetMax = useMemo(
    () => (availablePages.length ? Math.max(...availablePages) : Math.max(...sortedFeaturedIds, 1)),
    [availablePages, sortedFeaturedIds]
  );

  const recipeConfidence = recipe?.recipe_confidence ?? selectedFeatured?.recipe_confidence ?? 0;
  const recipeLoaded = recipe !== null;
  const recipeDetected = recipe && recipe.is_recipe_page !== false;

  const goToPage = (pageId: number | null) => {
    if (pageId === null || Number.isNaN(pageId)) return;
    setSelectedPage(pageId);
    setPageInput(String(pageId));
    router.replace(`/demo?page=${pageId}`);
  };

  const nextPrev = (direction: 1 | -1) => {
    if (selectedPage == null) return;
    if (recipeOnlyNav && sortedFeaturedIds.length) {
      const idx = sortedFeaturedIds.indexOf(selectedPage);
      const nextIdx = idx === -1 ? 0 : idx + direction;
      const bounded = ((nextIdx % sortedFeaturedIds.length) + sortedFeaturedIds.length) % sortedFeaturedIds.length;
      goToPage(sortedFeaturedIds[bounded]);
      return;
    }
    const pool = availablePages.length ? availablePages : null;
    if (pool) {
      const idx = pool.findIndex((p) => p === selectedPage);
      const nextIdx = idx === -1 ? 0 : Math.min(Math.max(idx + direction, 0), pool.length - 1);
      goToPage(pool[nextIdx]);
    } else {
      goToPage(Math.min(Math.max(selectedPage + direction, datasetMin), datasetMax));
    }
  };

  const randomFeatured = () => {
    if (!featured.length) return;
    const choices = featured.filter((f) => f.page_id !== selectedPage);
    const pool = choices.length ? choices : featured;
    const pick = pool[Math.floor(Math.random() * pool.length)];
    goToPage(pick.page_id);
  };

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-10">
      <div className="flex flex-col gap-2">
        <p className="text-xs uppercase tracking-[0.28em] text-[#6b8b6f]">AI Parse Demo</p>
        <h1 className="display-font text-4xl font-semibold">See the scan and the structured recipe side-by-side.</h1>
        <p className="max-w-3xl text-sm leading-relaxed text-[#4b4237]">
          Turn on AI Parse View to see how LayoutLMv3 labels tokens and groups them into recipe fields. Featured pages
          start you on recipe-heavy sections instead of the front matter.
        </p>
      </div>
      {degradedMessage ? (
        <div className="rounded-xl border border-[#b8793b]/30 bg-[#fff8ed] px-4 py-3 text-sm text-[#5b3d22]">
          {degradedMessage}
        </div>
      ) : null}
      {degradedMessage ? (
        <div className="rounded-xl border border-[#b8793b]/30 bg-[#fff8ed] px-4 py-3 text-sm text-[#5b3d22]">
          {degradedMessage}
        </div>
      ) : null}

      <section className="paper-card flex flex-col gap-3 p-5">
        <div className="flex items-center justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">How it works</p>
            <p className="text-sm text-[#4b4237]">
              Choose a featured scan, toggle the overlays, and copy the markdown or JSON export for your workflow.
            </p>
          </div>
          <span className="rounded-full bg-[#2c2620]/5 px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-[#2c2620]">
            LayoutLMv3 Parse View
          </span>
        </div>
        <ol className="grid gap-2 text-sm leading-relaxed text-[#2c2620] md:grid-cols-3">
          <li>Featured AI Pages are scored by recipe confidence (title + ingredients + instructions).</li>
          <li>Toggle labels and confidence thresholds to inspect how the model tags each token.</li>
          <li>Copy the formatted recipe or raw JSON once the extraction looks right.</li>
        </ol>
      </section>

      <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="flex flex-col gap-4">
          <div className="glass-panel flex flex-col gap-3 p-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">
                  {featuredMode === "fallback_any" ? "Available Pages" : "Featured AI Pages"}
                </p>
                <p className="text-sm text-[#4b4237]">
                  {featuredMode === "fallback_any"
                    ? "No detected recipe pages; showing available pages so you can still browse the overlay."
                    : "Recipe-heavy pages ranked by recipe confidence so the demo opens on a real recipe."}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  className="rounded-full border border-[#2c2620]/15 bg-white px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-[#2c2620] shadow-sm transition hover:-translate-y-[1px] hover:shadow-md disabled:opacity-60"
                  onClick={randomFeatured}
                  disabled={!featured.length}
                >
                  Try another recipe
                </button>
                {selectedFeatured ? (
                  <span className="rounded-full bg-[#2c2620]/5 px-3 py-1 text-[11px] uppercase tracking-[0.18em] text-[#2c2620]">
                    {Math.round((selectedFeatured.recipe_confidence ?? 0) * 100)}%
                  </span>
                ) : null}
              </div>
            </div>
            <div className="flex gap-2 overflow-x-auto pb-1">
              {featured.length ? (
                featured.map((f) => (
                  <button
                    key={f.page_id}
                    onClick={() => goToPage(f.page_id)}
                    className={`min-w-[140px] rounded-xl border px-3 py-2 text-left text-xs transition ${
                      selectedPage === f.page_id ? "border-[#2c2620] bg-[#f7efe3]" : "border-[#2c2620]/10 bg-white"
                    }`}
                  >
                    <p className="text-[10px] uppercase tracking-[0.2em] text-[#6b8b6f]">
                      Page {f.page_num ?? f.page_id}
                    </p>
                    <p className="font-semibold text-[#2c2620]">{f.title || "Recipe page"}</p>
                    <p className="text-[10px] text-[#4b4237]">
                      {f.is_recipe_page === false
                        ? "Not a recipe"
                        : `Confidence ${(f.recipe_confidence * 100).toFixed(0)}%`}
                    </p>
                  </button>
                ))
              ) : (
                <p className="text-sm text-[#4b4237]">
                  {degradedMessage
                    ? "Featured recipe pages are unavailable. Rebuild the dataset to refresh the demo."
                    : "Featured pages will appear once the backend warms up."}
                </p>
              )}
            </div>
          </div>

          <div className="glass-panel flex flex-wrap items-center gap-3 p-4">
            <div className="flex items-center gap-2">
              <button
                className="rounded-full border border-[#2c2620]/15 bg-white px-3 py-1 text-xs uppercase tracking-[0.18em] text-[#2c2620] shadow-sm transition hover:-translate-y-[1px] hover:shadow-md disabled:opacity-60"
                onClick={() => nextPrev(-1)}
                disabled={selectedPage == null}
              >
                Prev
              </button>
              <button
                className="rounded-full border border-[#2c2620]/15 bg-white px-3 py-1 text-xs uppercase tracking-[0.18em] text-[#2c2620] shadow-sm transition hover:-translate-y-[1px] hover:shadow-md disabled:opacity-60"
                onClick={() => nextPrev(1)}
                disabled={selectedPage == null}
              >
                Next
              </button>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-[11px] uppercase tracking-[0.18em] text-[#6b8b6f]">Go to page</label>
              <input
                type="number"
                value={pageInput}
                min={datasetMin}
                max={datasetMax}
                onChange={(e) => setPageInput(e.target.value)}
                onBlur={() => {
                  const num = Number(pageInput);
                  if (Number.isFinite(num)) goToPage(num);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    const num = Number(pageInput);
                    if (Number.isFinite(num)) goToPage(num);
                  }
                }}
                className="w-20 rounded-lg border border-[#2c2620]/20 px-2 py-1 text-sm"
              />
            </div>
            <div className="flex items-center gap-2">
              <label className="text-[11px] uppercase tracking-[0.18em] text-[#6b8b6f]">Recipe pages only</label>
              <input
                type="checkbox"
                checked={recipeOnlyNav}
                onChange={(e) => setRecipeOnlyNav(e.target.checked)}
                className="h-4 w-4 accent-[#2c2620]"
                disabled={featuredMode === "fallback_any" || !featured.length}
                title={
                  featuredMode === "fallback_any"
                    ? "No recipe pages detected; showing all available pages."
                    : undefined
                }
              />
            </div>
          </div>

          <AiControlsPanel
            labels={overlayLabels}
            selected={selectedLabels}
            onToggleLabel={(lbl) =>
              setSelectedLabels((prev) => {
                const next = new Set(prev);
                if (next.has(lbl)) next.delete(lbl);
                else next.add(lbl);
                return next;
              })
            }
            onSelectAll={() => setSelectedLabels(new Set(overlayLabels))}
            onClear={() => setSelectedLabels(new Set())}
            showBoxes={showBoxes}
            onToggleBoxes={setShowBoxes}
            confidence={confidence}
            onConfidenceChange={setConfidence}
            disabled={overlayTokens.length === 0}
          />

          <div className="paper-card flex flex-col gap-4 p-4">
            {frontMatterBanner ? (
              <div className="rounded-xl border border-[#b8793b]/30 bg-[#fff8ed] px-4 py-3 text-xs text-[#5b3d22]">
                This page is front matter. Jumping to a featured recipe page.
              </div>
            ) : null}
            {outOfFeaturedBanner ? (
              <div className="rounded-xl border border-[#b8793b]/30 bg-[#fff8ed] px-4 py-3 text-xs text-[#5b3d22]">
                {attemptedPageId
                  ? `Page ${attemptedPageId} isn’t part of the featured demo set. Jumping to a recipe page.`
                  : "That page isn’t part of the featured demo set. Jumping to a recipe page."}
              </div>
            ) : null}
            {error ? (
              <div
                className={`rounded-xl border px-4 py-3 text-sm ${
                  error.variant === "offline"
                    ? "border-[#d34b4b]/30 bg-[#fff5f5] text-[#7a1d1d]"
                    : "border-[#b8793b]/30 bg-[#fff8ed] text-[#5b3d22]"
                }`}
              >
                {error.message}
              </div>
            ) : null}
            {loading || !imageUrl ? (
              <div className="flex h-[480px] items-center justify-center rounded-xl border border-dashed border-[#2c2620]/20 bg-white/60 text-sm text-[#4b4237]">
                <div className="h-[440px] w-full animate-pulse rounded-xl bg-gradient-to-br from-[#f7efe3] to-white/70" />
              </div>
            ) : (
              <DocOverlayViewer
                imageUrl={imageUrl}
                imageSize={undefined}
                tokens={filtered}
                visibleLabels={selectedLabels}
                showBoxes={showBoxes}
                confidenceThreshold={confidence}
              />
            )}
            <AiLegend entries={legendEntries} />
          </div>
        </div>

        <aside className="flex flex-col gap-4">
          <div className="paper-card flex flex-col gap-3 p-5">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.22em] text-[#6b8b6f]">Extracted Recipe</p>
                <h3 className="display-font text-2xl font-semibold text-[#2c2620]">
                  {recipeDetected
                    ? recipe?.title || "AI recipe preview"
                    : recipeLoaded
                      ? "No recipe detected"
                      : "Loading recipe preview..."}
                </h3>
                <p className="text-xs text-[#4b4237]">
                  {recipeDetected
                    ? "Ingredients and instructions grouped from the overlay tokens."
                    : recipeLoaded
                      ? "This looks like front matter or a non-recipe page."
                      : "Fetching the recipe extraction for this page."}
                </p>
              </div>
              <div className="flex flex-col items-end gap-1 text-[11px] uppercase tracking-[0.18em] text-[#2c2620]">
                <span className="rounded-full bg-[#2c2620]/5 px-3 py-1">
                  Recipe {Math.round((recipeConfidence ?? 0) * 100)}%
                </span>
                {typeof recipe?.confidence?.overall === "number" ? (
                  <span className="rounded-full bg-[#2c2620]/5 px-3 py-1">
                    Overall {(recipe.confidence.overall * 100).toFixed(0)}%
                  </span>
                ) : null}
              </div>
            </div>

            {recipeDetected ? (
              <div className="grid gap-3 text-sm leading-relaxed text-[#2c2620]">
                <div>
                  <p className="text-[11px] uppercase tracking-[0.2em] text-[#6b8b6f]">Ingredients</p>
                  <ul className="mt-1 list-disc pl-5">
                    {recipe?.ingredients?.length
                      ? recipe.ingredients.map((ing, idx) => <li key={idx}>{ing}</li>)
                      : "Waiting for AI parse..."}
                  </ul>
                </div>
                <div>
                  <p className="text-[11px] uppercase tracking-[0.2em] text-[#6b8b6f]">Instructions</p>
                  <ol className="mt-1 list-decimal pl-5">
                    {recipe?.instructions?.length
                      ? recipe.instructions.map((step, idx) => <li key={idx}>{step}</li>)
                      : "Waiting for AI parse..."}
                  </ol>
                </div>
              </div>
            ) : (
              <>
                {recipeLoaded ? (
                  <div className="rounded-xl border border-[#2c2620]/10 bg-[#fff8ed] px-4 py-3 text-sm text-[#5b3d22]">
                    {recipe?.message ?? "No recipe detected on this page."}
                    <p className="text-[11px] text-[#4b4237]">
                      Use the overlay to inspect how the model labeled tokens.
                    </p>
                  </div>
                ) : (
                  <div className="rounded-xl border border-[#2c2620]/10 bg-white px-4 py-3 text-sm text-[#4b4237]">
                    Loading recipe extraction...
                  </div>
                )}
              </>
            )}

            <div className="flex flex-wrap items-center gap-3">
              <button
                className="rounded-full border border-[#2c2620]/15 bg-white px-4 py-2 text-xs uppercase tracking-[0.2em] text-[#2c2620] shadow-sm transition hover:-translate-y-[1px] hover:shadow-md disabled:opacity-60"
                onClick={() => {
                  if (!recipe) return;
                  void navigator.clipboard.writeText(recipeToMarkdown(recipe));
                  setCopyStatus("copied");
                }}
                disabled={!recipe || recipe.is_recipe_page === false}
              >
                Copy recipe
              </button>
              <JsonViewerModal data={recipe ?? {}} />
              {copyStatus ? <span className="text-xs text-[#6b8b6f]">Recipe copied to clipboard</span> : null}
            </div>
          </div>

          <div className="rounded-2xl border border-[#2c2620]/10 bg-white/80 p-4 text-sm text-[#4b4237]">
            <p className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">Upload your own page</p>
            <p className="mt-1">
              Run OCR + LayoutLMv3 on your own scan, then compare overlays and exports to the demo pages.
            </p>
            <a
              href="/upload"
              className="mt-3 inline-flex rounded-full bg-[#2c2620] px-4 py-2 text-xs uppercase tracking-[0.2em] text-white shadow transition hover:-translate-y-[1px] hover:shadow-md"
            >
              Upload a page
            </a>
          </div>
        </aside>
      </div>
    </div>
  );
}

export default function DemoPage() {
  return (
    <Suspense
      fallback={
        <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-10">
          <p className="text-sm text-[#4b4237]">Loading demo...</p>
        </div>
      }
    >
      <DemoPageInner />
    </Suspense>
  );
}
