"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { Recipe } from "../lib/types";
import RecipeDetailAiView from "./RecipeDetailAiView";
import RecipeDetailCookView from "./RecipeDetailCookView";
import { cn } from "../lib/utils";
import RecipeMeta from "./RecipeMeta";
import PrintButton from "./PrintButton";
import { useOverlayAvailability } from "../hooks/useOverlayAvailability";

export default function RecipeDetail({ recipe }: { recipe: Recipe }) {
  const [view, setView] = useState<"cook" | "ai">("cook");
  const [cookMode, setCookMode] = useState(false);
  const [wakeLockActive, setWakeLockActive] = useState(false);
  const router = useRouter();
  const pageNum = recipe.source.page;
  const { available, loading, availablePageIds } = useOverlayAvailability(pageNum);

  // If overlay is not available, keep the UI in Cook mode
  useEffect(() => {
    if (!loading && !available && view === "ai") {
      setView("cook");
    }
  }, [available, loading, view]);

  // Wake lock to prevent screen from sleeping in cook mode
  useEffect(() => {
    if (!cookMode) {
      setWakeLockActive(false);
      return;
    }
    let wakeLock: WakeLockSentinel | null = null;
    const requestWakeLock = async () => {
      try {
        if ("wakeLock" in navigator) {
          wakeLock = await navigator.wakeLock.request("screen");
          setWakeLockActive(true);
          wakeLock.addEventListener("release", () => setWakeLockActive(false));
        }
      } catch {
        setWakeLockActive(false);
      }
    };
    requestWakeLock();
    // Re-acquire on visibility change (e.g., tab switch back)
    const handleVisibility = () => {
      if (document.visibilityState === "visible" && cookMode) {
        requestWakeLock();
      }
    };
    document.addEventListener("visibilitychange", handleVisibility);
    return () => {
      document.removeEventListener("visibilitychange", handleVisibility);
      if (wakeLock) wakeLock.release();
    };
  }, [cookMode]);

  const demoOptions = useMemo(() => availablePageIds.slice(0, 50), [availablePageIds]);

  return (
    <section className={cn("flex flex-col gap-8", cookMode && "cook-mode")}>
      <div className="flex flex-col gap-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-[0.25em] text-[#6b8b6f]">
              {recipe.source.book} - Page {recipe.source.page}
            </p>
            <h1 className="display-font text-4xl font-semibold text-balance">
              {recipe.title}
            </h1>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <PrintButton />
            <div className="print-hidden flex items-center rounded-full border border-[#2c2620]/20 bg-white/70 p-1 text-xs uppercase tracking-[0.22em]">
              <button
                className={cn(
                  "rounded-full px-4 py-2 transition",
                  view === "cook"
                    ? "bg-[#2c2620] text-[#f7efe3]"
                    : "text-[#4b4237]"
                )}
                onClick={() => setView("cook")}
              >
                Cook View
              </button>
              <button
                className={cn(
                  "rounded-full px-4 py-2 transition",
                  view === "ai"
                    ? "bg-[#2c2620] text-[#f7efe3]"
                    : "text-[#4b4237]",
                  (!available && !loading) && "cursor-not-allowed opacity-50"
                )}
                onClick={() => (available ? setView("ai") : null)}
                disabled={!available && !loading}
              >
                AI Parse View
              </button>
            </div>
          </div>
        </div>
        {recipe.description ? (
          <p className="max-w-2xl text-base leading-relaxed text-[#4b4237]">
            {recipe.description}
          </p>
        ) : null}
        <div className="flex flex-wrap items-center justify-between gap-3 print:hidden">
          <RecipeMeta recipe={recipe} className="w-full md:w-auto" />
          <button
            type="button"
            onClick={() => setCookMode((prev) => !prev)}
            className={cn(
              "rounded-full border border-[#2c2620]/15 bg-white px-4 py-2 text-xs uppercase tracking-[0.2em] text-[#2c2620] shadow-sm transition hover:-translate-y-[1px] hover:shadow-md",
              cookMode && "bg-[#2c2620] text-[#f7efe3]"
            )}
            title={cookMode && wakeLockActive ? "Screen will stay on" : ""}
          >
            {cookMode ? (
              <>Exit Cook Mode {wakeLockActive && <span className="ml-1 opacity-70">☀</span>}</>
            ) : (
              "Cook Mode"
            )}
          </button>
        </div>
        {!loading && !available ? (
          <div className="print-hidden rounded-2xl border border-[#e5c08f]/40 bg-[#fffaf2] px-4 py-3 text-sm text-[#5c4120]">
            <p className="font-semibold">AI overlay not available for this page.</p>
            <p className="text-xs text-[#6b4a2c]">
              The demo subset includes {availablePageIds.length || 0} featured pages. Choose a demo page to see AI
              Parse View, or visit the Demo gallery.
            </p>
            <div className="mt-3 flex flex-wrap items-center gap-3">
              <label className="text-xs uppercase tracking-[0.2em] text-[#8b6d46]">
                Featured AI demo pages
              </label>
              <select
                className="rounded-full border border-[#2c2620]/15 bg-white px-3 py-2 text-xs uppercase tracking-[0.18em] text-[#2c2620] shadow-sm"
                defaultValue=""
                onChange={(e) => {
                  const v = e.target.value;
                  if (v) router.push(`/demo?page=${v}`);
                }}
              >
                <option value="">Select a page</option>
                {demoOptions.map((p) => (
                  <option value={p} key={p}>
                    Page {p.toString().padStart(4, "0")}
                  </option>
                ))}
              </select>
              <a
                href="/demo"
                className="inline-flex items-center rounded-full border border-[#2c2620]/15 bg-white px-3 py-2 text-xs uppercase tracking-[0.2em] text-[#2c2620] shadow-sm transition hover:-translate-y-[1px] hover:shadow-md"
              >
                Go to Demo
              </a>
            </div>
          </div>
        ) : null}
      </div>

      {view === "cook" ? (
        <RecipeDetailCookView recipe={recipe} />
      ) : (
        <div className="print-hidden">
          <RecipeDetailAiView recipe={recipe} enabled={available} />
        </div>
      )}

      {view === "cook" ? (
        <div className="print-hidden fixed bottom-4 left-0 right-0 z-10 px-4 md:hidden">
          <div className="mx-auto flex max-w-xl items-center justify-center gap-3 rounded-full bg-white/95 px-4 py-3 shadow-[0_12px_30px_rgba(31,27,22,0.2)] backdrop-blur">
            <a
              href="#ingredients"
              className="flex-1 text-center text-xs uppercase tracking-[0.22em] text-[#2c2620]"
            >
              Jump to Ingredients
            </a>
            <span className="h-4 w-px bg-[#2c2620]/10" aria-hidden />
            <a
              href="#steps"
              className="flex-1 text-center text-xs uppercase tracking-[0.22em] text-[#2c2620]"
            >
              Jump to Steps
            </a>
          </div>
        </div>
      ) : null}
    </section>
  );
}
