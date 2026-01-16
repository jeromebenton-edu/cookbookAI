"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { RecipeToken } from "../../lib/types";
import { SectionBox, buildSectionOverlays } from "../../lib/overlays/sectionOverlays";
import { getLabelColor } from "./constants";

export type SectionOverlayViewerProps = {
  imageUrl: string;
  imageSize?: { width: number; height: number };
  tokens: RecipeToken[];
  selectedTitle: string;
  showBoxes: boolean;
};

type TooltipState = {
  section: SectionBox;
  x: number;
  y: number;
};

export default function SectionOverlayViewer({
  imageUrl,
  imageSize,
  tokens,
  selectedTitle,
  showBoxes,
}: SectionOverlayViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const [renderSize, setRenderSize] = useState<{ width: number; height: number } | null>(null);

  // Build section-level overlays
  const sectionOverlays = useMemo(() => {
    const result = buildSectionOverlays(tokens, selectedTitle, {
      padding: 8,
      twoColumnThreshold: 250,
    });

    // DEV: Log overlay computation
    console.log("[SectionOverlayViewer] Built overlays:", {
      selectedTitle,
      tokenCount: tokens.length,
      titleBox: result.titleBox ? "✓" : "✗",
      ingredientBoxes: result.ingredientBoxes.length,
      instructionBox: result.instructionBox ? "✓" : "✗",
      debug: result.debug,
    });

    return result;
  }, [tokens, selectedTitle]);

  // Flatten to array for rendering
  const sections = useMemo(() => {
    const result: SectionBox[] = [];
    if (sectionOverlays.titleBox) result.push(sectionOverlays.titleBox);
    result.push(...sectionOverlays.ingredientBoxes);
    if (sectionOverlays.instructionBox) result.push(sectionOverlays.instructionBox);

    console.log("[SectionOverlayViewer] Sections to render:", result.length);

    return result;
  }, [sectionOverlays]);

  useEffect(() => {
    function updateSize() {
      if (imageRef.current) {
        setRenderSize({
          width: imageRef.current.clientWidth,
          height: imageRef.current.clientHeight,
        });
      }
    }
    updateSize();
    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative overflow-hidden rounded-2xl border border-[#2c2620]/10 bg-[#f7efe3] print:hidden"
    >
      <img
        ref={imageRef}
        src={imageUrl}
        alt="Scanned recipe page"
        className="h-auto w-full object-contain"
        onLoad={() => {
          if (imageRef.current) {
            setRenderSize({
              width: imageRef.current.clientWidth,
              height: imageRef.current.clientHeight,
            });
          }
        }}
      />
      {showBoxes && sections.length > 0 && renderSize
        ? sections.map((section, idx) => {
            const [x1, y1, x2, y2] = section.bbox;

            // CRITICAL: Bboxes are in the ORIGINAL image pixel space (e.g., 500x819)
            // We need to get the natural image dimensions to scale properly
            // The imageSize prop would have this, but we can also use imageRef.naturalWidth/Height
            const naturalW = imageRef.current?.naturalWidth ?? 500;
            const naturalH = imageRef.current?.naturalHeight ?? 819;

            // Scale from original image space to displayed size
            const left = (x1 / naturalW) * renderSize.width;
            const top = (y1 / naturalH) * renderSize.height;
            const width = ((x2 - x1) / naturalW) * renderSize.width;
            const height = ((y2 - y1) / naturalH) * renderSize.height;

            // DEV: Log box rendering
            if (idx === 0) {
              console.log("[SectionOverlayViewer] Rendering box:", {
                label: section.label,
                rawBbox: section.bbox,
                naturalW,
                naturalH,
                renderSize,
                computed: { left, top, width, height },
              });
            }

            // Map section label to color
            const colorKey =
              section.label === "TITLE"
                ? "TITLE"
                : section.label === "INGREDIENTS"
                ? "INGREDIENT_LINE"
                : "INSTRUCTION_STEP";
            const color = getLabelColor(colorKey);

            // Display label text
            const displayLabel =
              section.label === "TITLE"
                ? "Title"
                : section.label === "INGREDIENTS"
                ? "Ingredients"
                : "Instructions";

            return (
              <div key={`section-${idx}`} className="absolute">
                {/* Bounding box */}
                <div
                  className="absolute rounded-lg border-2"
                  style={{
                    left,
                    top,
                    width,
                    height,
                    backgroundColor: `${color}11`, // Very low opacity fill (11 = 6.7%)
                    borderColor: color,
                  }}
                  onMouseMove={(event) => {
                    const rect = containerRef.current?.getBoundingClientRect();
                    if (!rect) return;
                    setTooltip({
                      section,
                      x: event.clientX - rect.left,
                      y: event.clientY - rect.top,
                    });
                  }}
                  onMouseLeave={() => setTooltip(null)}
                />

                {/* Label tag in top-left corner */}
                <div
                  className="absolute rounded-br-md rounded-tl-md px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-white shadow-sm"
                  style={{
                    left,
                    top,
                    backgroundColor: color,
                  }}
                >
                  {displayLabel}
                </div>
              </div>
            );
          })
        : null}

      {/* Dev debug banner */}
      {process.env.NODE_ENV === "development" && showBoxes && sections.length > 0 && (
        <div className="pointer-events-none absolute right-2 top-2 rounded-lg bg-blue-500 px-2 py-1 text-[10px] font-mono text-white opacity-75">
          Sections: {sections.length} | Render: {renderSize?.width ?? 0}x{renderSize?.height ?? 0}
        </div>
      )}

      {!showBoxes || sections.length === 0 ? (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-sm text-[#4b4237]">
          <div className="rounded-xl bg-white/80 px-4 py-2 shadow">
            {showBoxes ? (
              <div>
                <p className="font-semibold">No section overlays found</p>
                {sectionOverlays.debug && (
                  <p className="mt-1 text-xs">
                    Title match: {sectionOverlays.debug.selectedTitleMatch?.text ?? "none"} |
                    Ingredients: {sectionOverlays.debug.filteredIngredientCount} |
                    Instructions: {sectionOverlays.debug.filteredInstructionCount}
                  </p>
                )}
              </div>
            ) : (
              "Boxes hidden — enable in controls."
            )}
          </div>
        </div>
      ) : null}

      {tooltip && renderSize ? (
        <div
          className="pointer-events-none absolute max-w-[220px] rounded-xl border border-[#2c2620]/20 bg-white/95 px-3 py-2 text-xs text-[#1f1b16] shadow-lg"
          style={{
            left: Math.min(tooltip.x + 12, renderSize.width - 240),
            top: Math.max(tooltip.y - 10, 12),
          }}
        >
          <p className="text-[10px] uppercase tracking-[0.2em] text-[#6b8b6f]">
            {tooltip.section.label}
          </p>
          <p className="font-medium">
            {tooltip.section.tokenIds.length} token{tooltip.section.tokenIds.length !== 1 ? "s" : ""}
          </p>
          <p className="text-[10px] text-[#4b4237]">
            Confidence {(tooltip.section.confidence * 100).toFixed(1)}%
          </p>
        </div>
      ) : null}
    </div>
  );
}
