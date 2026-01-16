"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { RecipeToken } from "../../lib/types";
import { getLabelColor } from "./constants";

export type DocOverlayViewerProps = {
  imageUrl: string;
  imageSize?: { width: number; height: number };
  tokens: RecipeToken[];
  visibleLabels: Set<string>;
  showBoxes: boolean;
  confidenceThreshold: number;
};

type TooltipState = {
  token: RecipeToken;
  x: number;
  y: number;
};

export default function DocOverlayViewer({
  imageUrl,
  imageSize,
  tokens,
  visibleLabels,
  showBoxes,
  confidenceThreshold
}: DocOverlayViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const [renderSize, setRenderSize] = useState<{ width: number; height: number } | null>(null);

  const filteredTokens = useMemo(() => {
    return tokens.filter(
      (token) =>
        visibleLabels.has(token.label) &&
        token.score >= confidenceThreshold &&
        token.bbox?.length === 4
    );
  }, [tokens, visibleLabels, confidenceThreshold]);

  useEffect(() => {
    function updateSize() {
      if (imageRef.current) {
        setRenderSize({
          width: imageRef.current.clientWidth,
          height: imageRef.current.clientHeight
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
              height: imageRef.current.clientHeight
            });
          }
        }}
      />
      {showBoxes && filteredTokens.length > 0 && renderSize
        ? filteredTokens.map((token, idx) => {
            const [x1, y1, x2, y2] = token.bbox;

            // If imageSize is provided, bboxes are in absolute pixels of the original image
            // Otherwise, assume normalized coordinates (0-1000 range)
            const baseW = imageSize?.width ?? 1000;
            const baseH = imageSize?.height ?? 1000;

            // Debug logging for first token
            if (idx === 0) {
              console.log('[DocOverlay] First token debug:', {
                bbox: token.bbox,
                imageSize,
                renderSize,
                baseW,
                baseH,
                scaledLeft: (x1 / baseW) * renderSize.width,
                scaledTop: (y1 / baseH) * renderSize.height
              });
            }

            // Scale from original image coordinates to rendered size
            const left = (x1 / baseW) * renderSize.width;
            const top = (y1 / baseH) * renderSize.height;
            const width = ((x2 - x1) / baseW) * renderSize.width;
            const height = ((y2 - y1) / baseH) * renderSize.height;
            const color = getLabelColor(token.label);
            return (
              <div
                key={token.id}
                className="absolute rounded-[6px] border"
                style={{
                  left,
                  top,
                  width,
                  height,
                  backgroundColor: `${color}22`,
                  borderColor: color
                }}
                onMouseMove={(event) => {
                  const rect = containerRef.current?.getBoundingClientRect();
                  if (!rect) return;
                  setTooltip({
                    token,
                    x: event.clientX - rect.left,
                    y: event.clientY - rect.top
                  });
                }}
                onMouseLeave={() => setTooltip(null)}
              />
            );
          })
        : null}

      {!showBoxes || filteredTokens.length === 0 ? (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-sm text-[#4b4237]">
          <div className="rounded-xl bg-white/80 px-4 py-2 shadow">
            {showBoxes ? "No token overlays available yet." : "Boxes hidden — enable in controls."}
          </div>
        </div>
      ) : null}

      {tooltip && renderSize ? (
        <div
          className="pointer-events-none absolute max-w-[220px] rounded-xl border border-[#2c2620]/20 bg-white/95 px-3 py-2 text-xs text-[#1f1b16] shadow-lg"
          style={{
            left: Math.min(tooltip.x + 12, renderSize.width - 240),
            top: Math.max(tooltip.y - 10, 12)
          }}
        >
          <p className="text-[10px] uppercase tracking-[0.2em] text-[#6b8b6f]">
            {tooltip.token.label}
          </p>
          <p className="font-medium">{tooltip.token.text}</p>
          <p className="text-[10px] text-[#4b4237]">
            Confidence {(tooltip.token.score * 100).toFixed(1)}%
          </p>
        </div>
      ) : null}
    </div>
  );
}
