"use client";

import { useRef, useState } from "react";
import { RecipeToken } from "../lib/types";

const labelStyles: Record<string, string> = {
  TITLE: "#b8793b",
  INGREDIENT_LINE: "#6b8b6f",
  INSTRUCTION_STEP: "#2c2620",
  META: "#8b5e3c",
  OTHER: "#7c6f64"
};

type DocOverlayViewerProps = {
  imageUrl: string;
  pageSize: { width: number; height: number };
  tokens: RecipeToken[];
};

export default function DocOverlayViewer({
  imageUrl,
  pageSize,
  tokens
}: DocOverlayViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [hovered, setHovered] = useState<RecipeToken | null>(null);
  const [cursor, setCursor] = useState<{
    x: number;
    y: number;
    width: number;
    height: number;
  } | null>(null);

  const handleMove = (
    event: React.MouseEvent<SVGRectElement, MouseEvent>,
    token: RecipeToken
  ) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) {
      return;
    }
    setHovered(token);
    setCursor({
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
      width: rect.width,
      height: rect.height
    });
  };

  return (
    <div
      ref={containerRef}
      className="relative overflow-hidden rounded-2xl border border-[#2c2620]/10 bg-[#f7efe3]"
    >
      <img
        src={imageUrl}
        alt="Scanned recipe page"
        className="h-auto w-full object-contain"
      />
      <svg
        className="absolute inset-0 h-full w-full"
        viewBox={`0 0 ${pageSize.width} ${pageSize.height}`}
        preserveAspectRatio="xMidYMid meet"
      >
        {tokens.map((token) => {
          const [x1, y1, x2, y2] = token.bbox;
          const color = labelStyles[token.label] ?? "#7c6f64";
          return (
            <rect
              key={token.id}
              x={x1}
              y={y1}
              width={x2 - x1}
              height={y2 - y1}
              fill={`${color}22`}
              stroke={color}
              strokeWidth={1.5}
              onMouseMove={(event) => handleMove(event, token)}
              onMouseLeave={() => {
                setHovered(null);
                setCursor(null);
              }}
            />
          );
        })}
      </svg>
      {hovered && cursor ? (
        <div
          className="pointer-events-none absolute rounded-xl border border-[#2c2620]/20 bg-white/90 px-3 py-2 text-xs text-[#1f1b16] shadow-lg"
          style={{
            left: Math.max(12, Math.min(cursor.x + 12, cursor.width - 220)),
            top: Math.max(cursor.y - 10, 12)
          }}
        >
          <p className="text-[10px] uppercase tracking-[0.2em] text-[#6b8b6f]">
            {hovered.label}
          </p>
          <p className="font-medium">{hovered.text}</p>
          <p className="text-[10px] text-[#4b4237]">
            {Math.round(hovered.score * 100)}% confidence
          </p>
        </div>
      ) : null}
    </div>
  );
}
