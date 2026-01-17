"use client";

import { useState } from "react";
import { Recipe } from "../../lib/types";
import ConfidenceBadge from "./ConfidenceBadge";
import JsonViewerModal from "./JsonViewerModal";

type AiNarrativeProps = {
  recipe: Recipe;
  overallConfidence: number;
};

export default function AiNarrative({ recipe, overallConfidence }: AiNarrativeProps) {
  const [showOcrNote, setShowOcrNote] = useState(false);

  return (
    <div className="paper-card flex flex-col gap-3 p-6 text-sm text-[#2c2620]">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.22em] text-[#6b8b6f]">AI Parse View</p>
          <h3 className="display-font text-2xl font-semibold">Parsed from scan using LayoutLMv3</h3>
        </div>
        <ConfidenceBadge value={overallConfidence} />
      </div>
      <p className="leading-relaxed text-[#4b4237]">
        Visualize token predictions and bounding boxes over the scanned page. Adjust label filters and confidence to explore how the model reads the document.
      </p>
      <div className="border-t border-[#2c2620]/10 pt-3">
        <button
          onClick={() => setShowOcrNote(!showOcrNote)}
          className="flex items-center gap-2 text-xs text-[#6b8b6f] hover:text-[#4b7050] transition"
        >
          <span className="text-base">{showOcrNote ? "▼" : "▶"}</span>
          <span className="uppercase tracking-[0.15em]">Why does text look garbled?</span>
        </button>
        {showOcrNote && (
          <div className="mt-3 rounded-lg bg-[#f5f0e8] px-4 py-3 text-xs leading-relaxed text-[#4b4237]">
            <p className="font-medium text-[#2c2620]">Teaching Moment: OCR Pipeline Challenges</p>
            <p className="mt-2">
              Tesseract OCR struggles with 1896 two-column layouts, often reading <em>across</em> columns
              instead of <em>down</em> each column. This creates garbled text that LayoutLMv3 inherits.
            </p>
            <p className="mt-2">
              Notice the bounding boxes are correctly placed and confidence scores remain high—the model
              successfully identifies <em>what kind</em> of tokens these are, even when the text itself is wrong.
              This demonstrates how ML models can be robust at classification while still being limited by
              input quality. <span className="font-medium">Garbage in, garbage out.</span>
            </p>
            <a
              href="https://github.com/jeromebenton-edu/cookbookAI/blob/main/docs/DATA_QUALITY_TEACHING_NOTES.md#challenge-3-ocr-quality---the-achilles-heel"
              target="_blank"
              rel="noopener noreferrer"
              className="mt-2 inline-block text-[#6b8b6f] hover:text-[#4b7050] underline"
            >
              Read more in Teaching Notes →
            </a>
          </div>
        )}
      </div>
      <JsonViewerModal data={recipe.ai.raw ?? recipe.ai} />
    </div>
  );
}
