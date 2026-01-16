import { Recipe } from "../../lib/types";
import ConfidenceBadge from "./ConfidenceBadge";
import JsonViewerModal from "./JsonViewerModal";

type AiNarrativeProps = {
  recipe: Recipe;
  overallConfidence: number;
};

export default function AiNarrative({ recipe, overallConfidence }: AiNarrativeProps) {
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
      <JsonViewerModal data={recipe.ai.raw ?? recipe.ai} />
    </div>
  );
}
