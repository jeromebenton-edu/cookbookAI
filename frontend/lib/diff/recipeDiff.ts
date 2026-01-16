export type DiffEntry = {
  curated?: string;
  ai?: string;
  status: "match" | "missing" | "extra" | "mismatch";
  similarity?: number;
};

function normalize(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, "").replace(/\s+/g, " ").trim();
}

function similarity(a: string, b: string): number {
  const na = normalize(a);
  const nb = normalize(b);
  if (!na && !nb) return 1;
  if (!na || !nb) return 0;
  const setA = new Set(na.split(" "));
  const setB = new Set(nb.split(" "));
  const inter = [...setA].filter((w) => setB.has(w)).length;
  const denom = (setA.size + setB.size) / 2 || 1;
  return inter / denom;
}

export function diffLines(curated: string[], ai: string[], threshold = 0.75): DiffEntry[] {
  const entries: DiffEntry[] = [];
  const usedAi = new Set<number>();

  curated.forEach((cur) => {
    let bestIdx = -1;
    let bestScore = -1;
    ai.forEach((aiLine, idx) => {
      if (usedAi.has(idx)) return;
      const score = similarity(cur, aiLine);
      if (score > bestScore) {
        bestScore = score;
        bestIdx = idx;
      }
    });

    if (bestIdx >= 0 && bestScore >= threshold) {
      usedAi.add(bestIdx);
      const status = bestScore >= 0.95 ? "match" : "mismatch";
      entries.push({ curated: cur, ai: ai[bestIdx], status, similarity: bestScore });
    } else if (bestIdx >= 0 && bestScore > 0) {
      usedAi.add(bestIdx);
      entries.push({ curated: cur, ai: ai[bestIdx], status: "mismatch", similarity: bestScore });
    } else {
      entries.push({ curated: cur, status: "missing", similarity: 0 });
    }
  });

  ai.forEach((aiLine, idx) => {
    if (usedAi.has(idx)) return;
    entries.push({ ai: aiLine, status: "extra", similarity: undefined });
  });

  return entries;
}
