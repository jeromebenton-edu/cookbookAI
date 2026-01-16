"use client";

import { areExamplesValid } from "../../lib/bundledExamples";

/**
 * Shows a warning banner if demo examples have validation issues
 * Only visible in production (dev shows console warnings instead)
 */
export default function ValidationBanner() {
  const isValid = areExamplesValid();
  const isProd = process.env.NODE_ENV === "production";

  // Only show banner in production if invalid
  if (isValid || !isProd) {
    return null;
  }

  return (
    <div className="rounded-xl border border-yellow-300 bg-yellow-50 px-4 py-3 text-sm">
      <div className="flex items-start gap-3">
        <span className="text-yellow-600">⚠️</span>
        <div>
          <p className="font-medium text-yellow-800">Demo assets misconfigured</p>
          <p className="text-yellow-700">
            Showing bundled output. Some scan images may not match the extracted recipes.
          </p>
        </div>
      </div>
    </div>
  );
}
