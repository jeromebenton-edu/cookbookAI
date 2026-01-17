"use client";

import { useState, useRef, useEffect } from "react";
import { downloadJson } from "../lib/export/downloadJson";

interface CorrectionExportProps {
  exportJson: (meta: Record<string, unknown>) => Record<string, unknown>;
  pageNum: number;
  overallConf: number;
}

export default function CorrectionExport({ exportJson, pageNum, overallConf }: CorrectionExportProps) {
  const [showDropdown, setShowDropdown] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [copied, setCopied] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowDropdown(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const filename = `boston_page_${pageNum.toString().padStart(4, "0")}_corrected.json`;
  const meta = { model: "layoutlmv3", overall_conf: overallConf };

  const handleDownload = () => {
    downloadJson(exportJson(meta), filename);
    setShowDropdown(false);
  };

  const handleCopy = async () => {
    await navigator.clipboard.writeText(JSON.stringify(exportJson(meta), null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    setShowDropdown(false);
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        {/* Export dropdown */}
        <div className="relative" ref={dropdownRef}>
          <button
            className="rounded-full border border-[#2c2620]/15 bg-white px-4 py-2 text-xs uppercase tracking-[0.2em] text-[#2c2620] shadow-sm transition hover:-translate-y-[1px] hover:shadow-md flex items-center gap-2"
            onClick={() => setShowDropdown(!showDropdown)}
          >
            Export Corrections
            <span className="text-[10px]">{showDropdown ? "▲" : "▼"}</span>
          </button>
          {showDropdown && (
            <div className="absolute left-0 top-full mt-1 z-10 min-w-[180px] rounded-lg border border-[#2c2620]/15 bg-white shadow-lg">
              <button
                className="w-full px-4 py-2 text-left text-xs hover:bg-[#f7efe3] transition flex items-center gap-2"
                onClick={handleDownload}
              >
                <span>📥</span> Download as file
              </button>
              <button
                className="w-full px-4 py-2 text-left text-xs hover:bg-[#f7efe3] transition flex items-center gap-2 border-t border-[#2c2620]/10"
                onClick={handleCopy}
              >
                <span>📋</span> {copied ? "Copied!" : "Copy to clipboard"}
              </button>
            </div>
          )}
        </div>

        {/* Help toggle */}
        <button
          className="text-xs text-[#6b8b6f] hover:text-[#4b7050] transition flex items-center gap-1"
          onClick={() => setShowHelp(!showHelp)}
        >
          <span>{showHelp ? "▼" : "▶"}</span>
          What do I do with this?
        </button>
      </div>

      {/* Help panel */}
      {showHelp && (
        <div className="rounded-lg bg-[#f5f0e8] px-4 py-3 text-xs leading-relaxed text-[#4b4237]">
          <p className="font-medium text-[#2c2620]">Corrections Workflow</p>
          <ol className="mt-2 list-decimal pl-4 space-y-1">
            <li>Edit AI-extracted fields until they match the source document</li>
            <li>Click <strong>Export Corrections</strong> → <strong>Download as file</strong></li>
            <li>Save to <code className="bg-white/50 px-1 rounded">data/corrections/</code> folder</li>
            <li>Use corrections to update curated recipes or retrain the model</li>
          </ol>
          <p className="mt-2 text-[#6b8b6f]">
            The exported JSON includes your edits, the original AI output, and metadata for training pipelines.
          </p>
        </div>
      )}
    </div>
  );
}
