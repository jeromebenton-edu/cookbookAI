"use client";

import { useEffect, useState } from "react";

function copyToClipboard(text: string) {
  if (typeof navigator !== "undefined" && navigator.clipboard) {
    void navigator.clipboard.writeText(text);
  }
}

type JsonViewerModalProps = {
  data: Record<string, unknown>;
  label?: string;
};

export default function JsonViewerModal({ data, label = "View raw JSON" }: JsonViewerModalProps) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return;
    const timeout = setTimeout(() => setCopied(false), 1500);
    return () => clearTimeout(timeout);
  }, [copied]);

  const pretty = JSON.stringify(data, null, 2);

  return (
    <div className="print-hidden">
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="rounded-full border border-[#2c2620]/15 bg-white px-4 py-2 text-xs uppercase tracking-[0.2em] text-[#2c2620] shadow-sm transition hover:-translate-y-[1px] hover:shadow-md"
      >
        {label}
      </button>

      {open ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm">
          <div className="max-h-[80vh] w-[min(960px,92vw)] overflow-hidden rounded-2xl border border-[#2c2620]/20 bg-white shadow-[0_24px_60px_rgba(31,27,22,0.25)]">
            <div className="flex items-center justify-between border-b border-[#2c2620]/10 px-4 py-3 text-sm text-[#2c2620]">
              <p className="text-xs uppercase tracking-[0.2em] text-[#6b8b6f]">Raw JSON</p>
              <div className="flex items-center gap-2 text-xs">
                <button
                  type="button"
                  onClick={() => {
                    copyToClipboard(pretty);
                    setCopied(true);
                  }}
                  className="rounded-full border border-[#2c2620]/15 bg-white px-3 py-1 shadow-sm transition hover:-translate-y-[1px] hover:shadow-md"
                >
                  {copied ? "Copied" : "Copy JSON"}
                </button>
                <button
                  type="button"
                  onClick={() => setOpen(false)}
                  className="rounded-full border border-[#2c2620]/15 bg-white px-3 py-1 shadow-sm transition hover:-translate-y-[1px] hover:shadow-md"
                >
                  Close
                </button>
              </div>
            </div>
            <pre className="max-h-[70vh] overflow-auto bg-[#f7efe3] px-4 py-4 text-xs leading-relaxed text-[#2c2620]">
              {pretty}
            </pre>
          </div>
        </div>
      ) : null}
    </div>
  );
}
