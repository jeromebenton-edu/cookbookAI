"use client";

import UploadDropzone from "../../components/UploadDropzone";

export default function TryPage() {
  return (
    <section className="flex flex-col gap-6">
      <div className="flex flex-col gap-4">
        <p className="text-xs uppercase tracking-[0.25em] text-[#6b8b6f]">
          Phase 2
        </p>
        <h1 className="display-font text-4xl font-semibold">
          Upload your own recipe scan.
        </h1>
        <p className="max-w-2xl text-sm text-[#4b4237]">
          The backend API accepts a PDF or image and runs LayoutLMv3 token
          classification. In mock mode, we return a plausible sample result so
          the UI stays usable.
        </p>
      </div>
      <UploadDropzone />
    </section>
  );
}
