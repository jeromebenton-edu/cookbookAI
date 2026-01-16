"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

type SearchBarProps = {
  initialValue?: string;
  onSearch?: (value: string) => void;
  placeholder?: string;
};

export default function SearchBar({
  initialValue = "",
  onSearch,
  placeholder = "Search recipes or ingredients"
}: SearchBarProps) {
  const router = useRouter();
  const [value, setValue] = useState(initialValue);

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    const trimmed = value.trim();
    if (onSearch) {
      onSearch(trimmed);
      return;
    }
    const params = new URLSearchParams();
    if (trimmed) {
      params.set("query", trimmed);
    }
    router.push(`/recipes?${params.toString()}`);
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="flex w-full flex-col gap-3 rounded-2xl border border-[#2c2620]/10 bg-white/80 p-4 shadow-[0_15px_35px_rgba(31,27,22,0.1)] sm:flex-row sm:items-center"
    >
      <input
        className="flex-1 rounded-xl border border-[#2c2620]/10 bg-white/70 px-4 py-3 text-sm focus:border-[#b8793b] focus:outline-none"
        value={value}
        onChange={(event) => setValue(event.target.value)}
        placeholder={placeholder}
        aria-label="Search recipes"
      />
      <button
        type="submit"
        className="rounded-xl bg-[#2c2620] px-6 py-3 text-xs uppercase tracking-[0.25em] text-[#f7efe3] transition hover:translate-y-[-1px]"
      >
        Search
      </button>
    </form>
  );
}
