import Link from "next/link";

const steps = [
  {
    title: "Scan",
    description: "Digitized pages from the 1918 Boston Cooking-School Cook Book."
  },
  {
    title: "OCR",
    description: "Text and bounding boxes from optical character recognition."
  },
  {
    title: "LayoutLMv3",
    description: "Token classification for titles, ingredients, and steps."
  },
  {
    title: "Postprocess",
    description: "Line grouping, regex extraction, and confidence scoring."
  },
  {
    title: "Cook View",
    description: "A clean recipe experience built with Next.js and Tailwind."
  }
];

export default function AboutPage() {
  return (
    <section className="flex flex-col gap-12">
      <div className="flex flex-col gap-4">
        <p className="text-xs uppercase tracking-[0.25em] text-[#6b8b6f]">
          About the Pipeline
        </p>
        <h1 className="display-font text-4xl font-semibold">Why LayoutLMv3?</h1>
        <p className="max-w-2xl text-sm text-[#4b4237]">
          LayoutLMv3 blends text and layout, helping us reconstruct recipe
          structure from noisy scans. This project shows both the polished cook
          view and the raw AI parse so you can see what the model sees.
        </p>
      </div>

      <div className="paper-card grid gap-6 p-8 md:grid-cols-5">
        {steps.map((step, index) => (
          <div key={step.title} className="flex flex-col gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl border border-[#2c2620]/20 bg-white text-sm font-semibold">
              {index + 1}
            </div>
            <p className="display-font text-lg font-semibold">{step.title}</p>
            <p className="text-xs text-[#4b4237]">{step.description}</p>
          </div>
        ))}
      </div>

      <div className="glass-panel flex flex-col gap-4 p-8">
        <p className="text-xs uppercase tracking-[0.25em] text-[#6b8b6f]">
          Origin Story
        </p>
        <h2 className="display-font text-2xl font-semibold">
          How This Project Came to Be
        </h2>
        <div className="flex flex-col gap-3 text-sm text-[#4b4237]">
          <p>
            On a drive home, I mentioned to my wife that I wanted to start a new teaching project for my AI students. She suggested a cooking website that somehow used AI to organize recipes. After some research, I discovered Fanny Farmer's <em>Boston Cooking-School Cook Book</em>, published in 1896.
          </p>
          <p>
            When I saw the abysmal quality of the scanned copy, I knew I had found the perfect challenge: parse a horrible scan of a book from the letterpress era with hot-metal composition. This guaranteed a host of problems—page geometry artifacts, typography quirks, complex layout structure, semantic corruption from tables and numeric density, plus all the usual scanning issues like noise, stains, and paper degradation.
          </p>
          <p>
            Add to that language drift: archaic terms, abbreviations, and period-specific recipe conventions. I remember thinking, "Well, I've completed NLP and parsing projects before, and I've used LayoutLMv3 on a recent project with great success, so why not?"
          </p>
        </div>
      </div>

      <div className="paper-card flex flex-col gap-4 p-8">
        <h2 className="display-font text-2xl font-semibold">
          Project Documentation
        </h2>
        <p className="text-sm text-[#4b4237]">
          The GitHub repository includes labeling scripts and training
          scaffolding for future fine-tuning.
        </p>
        <div className="flex gap-3 text-xs uppercase tracking-[0.25em] text-[#6b8b6f]">
          <Link href="https://github.com/jeromebenton-edu/cookbookAI" className="hover:text-[#1f1b16]">
            GitHub Docs
          </Link>
        </div>
      </div>
    </section>
  );
}
