import { useEffect, useState, useRef } from "react";
import ReactMarkdown from "react-markdown";

type Step = {
  id: string;
  navLabel: string;
  title: string;
  description: string;
  points?: string[];
};

type InstructionsModalProps = {
  open: boolean;
  onClose: () => void;
};

const steps: Step[] = [
  {
    id: "overview",
    navLabel: "Overview",
    title: "Welcome to VivBot - An AI Document Knowledge Search",
    description:
      "VivBot lets you upload documents, set configurable similarity metrics and Top K, run grounded Q&A, compare LLMs, generate AI reports and insights, and run critique pipelines for research.",
  },
  {
    id: "upload",
    navLabel: "Document upload & index",
    title: "1. Upload & index a document",
    description:
      "Start by uploading a document in section 1: “Upload & index a document”. Supported formats include PDF, TXT, CSV, DOCX, XLSX, and PPTX.",
    points: [
      "Click “Choose File” and pick a document from your computer.",
      "Press “Upload & Index” to chunk the file and create vector embeddings.",
      "Once indexed, the document appears in the dropdown used by the later sections.",
    ],
  },
  {
    id: "scope",
    navLabel: "Search scope",
    title: "2. Documents & search scope",
    description:
      "Use section 2 to decide which documents are used when answering questions or generating reports.",
    points: [
      "Choose “All documents” to search across all uploaded documents, or select a single document from the dropdown.",
      "“Generate AI Report” creates a structured explanation of the selected document.",
      "“Relations between these documents” explores how multiple documents are related in the vector space.",
      "“Remove all documents” removes all uploaded documents so you can start over.",
    ],
  },
  {
    id: "similarity",
    navLabel: "Similarity func. and Top K",
    title: "3. Similarity functions and Top K",
    description:
      "For experiments, VivBot lets you choose how similarity between embeddings is measured and how many Top K relevant chunks are retrieved.",
    points: [
      "You can pick the similarity function used to rank context chunks in all operations (Relations, Ask, Compare, Critique).",
      "Available metrics include:",
      "**Cosine similarity**: Measures the angle between two vectors. High when the embeddings point in the same direction, regardless of magnitude. Formula: (x · y) / (‖x‖ ‖y‖)",
      "**Negative Manhattan distance (L1)**: Uses the sum of absolute differences between coordinates. High (good) when vectors differ only slightly across many dimensions. Resilient to noise. Formula: −Σ |xᵢ − yᵢ|",
      "**Negative Euclidean distance (L2)**: Uses the straight line distance between vectors. Punishes large individual deviations strongly, making mismatches stand out. Formula: −√(Σ (xᵢ − yᵢ)²)",
      "**Dot product**: Measures magnitude × alignment. Favors vectors that are both similar and high energy (large norms). Formula: Σ xᵢ yᵢ",
      "**Hybrid (Cosine + Jaccard)**: Blends semantic similarity (cosine) with token overlap similarity (Jaccard). Rewards embeddings that match in meaning and share lexical structure. Formula: α·cosine(x,y) + (1−α)·(|A ∩ B| / |A ∪ B|)",
      "Changing the metric can affect which chunks are selected, how similarity is grounded, and how highlight rankings behave.",
      "Set “Top K” to control how many relevant chunks are retrieved from the vector store. **Top K** sets how many top ranked document chunks are selected. Higher values give the model more context, lower values make retrieval stricter.",
    ],
  },
  {
    id: "ask",
    navLabel: "Ask (Q&A)",
    title: "4. Ask questions (grounded Q&A)",
    description:
      "Section 3 answers questions using only the selected document(s) or all documents as context.",
    points: [
      "Check the “Answering for document …” text to see which document is active.",
      "Pick an LLM (for example LLaMA 3.1 8B Instant).",
      "Type a question about the selected document and press “Ask”.",
      "The answer appears on the right, together with the retrieved context.",
      "Use the **Highlight Context** after using Auto Insights on each answer card to inspect context and check highlighted parts of it:",
      "**AI**: highlights the exact spans the model seems to rely on most.",
      "**Keywords**: highlights chunks that best match your query terms.",
      "**Sentences**: highlights the most similar sentences in each chunk.",
      "**Off**: hides all highlighting if you just want to read the context.",
    ],
  },
  {
    id: "insights",
    navLabel: "Auto Insights",
    title: "5. Auto Insights",
    description:
      "Auto Insights provides a higher level layer of reasoning on top of Q&A and your uploaded documents. It transforms raw answers and source chunks into structured insights.",
    points: [
      "Creates a concise summary that captures the key ideas from the retrieved context.",
      "Extracts important entities such as people, organisations, technologies, frameworks, and locations.",
      "Identifies relevant keywords to give a quick sense of the document's focus areas.",
      "Generates 'Suggested Questions', that help you explore deeper or prepare for interviews/presentations.",
      "Builds a compact 'Mindmap style' text representation that connects different topics of the content.",
    ],
  },
  {
    id: "compare",
    navLabel: "Compare models",
    title: "6. Compare models",
    description:
      "Section 4 lets you compare how two different LLMs answer the same question.",
    points: [
      "Enter a question in the “Compare models” box.",
      "Choose Model A and Model B from the dropdowns.",
      "VivBot uses the same retrieved context for both models, so the comparison is fair.",
      "The Output panel shows a side by side card with both answers and their sources.",
    ],
  },
  {
    id: "critique",
    navLabel: "Critique",
    title: "7. Critique answer & prompt",
    description:
      "Section 5 analyses both the prompt and the model answer, and can run a double critique loop.",
    points: [
      "Ask a question you want to analyse more deeply in the “Critique answer & prompt” section.",
      "VivBot checks for prompt problems such as: missing context, vagueness, multi questions, or unclear audience.",
      "The answer itself is critiqued for grounding, reasoning quality, structure, and possible hallucinations.",
      "You can enable a **double critique loop**, where one model first critiques the answer and the prompt, suggest a new improved prompt, and this is again given back to the answering model to refine the answer.",
      "This pipeline helps you iteratively improve both the prompt and the answer quality.",
      "The output also shows the differences in the accuracy of the answer from both rounds.",
      "The output also shows the differences in the correctness, hallucinations, and similarity of the answer from both rounds.",
    ],
  },
  {
    id: "advanced",
    navLabel: "Advanced analysis",
    title: "8. Advanced analysis",
    description:
      "Advanced analysis runs a deeper, structured breakdown of an existing Ask / Compare / Critique result using the same scope, and Top K, but with every similarity function.",
    points: [
      "After you run **Ask**, **Compare**, or **Critique**, use the **Advanced analysis** button on that output card.",
      "This generates an additional analysis card that helps you inspect the result more deeply (useful for research / evaluation).",
      "If your retrieval settings change (scope / Top K), rerun the operation to analyze the new grounding context.",
    ],
  },
  {
    id: "output",
    navLabel: "Output panel",
    title: "9. Output panel on the right",
    description:
      "All results show up as separate cards on the right side of the screen.",
    points: [
      "Ask, Compare, AI Reports, Auto Insights, and Critique each create their own card.",
      "You can scroll down through previous results at any time.",
      "Use the **OUTPUT** and **OPERATIONS** buttons at the bottom on mobile to switch between viewing the output panel and the operations panel.",
    ],
  },
];

export default function InstructionsModal({ open, onClose }: InstructionsModalProps) {
  const [currentIndex, setCurrentIndex] = useState(0);

  const mobileTabsContainerRef = useRef<HTMLDivElement | null>(null);
  const mobileTabRefs = useRef<(HTMLButtonElement | null)[]>([]);

  useEffect(() => {
    if (open) {
      setCurrentIndex(0);
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const btn = mobileTabRefs.current[currentIndex];
    if (btn) {
      btn.scrollIntoView({
        behavior: "smooth",
        inline: "center",
        block: "nearest",
      });
    }
  }, [currentIndex, open]);

  if (!open) return null;

  const step = steps[currentIndex];
  const isFirst = currentIndex === 0;
  const isLast = currentIndex === steps.length - 1;

  const handleClose = () => {
    setCurrentIndex(0);
    onClose();
  };

  const handleNext = () => {
    if (!isLast) setCurrentIndex((i) => i + 1);
    else handleClose();
  };

  const handlePrev = () => {
    if (!isFirst) setCurrentIndex((i) => i - 1);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-2 sm:px-4">
      <div className="w-full max-w-5xl h-[90vh] sm:h-[80vh] rounded-2xl bg-zinc-900 p-4 sm:p-6 shadow-xl border border-zinc-700 flex flex-col">
        <div className="mb-4 flex items-center justify-between gap-2">
          <h2 className="text-lg font-semibold text-zinc-50">INSTRUCTIONS</h2>
          <button
            onClick={handleClose}
            className="rounded-full px-2 text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800"
          >
            ✕
          </button>
        </div>

        <div className="flex-1 flex flex-col md:flex-row gap-6 overflow-hidden">
          <div className="md:hidden">
            <p className="text-xs font-semibold uppercase tracking-wide text-zinc-400">
              Sections
            </p>
            <div
              className="mt-3 flex gap-2 overflow-x-auto pb-2 -mx-1 px-1"
              ref={mobileTabsContainerRef}
            >
              {steps.map((s, idx) => {
                const active = idx === currentIndex;
                return (
                  <button
                    key={s.id}
                    type="button"
                    onClick={() => setCurrentIndex(idx)}
                    ref={(el) => {
                      mobileTabRefs.current[idx] = el;
                    }}
                    className={`shrink-0 rounded-full px-3 py-1.5 text-xs sm:text-sm whitespace-nowrap border transition
                      ${active
                        ? "bg-emerald-500/10 text-emerald-300 border-emerald-500/60"
                        : "border-zinc-700 text-zinc-300 hover:bg-zinc-800 hover:text-zinc-50"
                      }`}
                  >
                    {s.navLabel}
                  </button>
                );
              })}
            </div>
          </div>

          <aside className="hidden md:block w-56 border-r border-zinc-800 pr-4 overflow-y-auto">
            <p className="text-xs font-semibold uppercase tracking-wide text-zinc-400">
              Sections
            </p>
            <div className="mt-3 flex flex-col gap-1">
              {steps.map((s, idx) => {
                const active = idx === currentIndex;
                return (
                  <button
                    key={s.id}
                    type="button"
                    onClick={() => setCurrentIndex(idx)}
                    className={`w-full rounded-lg px-3 py-1.5 text-left text-sm transition ${active
                      ? "bg-emerald-500/10 text-emerald-300 border border-emerald-500/40"
                      : "text-zinc-300 hover:bg-zinc-800 hover:text-zinc-50"
                      }`}
                  >
                    {s.navLabel}
                  </button>
                );
              })}
            </div>
          </aside>

          <main className="flex-1 flex flex-col min-h-0">
            <div className="flex-1 overflow-y-auto pr-1">
              <h3 className="text-xl font-semibold text-zinc-50">
                {step.title}
              </h3>
              <p className="mt-2 text-sm text-zinc-300">{step.description}</p>

              {step.points && (
                <ul className="mt-4 list-disc space-y-1 pl-5 text-sm text-zinc-200">
                  {step.points.map((p, idx) => (
                    <li key={idx}>
                      <ReactMarkdown
                        components={{
                          p: ({ children }) => <span>{children}</span>,
                        }}
                      >
                        {p}
                      </ReactMarkdown>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div className="mt-4 pt-4 border-t border-zinc-800 flex items-center justify-between">
              <div className="text-xs text-zinc-500">
                Step {currentIndex + 1} of {steps.length}
              </div>
              <div className="flex items-center gap-2">
                {!isFirst && (
                  <button
                    type="button"
                    onClick={handlePrev}
                    className="rounded-lg border border-zinc-600 px-3 py-1.5 text-sm text-zinc-100 hover:bg-zinc-800"
                  >
                    Back
                  </button>
                )}
                <button
                  type="button"
                  onClick={handleNext}
                  className="rounded-lg bg-emerald-500 px-4 py-1.5 text-sm font-medium text-zinc-900 hover:bg-emerald-400"
                >
                  {isLast ? "Finish" : "Next"}
                </button>
              </div>
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}
