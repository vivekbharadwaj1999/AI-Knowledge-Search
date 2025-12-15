import { Fragment, useEffect, useRef, useState } from "react";
import { Dialog, Transition } from "@headlessui/react";

interface AnalysisModalProps {
  isOpen: boolean;
  onClose: () => void;
  data: any;
  selectedMethod?: string;
}

type MethodKey = "cosine" | "dot" | "neg_l2" | "neg_l1" | "hybrid";

const METHODS: MethodKey[] = ["cosine", "dot", "neg_l2", "neg_l1", "hybrid"];

const METHOD_INFO: Record<MethodKey, { name: string }> = {
  cosine: { name: "Cosine" },
  dot: { name: "Dot Product" },
  neg_l2: { name: "L2 Distance" },
  neg_l1: { name: "L1 Distance" },
  hybrid: { name: "Hybrid" },
};

const METHOD_STYLES: Record<
  MethodKey,
  {
    titleText: string;
    selectedContainer: string;
    rankText: string;
  }
> = {
  cosine: {
    titleText: "text-sky-300",
    selectedContainer: "bg-sky-950/30 border-sky-700/70",
    rankText: "text-sky-400",
  },
  dot: {
    titleText: "text-violet-300",
    selectedContainer: "bg-violet-950/30 border-violet-700/70",
    rankText: "text-violet-400",
  },
  neg_l2: {
    titleText: "text-amber-300",
    selectedContainer: "bg-amber-950/30 border-amber-700/70",
    rankText: "text-amber-400",
  },
  neg_l1: {
    titleText: "text-emerald-300",
    selectedContainer: "bg-emerald-950/30 border-emerald-700/70",
    rankText: "text-emerald-400",
  },
  hybrid: {
    titleText: "text-rose-300",
    selectedContainer: "bg-rose-950/30 border-rose-700/70",
    rankText: "text-rose-400",
  },
};

function clampMethodKey(m: string | undefined): MethodKey {
  if (!m) return "cosine";
  return (METHODS.includes(m as MethodKey) ? m : "cosine") as MethodKey;
}

function normalizeToFive(v: any): number | null {
  if (v === null || v === undefined) return null;
  const n = Number(v);
  if (Number.isNaN(n)) return null;
  if (n >= 0 && n <= 1) return Math.round(n * 5 * 10) / 10;
  return n;
}

function ScoreGrid({ scores }: { scores?: any }) {
  const c = normalizeToFive(scores?.correctness);
  const co = normalizeToFive(scores?.completeness);
  const cl = normalizeToFive(scores?.clarity);
  const h = normalizeToFive(scores?.hallucination_risk);

  const fmt = (x: number | null) => (x === null ? "—" : x);

  return (
    <div className="mt-2 bg-slate-800/40 rounded p-3">
      <div className="grid grid-cols-4 gap-2">
        <div className="text-center">
          <div className="text-[9px] text-slate-400">Correctness</div>
          <div className="text-sm font-bold text-emerald-400">
            {fmt(c)}/5
          </div>
        </div>
        <div className="text-center">
          <div className="text-[9px] text-slate-400">Completeness</div>
          <div className="text-sm font-bold text-sky-400">{fmt(co)}/5</div>
        </div>
        <div className="text-center">
          <div className="text-[9px] text-slate-400">Clarity</div>
          <div className="text-sm font-bold text-violet-400">{fmt(cl)}/5</div>
        </div>
        <div className="text-center">
          <div className="text-[9px] text-slate-400">Risk</div>
          <div className="text-sm font-bold text-amber-400">{fmt(h)}/5</div>
        </div>
      </div>
    </div>
  );
}

function Block({
  title,
  tone = "neutral",
  children,
}: {
  title: string;
  tone?: "neutral" | "answer" | "critique" | "prompt";
  children: React.ReactNode;
}) {
  const titleClass =
    tone === "answer"
      ? "text-emerald-300"
      : tone === "critique"
        ? "text-amber-300"
        : tone === "prompt"
          ? "text-sky-300"
          : "text-slate-300";

  const borderClass =
    tone === "answer"
      ? "border-emerald-900"
      : tone === "critique"
        ? "border-amber-900"
        : tone === "prompt"
          ? "border-sky-900"
          : "border-slate-800";

  return (
    <div className={`rounded-lg bg-slate-950/60 border ${borderClass} p-3`}>
      <div className={`text-[10px] font-semibold uppercase mb-2 ${titleClass}`}>
        {title}
      </div>
      <div className="text-xs text-slate-100 whitespace-pre-wrap leading-relaxed">
        {children}
      </div>
    </div>
  );
}

function HoverBubble({
  text,
  isPinned,
  bubbleRef,
}: {
  text: string;
  isPinned: boolean;
  bubbleRef?: React.RefObject<HTMLDivElement | null>;
}) {
  const visibilityClass = isPinned ? "block" : "hidden group-hover:block";
  return (
    <div
      ref={isPinned ? (bubbleRef as any) : undefined}
      className={`absolute z-[9999] ${visibilityClass}
                  left-0 top-full mt-2
                  w-[min(520px,80vw)]
                  rounded-xl border border-slate-700 bg-slate-950/95
                  shadow-2xl p-3`}
    >
      <div className="text-[10px] text-slate-400 mb-1">
        Full chunk {isPinned ? "(pinned)" : ""}
      </div>

      <pre className="whitespace-pre-wrap font-mono text-[10px] text-slate-100 leading-relaxed max-h-64 overflow-auto">
        {text}
      </pre>

      <div className="mt-2 text-[10px] text-slate-500">
        Tip: click outside to close
      </div>
    </div>
  );
}

export default function UnifiedAnalysisModal({
  isOpen,
  onClose,
  data,
  selectedMethod = "cosine",
}: AnalysisModalProps) {
  if (!data) return null;

  const [pinnedKey, setPinnedKey] = useState<string | null>(null);
  const pinnedContainerRef = useRef<HTMLDivElement | null>(null);
  const pinnedBubbleRef = useRef<HTMLDivElement | null>(null);
  const queryAnalysis = data.retrieval_details?.query_analysis || {};
  const embeddingPreview: number[] | undefined = queryAnalysis.embedding_preview;
  const embeddingDimension: number | undefined = queryAnalysis.embedding_dimension;
  const fullEmbedding: number[] | undefined = data.query_embedding;
  const [showFullEmbedding, setShowFullEmbedding] = useState(false);

  const numericEmbedding =
    showFullEmbedding && Array.isArray(fullEmbedding)
      ? fullEmbedding
      : embeddingPreview;

  const handleExportJson = () => {
    try {
      const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);

      const a = document.createElement("a");
      a.href = url;
      a.download = `vivbot-advanced-analysis-${data?.operation || "ask"
        }.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);

      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Failed to export analysis JSON", err);
    }
  };

  useEffect(() => {
    function onMouseDown(e: MouseEvent) {
      if (!pinnedKey) return;

      const target = e.target as Node;
      if (
        (pinnedContainerRef.current &&
          pinnedContainerRef.current.contains(target)) ||
        (pinnedBubbleRef.current && pinnedBubbleRef.current.contains(target))
      ) {
        return;
      }
      setPinnedKey(null);
    }

    document.addEventListener("mousedown", onMouseDown);
    return () => document.removeEventListener("mousedown", onMouseDown);
  }, [pinnedKey]);

  const operation: "ask" | "compare" | "critique" =
    data.operation || "ask"; // safe fallback
  const chosenMethod = clampMethodKey(selectedMethod);

  const renderOperationResults = (method: MethodKey) => {
    const result = data.results_by_method?.[method];
    if (!result) return null;

    if (operation === "ask") {
      return (
        <div className="mt-3 bg-slate-800/40 rounded p-3">
          <div className="text-[10px] text-slate-400 mb-2">Generated Answer:</div>
          <div className="text-xs text-slate-200 leading-relaxed whitespace-pre-wrap">
            {result.answer}
          </div>
          <div className="mt-2 text-[9px] text-slate-500">
            Length: {result.answer_length} chars | Context: {result.context_used} chunks
          </div>
        </div>
      );
    }

    if (operation === "compare") {
      return (
        <div className="mt-3 space-y-2">
          {Object.entries(result.answers_by_model || {}).map(
            ([model, answerObj]: [string, any]) => (
              <div key={model} className="bg-slate-800/40 rounded p-3">
                <div className="text-[10px] text-violet-400 font-semibold mb-1">
                  {model}
                </div>
                <div className="text-xs text-slate-200 leading-relaxed whitespace-pre-wrap">
                  {answerObj?.answer ?? ""}
                </div>
                <div className="mt-1 text-[9px] text-slate-500">
                  Length: {answerObj?.length ?? (answerObj?.answer?.length ?? 0)} chars
                </div>
              </div>
            )
          )}
        </div>
      );
    }
    if (operation === "critique") {
      const cr = result.critique_result || {};
      const rounds = Array.isArray(cr.rounds) ? cr.rounds : [];
      const hasRounds = rounds.length > 0;
      const fallbackAnswer =
        typeof cr.answer === "string" ? cr.answer : "";
      const fallbackCritique =
        typeof cr.answer_critique_markdown === "string"
          ? cr.answer_critique_markdown
          : typeof cr.answer_critique === "string"
            ? cr.answer_critique
            : "";
      const fallbackScores = cr.scores || cr.final_verdict || null;
      const fallbackImprovedPrompt =
        typeof cr.improved_prompt === "string" ? cr.improved_prompt : "";

      return (
        <div className="mt-3 space-y-3">
          <div className="text-[10px] text-slate-500">
            Rounds: {hasRounds ? rounds.length : cr.num_rounds ?? 1} | Context:{" "}
            {result.context_used} chunks
          </div>
          {!hasRounds && (
            <>
              {fallbackAnswer && (
                <Block title="Answer (base model)" tone="answer">
                  {fallbackAnswer}
                </Block>
              )}

              {fallbackCritique && (
                <Block title="Critique" tone="critique">
                  {fallbackCritique}
                </Block>
              )}

              {fallbackScores && <ScoreGrid scores={fallbackScores} />}
            </>
          )}
          {hasRounds && (
            <>
              {rounds[0] && (
                <>
                  {typeof rounds[0].answer === "string" && rounds[0].answer && (
                    <Block title="Round 1 — Answer" tone="answer">
                      {rounds[0].answer}
                    </Block>
                  )}

                  {typeof rounds[0].answer_critique_markdown === "string" &&
                    rounds[0].answer_critique_markdown && (
                      <Block title="Round 1 — Critique" tone="critique">
                        {rounds[0].answer_critique_markdown}
                      </Block>
                    )}

                  {rounds[0].scores && <ScoreGrid scores={rounds[0].scores} />}
                </>
              )}
              {rounds[0]?.improved_prompt && (
                <Block title="Improved prompt (for next round)" tone="prompt">
                  {rounds[0].improved_prompt}
                </Block>
              )}
              {rounds[1] && (
                <>
                  {typeof rounds[1].answer === "string" && rounds[1].answer && (
                    <Block title="Round 2 — Answer (final)" tone="answer">
                      {rounds[1].answer}
                    </Block>
                  )}

                  {rounds[1].scores && <ScoreGrid scores={rounds[1].scores} />}
                </>
              )}

              {!rounds[0]?.improved_prompt && fallbackImprovedPrompt && (
                <Block title="Improved prompt" tone="prompt">
                  {fallbackImprovedPrompt}
                </Block>
              )}
            </>
          )}
        </div>
      );
    }

    return null;
  };

  const input = data.input || {};
  const inputQuestion = input.question || "—";
  const inputTopK = input.top_k ?? "—";
  const answerModel = input.answer_model;
  const criticModel = input.critic_model;

  return (
    <Transition appear show={isOpen} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={onClose}>
        <Transition.Child
          as={Fragment}
          enter="ease-out duration-200"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-150"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-black/80 backdrop-blur-sm" />
        </Transition.Child>

        <div className="fixed inset-0 overflow-y-auto">
          <div className="flex min-h-full items-center justify-center p-1 sm:p-4">
            <Transition.Child
              as={Fragment}
              enter="ease-out duration-200"
              enterFrom="opacity-0 scale-95"
              enterTo="opacity-100 scale-100"
              leave="ease-in duration-150"
              leaveFrom="opacity-100 scale-100"
              leaveTo="opacity-0 scale-95"
            >
              <Dialog.Panel className="w-full max-w-7xl transform overflow-hidden rounded-xl sm:rounded-2xl bg-slate-900 border border-slate-700 p-3 sm:p-6 text-left align-middle shadow-2xl transition-all">
                <div className="flex items-start justify-between gap-3 mb-4">
                  <div className="min-w-0">
                    <Dialog.Title className="text-lg sm:text-xl font-bold text-slate-100 flex items-baseline gap-2 whitespace-nowrap">
                      <span>Advanced Analysis</span>
                      <span className="text-sm sm:text-base font-normal text-slate-400">
                        ({operation.toUpperCase()})
                      </span>
                    </Dialog.Title>

                    <div className="mt-1 text-[11px] sm:text-xs text-slate-400 leading-snug">
                      Comparing how different retrieval methods affect {operation} results
                    </div>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    <button
                      type="button"
                      onClick={handleExportJson}
                      className="text-[10px] px-2 py-1 rounded border border-slate-700 text-slate-300 hover:bg-slate-800 transition"
                    >
                      Export JSON
                    </button>

                    <button
                      onClick={onClose}
                      className="rounded-full p-2 hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition"
                    >
                      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                </div>

                <div className="space-y-6 max-h-[75vh] overflow-y-auto pr-2">
                  <section className="bg-slate-950/60 border border-slate-800 rounded-lg p-4">
                    <h3 className="text-sm font-bold text-sky-300 mb-3">
                      INPUT
                    </h3>
                    <div
                      className={
                        operation === "critique"
                          ? "grid grid-cols-1 md:grid-cols-5 gap-3 text-xs"
                          : "grid grid-cols-1 md:grid-cols-4 gap-3 text-xs"
                      }
                    >
                      <div
                        className={
                          operation === "critique"
                            ? "bg-slate-900/60 border border-slate-700 rounded p-3 md:col-span-2"
                            : "bg-slate-900/60 border border-slate-700 rounded p-3 md:col-span-2"
                        }
                      >
                        <div className="text-slate-400 mb-1">Question</div>
                        <div className="text-slate-100 font-mono text-[11px] truncate">
                          "{inputQuestion}"
                        </div>
                      </div>

                      {operation === "ask" && (
                        <div className="bg-slate-900/60 border border-slate-700 rounded p-3">
                          <div className="text-slate-400 mb-1">Model</div>
                          <div className="text-violet-300 font-mono text-sm pt-1 truncate">
                            {input.model ?? "—"}
                          </div>
                        </div>
                      )}

                      {operation === "compare" && (
                        <div className="bg-slate-900/60 border border-slate-700 rounded p-3">
                          <div className="text-slate-400 mb-1">Models</div>
                          <div className="text-violet-300 font-mono text-sm pt-1">
                            {Array.isArray(input.models) ? (
                              <div className="space-y-1">
                                {input.models.map((m: string, i: number) => (
                                  <div key={i} className="truncate">
                                    {m}
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div className="truncate">—</div>
                            )}
                          </div>
                        </div>
                      )}

                      {operation === "critique" && (
                        <>
                          <div className="bg-slate-900/60 border border-slate-700 rounded p-3">
                            <div className="text-slate-400 mb-1">Answer model</div>
                            <div className="text-violet-300 font-mono text-sm pt-1 truncate">
                              {answerModel ?? "—"}
                            </div>
                          </div>

                          <div className="bg-slate-900/60 border border-slate-700 rounded p-3">
                            <div className="text-slate-400 mb-1">Critique model</div>
                            <div className="text-violet-300 font-mono text-sm pt-1 truncate">
                              {criticModel ?? "—"}
                            </div>
                          </div>
                        </>
                      )}

                      <div className="bg-slate-900/60 border border-slate-700 rounded p-3">
                        <div className="text-slate-400 mb-1">Top-K</div>
                        <div className="text-emerald-400 font-bold text-lg">
                          {inputTopK}
                        </div>
                      </div>
                    </div>

                    {input.prompt && (
                      <div className="mt-3 bg-slate-900/60 border border-slate-700 rounded p-3">
                        <div className="text-slate-400 mb-1">Prompt</div>
                        <div className="text-slate-100 font-mono text-[11px] whitespace-pre-wrap">
                          {input.prompt}
                        </div>
                      </div>
                    )}
                    {embeddingPreview && embeddingPreview.length > 0 && (
                      <div className="mt-3 bg-slate-900/60 border border-slate-700 rounded p-3">
                        <div className="text-slate-400 mb-2 text-[11px] font-semibold">
                          Query Embedding (Vector Representation)
                        </div>

                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-[10px] text-slate-500">
                            Dimensions: {embeddingDimension ?? embeddingPreview.length}
                          </span>
                          <span className="text-[10px] text-slate-500">•</span>
                          <span className="text-[10px] text-slate-500">
                            First {embeddingPreview.length} values shown
                          </span>
                        </div>

                        {(() => {
                          const maxAbs = Math.max(
                            ...embeddingPreview.map((v) => Math.abs(v))
                          );
                          const safeMaxAbs = Math.max(maxAbs, 1e-9);

                          return (
                            <div className="relative mb-2 overflow-x-auto">
                              <div className="flex items-center gap-[2px] h-20 min-w-[360px] pr-2">
                                {embeddingPreview.map((val, idx) => {
                                  const isPositive = val >= 0;
                                  const norm = Math.abs(val) / safeMaxAbs;
                                  const height = Math.max(4, norm * 40);

                                  return (
                                    <div
                                      key={idx}
                                      className="relative group"
                                      style={{ width: "9.5px", height: "100%" }}
                                    >
                                      <div
                                        className={`absolute left-0 right-0 ${isPositive ? "bottom-1/2" : "top-1/2"
                                          } ${isPositive ? "bg-emerald-500" : "bg-rose-500"}`}
                                        style={{ height: `${height}px` }}
                                      />
                                      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block bg-slate-950 border border-slate-700 rounded px-2 py-1 text-[9px] whitespace-nowrap z-10">
                                        [{idx}]: {val.toFixed(4)}
                                      </div>
                                    </div>
                                  );
                                })}
                              </div>
                            </div>
                          );
                        })()}
                        <div className="flex items-center justify-between mb-1">
                          <div className="text-[10px] text-slate-500">
                            {showFullEmbedding && Array.isArray(fullEmbedding)
                              ? `Showing all ${fullEmbedding.length} dimensions`
                              : `Showing first ${embeddingPreview.length} of ${embeddingDimension} dimensions`}
                          </div>
                        </div>
                        <div className="font-mono text-[9px] text-slate-400 bg-slate-950/50 rounded p-2 overflow-x-auto max-h-32 overflow-y-auto">
                          [
                          {(numericEmbedding ?? []).map((val, idx) => (
                            <span key={idx}>
                              <span className={val >= 0 ? "text-emerald-400" : "text-rose-400"}>
                                {val.toFixed(4)}
                              </span>
                              {idx < (numericEmbedding!.length - 1) && ", "}
                            </span>
                          ))}
                          {showFullEmbedding ? "" : ", ..."}
                          ]
                        </div>
                        {Array.isArray(fullEmbedding) &&
                          fullEmbedding.length === (embeddingDimension ?? fullEmbedding.length) && (
                            <div className="mt-2 flex justify-end">
                              <button
                                type="button"
                                onClick={() => setShowFullEmbedding((prev) => !prev)}
                                className="text-[10px] px-2 py-1 rounded border border-slate-700 text-slate-300 hover:bg-slate-800 transition"
                              >
                                {showFullEmbedding ? "Show first 100 values" : "Show all values"}
                              </button>
                            </div>
                          )}
                        <div className="mt-2 text-[10px] text-slate-500">
                          This vector encodes your question in embedding space. Bars are
                          scaled using the min/max of the first 100 values. Green = positive values,
                          red = negative values. Use the toggle above to inspect all{" "}
                          {embeddingDimension} dimensions for experiments.
                        </div>
                      </div>
                    )}
                  </section>

                  <section className="bg-slate-950/60 border border-slate-800 rounded-lg p-4">
                    <h3 className="text-sm font-bold text-violet-300 mb-3">
                      RETRIEVAL METHOD AGREEMENT
                    </h3>
                    <div className="text-[10px] text-slate-400 mb-3">
                      Shows % overlap in retrieved chunks. Your choice (
                      {METHOD_INFO[chosenMethod].name}) is highlighted.
                    </div>

                    <div className="overflow-x-auto">
                      <table className="w-full text-[11px] border-collapse">
                        <thead>
                          <tr>
                            <th className="px-2 py-2 text-left text-slate-400 font-semibold w-28">
                              Method
                            </th>
                            {METHODS.map((m) => (
                              <th
                                key={m}
                                className="px-2 py-2 text-center text-slate-300 font-semibold"
                              >
                                {METHOD_INFO[m].name.split(" ")[0]}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {METHODS.map((m1) => (
                            <tr key={m1} className="border-t border-slate-800">
                              <td className="px-2 py-2 text-slate-300 font-semibold">
                                {METHOD_INFO[m1].name}
                              </td>
                              {METHODS.map((m2) => {
                                const agreement =
                                  data.retrieval_details?.method_agreement?.[m1]?.[m2] ??
                                  0;

                                const isSelected =
                                  m1 === chosenMethod || m2 === chosenMethod;

                                const pill =
                                  agreement >= 80
                                    ? "bg-emerald-900/50 text-emerald-300"
                                    : agreement >= 60
                                      ? "bg-yellow-900/50 text-yellow-300"
                                      : agreement >= 40
                                        ? "bg-amber-900/50 text-amber-300"
                                        : "bg-rose-900/50 text-rose-300";

                                return (
                                  <td
                                    key={m2}
                                    className={`px-2 py-2 text-center ${isSelected ? "bg-sky-900/20" : ""
                                      }`}
                                  >
                                    <span
                                      className={`inline-block px-2 py-1 rounded text-[10px] font-bold ${pill}`}
                                    >
                                      {agreement}%
                                    </span>
                                  </td>
                                );
                              })}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </section>

                  <section className="bg-slate-950/60 border border-slate-800 rounded-lg p-4">
                    <h3 className="text-sm font-bold text-amber-300 mb-3">
                      RESULTS BY SIMILARITY METHOD
                    </h3>
                    <p className="text-[10px] text-slate-500 mb-2">
                      Each chunk shows its score for the similarity metric. Cosine/Dot/Hybrid: higher = more
                      similar. L1/L2 use negated distances, so higher = closer.
                    </p>
                    <div className="space-y-4">
                      {METHODS.map((method) => {
                        const result = data.results_by_method?.[method];
                        if (!result) return null;

                        const isSelected = method === chosenMethod;
                        const styles = METHOD_STYLES[method];
                        const sources = Array.isArray(result.sources)
                          ? result.sources
                          : [];

                        return (
                          <div
                            key={method}
                            className={`border rounded-lg p-4 ${isSelected
                              ? styles.selectedContainer
                              : "bg-slate-900/40 border-slate-700"
                              }`}
                          >
                            <div className="flex items-center justify-between mb-3">
                              <h4 className="font-bold flex items-center gap-2 flex-wrap">
                                <span className={`text-base ${styles.titleText}`}>
                                  {METHOD_INFO[method].name} Similarity
                                </span>

                                {isSelected && (
                                  <span className="text-[11px] px-2 sm:mt-1 py-0.5 rounded-full bg-slate-800 text-slate-200 font-semibold whitespace-nowrap">
                                    Your choice
                                  </span>
                                )}
                              </h4>

                              <div className="text-[10px] text-slate-400">
                                Top {sources.length} chunks retrieved
                              </div>
                            </div>

                            <div className="mb-3">
                              <div className="text-[10px] text-slate-400 mb-2">
                                Retrieved Chunks:
                              </div>

                              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                                {sources.map((source: any, i: number) => {
                                  const key = `${method}-${i}`;
                                  const isPinned = pinnedKey === key;

                                  const preview =
                                    source.text_preview ??
                                    (typeof source.text === "string"
                                      ? source.text.slice(0, 140)
                                      : "");

                                  const full =
                                    source.full_text ??
                                    source.text ??
                                    source.text_preview ??
                                    "";

                                  return (
                                    <div
                                      key={i}
                                      ref={isPinned ? pinnedContainerRef : undefined}
                                      onClick={() =>
                                        setPinnedKey((prev) => (prev === key ? null : key))
                                      }
                                      className="relative bg-slate-950/40 border border-slate-700 rounded p-2 cursor-pointer"
                                      title="Click to open full chunk"
                                    >
                                      <div className="flex items-center gap-2 mb-1">
                                        <span className={`text-[9px] font-bold ${styles.rankText}`}>
                                          #{source.rank ?? i + 1}
                                        </span>
                                        <span className="text-[9px] text-slate-400 truncate">
                                          {source.doc_name ?? "Unknown source"}
                                        </span>
                                      </div>

                                      <div className="text-[9px] text-slate-300 line-clamp-2 font-mono">
                                        {preview}
                                      </div>

                                      <div className="mt-1 text-[9px] text-slate-500">
                                        {METHOD_INFO[method].name}{" "}
                                        <span className="font-mono text-emerald-300">
                                          {typeof source.score === "number"
                                            ? source.score.toFixed(4)
                                            : "—"}
                                        </span>{" "}
                                        {method === "neg_l1" || method === "neg_l2"
                                          ? "(negated distance; higher = closer)"
                                          : "(higher = more similar)"}
                                      </div>

                                      {full && (
                                        <HoverBubble
                                          text={String(full)}
                                          isPinned={isPinned}
                                          bubbleRef={isPinned ? pinnedBubbleRef : undefined}
                                        />
                                      )}
                                    </div>
                                  );
                                })}
                              </div>

                              {sources.length === 0 && (
                                <div className="text-[10px] text-slate-500">
                                  No chunks returned for this method.
                                </div>
                              )}
                            </div>

                            {renderOperationResults(method)}
                          </div>
                        );
                      })}
                    </div>
                  </section>
                </div>
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </div>
      </Dialog>
    </Transition>
  );
}
