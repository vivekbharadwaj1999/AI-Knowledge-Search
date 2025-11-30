// src/components/AskPanel.tsx
import { useState, type KeyboardEvent } from "react";
import { askQuestion, generateInsights } from "../api";
import type { AutoInsights } from "../api";
import type { SourceChunk } from "../api";

type AskPanelProps = {
  selectedDoc?: string;
};

type HighlightMode = "ai" | "keywords" | "sentences" | "off";

type Message = {
  id: number;
  question: string;
  answer: string;
  context: string[];
  modelUsed?: string;
  sources?: SourceChunk[];
  insights?: AutoInsights;
  insightsLoading?: boolean;
  insightsError?: string | null;
};

type Comparison = {
  id: number;
  question: string;
  left: {
    model: string;
    answer: string;
    context: string[];
    sources?: SourceChunk[];
  };
  right: {
    model: string;
    answer: string;
    context: string[];
    sources?: SourceChunk[];
  };
};

const MODEL_OPTIONS = [
  { id: "llama-3.1-8b-instant", label: "LLaMA 3.1 8B (fast)" },
  { id: "llama-3.3-70b-versatile", label: "LLaMA 3.3 70B (quality)" },
  { id: "openai/gpt-oss-20b", label: "GPT-OSS 20B (OpenAI OSS)" },
  { id: "openai/gpt-oss-120b", label: "GPT-OSS 120B (OpenAI OSS, large)" },
];

const STOPWORDS = new Set([
  "the",
  "a",
  "an",
  "and",
  "or",
  "of",
  "to",
  "in",
  "for",
  "on",
  "is",
  "are",
  "was",
  "were",
  "it",
  "this",
  "that",
  "with",
  "as",
  "by",
  "at",
]);

function escapeRegExp(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Simple question-based highlighter (used as fallback / baseline).
 */
function highlightChunk(text: string, question: string) {
  const tokens = question
    .toLowerCase()
    .split(/\s+/)
    .map((t) => t.trim())
    .filter((t) => t.length > 2 && !STOPWORDS.has(t));

  if (tokens.length === 0) {
    return text;
  }

  const pattern = tokens.map(escapeRegExp).join("|");
  const regex = new RegExp(`(${pattern})`, "gi");

  const parts = text.split(regex);
  return parts.map((part, idx) =>
    idx % 2 === 1 ? (
      <mark
        key={idx}
        className="bg-yellow-500/20 text-yellow-200 font-semibold rounded px-0.5"
      >
        {part}
      </mark>
    ) : (
      <span key={idx}>{part}</span>
    )
  );
}

function highlightWithKeywords(text: string, phrases: string[]) {
  if (!phrases || phrases.length === 0) return text;

  const cleaned = phrases
    .map((k) => k.toLowerCase().trim())
    .filter((k) => k.length > 2);

  if (cleaned.length === 0) return text;

  const pattern = cleaned
    .sort((a, b) => b.length - a.length)
    .map(escapeRegExp)
    .join("|");

  const regex = new RegExp(`(${pattern})`, "gi");
  const parts = text.split(regex);

  return parts.map((part, idx) =>
    idx % 2 === 1 ? (
      <mark
        key={idx}
        className="bg-yellow-300 text-black font-semibold rounded px-0.5"
      >
        {part}
      </mark>
    ) : (
      <span key={idx}>{part}</span>
    )
  );
}

function renderHighlightedChunk(
  text: string,
  message: Message,
  mode: HighlightMode
) {
  if (mode === "off") return text;

  const insights = message.insights;

  // --- Pure keyword mode (no sentence awareness) ---
  if (mode === "keywords") {
    const phrases =
      insights?.keywords && insights.keywords.length > 0
        ? insights.keywords
        : message.question.split(/\s+/);
    return highlightWithKeywords(text, phrases);
  }

  // --- Sentence-based modes (AI + sentences) ---
  const sentences = text.split(/(?<=[.!?])\s+/);

  // 1) Build scores per sentence (0–5)
  let scores: number[] = new Array(sentences.length).fill(0);

  if (insights?.sentence_importance && insights.sentence_importance.length > 0) {
    // Use LLM-provided sentence_importance if possible
    for (let i = 0; i < sentences.length; i++) {
      const s = sentences[i];
      const lower = s.toLowerCase();
      let maxScore = 0;

      for (const item of insights.sentence_importance) {
        const sn = item.sentence.toLowerCase().trim();
        if (!sn) continue;

        // loose matching: one contains the other
        if (lower.includes(sn) || sn.includes(lower)) {
          if (item.score > maxScore) maxScore = item.score;
        }
      }
      scores[i] = maxScore; // 0–5
    }
  }

  // If all scores are still 0 (no match / weird sentences), fall back to keyword-based scoring
  if (scores.every((s) => s === 0)) {
    const tokens = (insights?.keywords || message.question.split(/\s+/))
      .map((t) => t.toLowerCase().trim())
      .filter((t) => t.length > 2 && !STOPWORDS.has(t));

    scores = sentences.map((s) => {
      const lower = s.toLowerCase();
      let sc = 0;
      tokens.forEach((t) => {
        if (lower.includes(t)) sc++;
      });
      return sc;
    });
  }

  const tokensForPhrases =
    insights?.keywords && insights.keywords.length > 0
      ? insights.keywords
      : message.question.split(/\s+/);

  // 2) Pick TOP ~30% sentences (min 1, max 5)
  const scoredIndices = scores
    .map((s, i) => ({ score: s, index: i }))
    .filter((x) => x.score > 0);

  // If STILL nothing scored, just fall back to pure keyword highlighting
  if (scoredIndices.length === 0) {
    return highlightWithKeywords(text, tokensForPhrases);
  }

  // Sort by descending score
  scoredIndices.sort((a, b) => b.score - a.score);

  const maxSentences = Math.min(
    5,
    Math.max(1, Math.round(sentences.length * 0.3))
  );

  const keep = new Set<number>();
  for (const { index } of scoredIndices.slice(0, maxSentences)) {
    keep.add(index);
  }

  const maxScore = Math.max(...scores);

  // 3) Render sentences with bars + inner phrase highlights
  return sentences.map((sentence, idx) => {
    const content = sentence + (idx < sentences.length - 1 ? " " : "");
    const sc = scores[idx];

    // Not selected as one of the top 30% → plain text
    if (!keep.has(idx) || sc <= 0) {
      return <span key={idx}>{content}</span>;
    }

    // SENTENCES mode: same style for all important sentences
    if (mode === "sentences") {
      return (
        <span
          key={idx}
          className="bg-yellow-300/20 border-l-2 border-yellow-400/80 pl-1 pr-0.5 rounded-sm"
        >
          {content}
        </span>
      );
    }

    // AI mode: color-coded intensity based on relative score
    let wrapperClass = "";
    if (maxScore <= 0) {
      wrapperClass =
        "bg-yellow-300/10 border-l border-yellow-400/40 pl-1 pr-0.5 rounded-sm";
    } else {
      const norm = sc / maxScore;
      if (norm >= 0.66) {
        // 🔥 most important
        wrapperClass =
          "bg-amber-400/25 border-l-4 border-amber-400 pl-[6px] pr-0.5 rounded-sm";
      } else if (norm >= 0.33) {
        // ⭐ medium
        wrapperClass =
          "bg-yellow-300/20 border-l-2 border-yellow-400/80 pl-1 pr-0.5 rounded-sm";
      } else {
        // ✨ light
        wrapperClass =
          "bg-yellow-200/10 border-b border-dotted border-yellow-400/70";
      }
    }

    return (
      <span key={idx} className={wrapperClass}>
        {highlightWithKeywords(content, tokensForPhrases)}
      </span>
    );
  });
}


export default function AskPanel({ selectedDoc }: AskPanelProps) {
  // --- normal Q&A state ---
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [topK, setTopK] = useState(5);
  const [showContextFor, setShowContextFor] = useState<number | null>(null);
  const [modelId, setModelId] = useState<string>("llama-3.1-8b-instant");

  // --- comparison state ---
  const [compareQuestion, setCompareQuestion] = useState("");
  const [modelLeft, setModelLeft] = useState("llama-3.1-8b-instant");
  const [modelRight, setModelRight] = useState("llama-3.3-70b-versatile");
  const [comparisons, setComparisons] = useState<Comparison[]>([]);
  const [isCompareLoading, setIsCompareLoading] = useState(false);
  const [highlightMode, setHighlightMode] = useState<HighlightMode>("ai");
  const [showCompareContextId, setShowCompareContextId] = useState<number | null>(
    null
  );

  const canAsk = question.trim().length > 0 && !isLoading;
  const canCompare =
    compareQuestion.trim().length > 0 &&
    !isCompareLoading &&
    modelLeft !== modelRight;

  // ---------- normal ask ----------
  const handleAsk = async () => {
    if (!canAsk) return;
    setIsLoading(true);
    try {
      const trimmed = question.trim();
      const res = await askQuestion(trimmed, topK, selectedDoc, modelId);

      setMessages((prev) => [
        ...prev,
        {
          id: prev.length ? prev[prev.length - 1].id + 1 : 1,
          question: trimmed,
          answer: res.answer,
          context: res.context,
          modelUsed: res.model_used || modelId,
          sources: res.sources || [],
        },
      ]);
      setQuestion("");
    } catch (err) {
      console.error(err);
      const trimmed = question.trim();
      setMessages((prev) => [
        ...prev,
        {
          id: prev.length ? prev[prev.length - 1].id + 1 : 1,
          question: trimmed,
          answer: "Sorry, something went wrong while answering this question.",
          context: [],
          modelUsed: modelId,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  };

  // ---------- auto insights ----------
  const triggerInsights = async (id: number) => {
    setMessages((prev) =>
      prev.map((m) =>
        m.id === id
          ? { ...m, insightsLoading: true, insightsError: null }
          : m
      )
    );

    const target = messages.find((m) => m.id === id);
    if (!target) return;

    try {
      const contextForInsights =
        target.sources && target.sources.length > 0
          ? target.sources.map((s) => `[Source: ${s.doc_name}] ${s.text}`)
          : target.context;

      const insights = await generateInsights({
        question: target.question,
        answer: target.answer,
        context: contextForInsights,
        model: target.modelUsed || modelId,
      });

      setMessages((prev) =>
        prev.map((m) =>
          m.id === id ? { ...m, insights, insightsLoading: false } : m
        )
      );
    } catch (err) {
      console.error(err);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === id
            ? {
              ...m,
              insightsLoading: false,
              insightsError: "Failed to generate insights. Try again.",
            }
            : m
        )
      );
    }
  };

  // ---------- model comparison ----------
  const handleCompare = async () => {
    if (!canCompare) return;
    setIsCompareLoading(true);

    try {
      const trimmed = compareQuestion.trim();

      const [leftRes, rightRes] = await Promise.all([
        askQuestion(trimmed, topK, selectedDoc, modelLeft),
        askQuestion(trimmed, topK, selectedDoc, modelRight),
      ]);

      setComparisons((prev) => [
        ...prev,
        {
          id: prev.length ? prev[prev.length - 1].id + 1 : 1,
          question: trimmed,
          left: {
            model: leftRes.model_used || modelLeft,
            answer: leftRes.answer,
            context: leftRes.context,
            sources: leftRes.sources || [],
          },
          right: {
            model: rightRes.model_used || modelRight,
            answer: rightRes.answer,
            context: rightRes.context,
            sources: rightRes.sources || [],
          },
        },
      ]);
    } catch (err) {
      console.error(err);
    } finally {
      setIsCompareLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Shared header (document + controls) */}
      <h2 className="font-semibold text-lg mb-1">2. Ask & compare</h2>

      <div className="mb-2 text-xs text-slate-400 space-y-1">
        <div>
          {selectedDoc
            ? `Answering for document: ${selectedDoc}`
            : "Searching across all indexed documents."}
        </div>
        <div className="text-[11px] text-slate-500">
          Context Highlighting + Model Picker + Auto Insights + Side-by-side
          comparison.
        </div>

        <div className="mt-1 flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-1">
            <span>Top&nbsp;K:</span>
            <input
              type="number"
              min={1}
              max={20}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value) || 5)}
              className="w-14 bg-slate-800 border border-slate-700 rounded px-1 py-0.5 text-xs text-slate-100"
            />
          </label>

          <label className="flex items-center gap-1">
            <span>Default model:</span>
            <select
              className="bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-slate-100"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
            >
              {MODEL_OPTIONS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>

      {/* TWO COLUMNS: left = Ask questions, right = Model comparison */}
      <div className="grid gap-4 md:grid-cols-2 flex-1 min-h-0">
        {/* LEFT COLUMN: Ask questions (chat) */}
        <div className="flex flex-col min-h-0">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-semibold text-md">Ask questions</h3>
          </div>

          <div className="flex gap-2 mb-2">
            <textarea
              className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm
                         focus:outline-none focus:ring-2 focus:ring-sky-500"
              rows={2}
              placeholder="Ask something about the selected document..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
            />
            <button
              onClick={handleAsk}
              disabled={!canAsk}
              className="h-[40px] self-end inline-flex items-center justify-center rounded-lg px-4 text-sm font-medium
                         bg-sky-600 hover:bg-sky-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? "Thinking..." : "Ask"}
            </button>
          </div>

          <div className="flex-1 overflow-y-auto space-y-3 pr-1">
            {messages.length === 0 && (
              <p className="text-sm text-slate-400">
                Your questions and answers will appear here as a chat.
              </p>
            )}

            {messages.map((m) => (
              <div
                key={m.id}
                className="border border-slate-800 rounded-lg p-3 bg-slate-950/60"
              >
                <div className="flex items-center justify-between mb-1">
                  <div className="text-xs text-sky-300">You</div>
                  {m.modelUsed && (
                    <div className="text-[11px] text-slate-400">
                      Model:{" "}
                      <span className="font-mono text-slate-200">
                        {m.modelUsed}
                      </span>
                    </div>
                  )}
                </div>

                <div className="text-sm text-slate-100 mb-2 whitespace-pre-wrap">
                  {m.question}
                </div>

                <div className="text-xs text-emerald-300 mb-1">AI</div>
                <div className="text-sm text-slate-100 whitespace-pre-wrap">
                  {m.answer}
                </div>
                {/* Auto Insights FIRST */}
                <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
                  <button
                    type="button"
                    onClick={() => triggerInsights(m.id)}
                    className="inline-flex items-center gap-1 rounded-md px-2 py-1 bg-violet-600 hover:bg-violet-500 text-white disabled:opacity-50"
                    disabled={m.insightsLoading}
                  >
                    <span>✨ Auto insights</span>
                    {m.insightsLoading && <span>(thinking...)</span>}
                  </button>

                  {m.insightsError && (
                    <span className="text-red-400">{m.insightsError}</span>
                  )}
                </div>

                {/* Insights panel */}
                {m.insights && (
                  <div className="mt-2 border border-violet-700 bg-violet-950/40 rounded p-2 text-[11px] text-slate-100 space-y-1">
                    <div>
                      <div className="font-semibold text-violet-200 mb-0.5">
                        Summary
                      </div>
                      <div className="whitespace-pre-wrap">
                        {m.insights.summary}
                      </div>
                    </div>

                    {m.insights.key_points?.length > 0 && (
                      <div>
                        <div className="font-semibold text-violet-200 mb-0.5">
                          Key points
                        </div>
                        <ul className="list-disc list-inside space-y-0.5">
                          {m.insights.key_points.map((p, i) => (
                            <li key={i}>{p}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {m.insights.entities?.length > 0 && (
                      <div>
                        <div className="font-semibold text-violet-200 mb-0.5">
                          Entities
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {m.insights.entities.map((e, i) => (
                            <span
                              key={i}
                              className="px-2 py-0.5 rounded-full bg-slate-900/70 border border-slate-700"
                            >
                              {e}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {m.insights.keywords?.length > 0 && (
                      <div>
                        <div className="font-semibold text-violet-200 mb-0.5">
                          Keywords
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {m.insights.keywords.map((kw, i) => (
                            <span
                              key={i}
                              className="px-2 py-0.5 rounded-full bg-slate-900/70 border border-slate-700"
                            >
                              {kw}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {m.insights.suggested_questions?.length > 0 && (
                      <div>
                        <div className="font-semibold text-violet-200 mb-0.5">
                          Suggested questions
                        </div>
                        <ul className="list-disc list-inside space-y-0.5">
                          {m.insights.suggested_questions.map((q, i) => (
                            <li key={i}>{q}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {m.insights.mindmap && (
                      <div>
                        <div className="font-semibold text-violet-200 mb-0.5">
                          Mindmap (text)
                        </div>
                        <pre className="whitespace-pre-wrap font-mono">
                          {m.insights.mindmap}
                        </pre>
                      </div>
                    )}

                    <div className="flex flex-wrap gap-3">
                      {m.insights.reading_difficulty && (
                        <div>
                          <span className="font-semibold text-violet-200">
                            Reading difficulty:
                          </span>{" "}
                          <span>{m.insights.reading_difficulty}</span>
                        </div>
                      )}
                      {m.insights.sentiment && (
                        <div>
                          <span className="font-semibold text-violet-200">
                            Sentiment:
                          </span>{" "}
                          <span>{m.insights.sentiment}</span>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Context viewer + HIGHLIGHT MENU (disabled until insights exist) */}
                {m.context.length > 0 && (
                  <div className="mt-2 text-xs">
                    <button
                      type="button"
                      disabled={!m.insights}
                      onClick={() =>
                        m.insights &&
                        setShowContextFor((prev) => (prev === m.id ? null : m.id))
                      }
                      className={`inline-flex items-center gap-1 rounded-md px-2 py-1 border ${m.insights
                        ? "bg-slate-800 hover:bg-slate-700 text-sky-200 border-slate-600"
                        : "bg-slate-900 text-slate-500 border-slate-700 cursor-not-allowed opacity-60"
                        }`}
                    >
                      {showContextFor === m.id
                        ? "Hide highlighted context"
                        : "Show highlighted context"}
                    </button>

                    {!m.insights && (
                      <span className="ml-2 text-[10px] text-slate-500">
                        Run Auto insights first to enable AI highlighting.
                      </span>
                    )}

                    {showContextFor === m.id && m.context.length > 0 && (
                      <div className="mt-2 text-[11px] text-slate-300 bg-slate-900/80 border border-slate-700 rounded max-h-80 overflow-y-auto relative">
                        {/* Sticky header inside the scrollable context */}
                        <div className="sticky top-0 z-10 bg-slate-900/95 border-b border-slate-700 px-2 py-1 flex items-center justify-between">
                          <span className="text-[10px] text-slate-400">
                            Highlighted context
                          </span>
                          <div className="flex items-center gap-1 text-[10px]">
                            <span className="mr-1 text-slate-500">Highlight Mode:</span>
                            {(
                              [
                                ["ai", "AI"],
                                ["keywords", "Keywords"],
                                ["sentences", "Sentences"],
                                ["off", "Off"],
                              ] as [HighlightMode, string][]
                            ).map(([mode, label]) => (
                              <button
                                key={mode}
                                type="button"
                                onClick={() => setHighlightMode(mode)}
                                className={
                                  "px-1.5 py-0.5 rounded-full border text-[10px]" +
                                  (highlightMode === mode
                                    ? " bg-sky-600 text-white border-sky-500"
                                    : " bg-slate-900 text-slate-300 border-slate-700 hover:bg-slate-800")
                                }
                              >
                                {label}
                              </button>
                            ))}
                          </div>
                        </div>

                        {/* Actual chunks (scroll under the sticky header) */}
                        {/* Actual chunks (scroll under the sticky header) */}
                        <div className="px-2 py-1 space-y-1">
                          {m.sources && m.sources.length > 0 ? (
                            <>
                              <div className="text-[10px] text-slate-400 mb-1">
                                Used documents:{" "}
                                {Array.from(new Set(m.sources.map((s) => s.doc_name))).join(", ")}
                              </div>

                              {Array.from(
                                Object.entries(
                                  m.sources.reduce((acc, s) => {
                                    if (!acc[s.doc_name]) acc[s.doc_name] = [];
                                    acc[s.doc_name].push(s.text);
                                    return acc;
                                  }, {} as Record<string, string[]>)
                                )
                              ).map(([docName, chunks]) => (
                                <div key={docName} className="mb-1">
                                  <div className="text-[11px] text-sky-300 mb-0.5">{docName}</div>
                                  {chunks.map((c, idx) => (
                                    <pre
                                      key={docName + idx}
                                      className="whitespace-pre-wrap border-b border-slate-800 last:border-none pb-1"
                                    >
                                      {renderHighlightedChunk(c, m, highlightMode)}
                                    </pre>
                                  ))}
                                </div>
                              ))}
                            </>
                          ) : (
                            m.context.map((c, idx) => (
                              <pre
                                key={idx}
                                className="whitespace-pre-wrap border-b border-slate-800 last:border-none pb-1"
                              >
                                {renderHighlightedChunk(c, m, highlightMode)}
                              </pre>
                            ))
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* RIGHT COLUMN: Model comparison */}
        <div className="flex flex-col min-h-0">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-semibold text-md">Model comparison</h3>
            <span className="text-[11px] text-slate-500">
              Side-by-side answers for the same question.
            </span>
          </div>

          <div className="flex flex-col gap-2 mb-2">
            <textarea
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm
                         focus:outline-none focus:ring-2 focus:ring-violet-500"
              rows={2}
              placeholder="Enter a question to compare two models..."
              value={compareQuestion}
              onChange={(e) => setCompareQuestion(e.target.value)}
            />

            <div className="flex flex-col sm:flex-row gap-2">
              <div className="flex-1">
                <label className="block text-[11px] text-slate-400 mb-0.5">
                  Model A
                </label>
                <select
                  className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-slate-100"
                  value={modelLeft}
                  onChange={(e) => setModelLeft(e.target.value)}
                >
                  {MODEL_OPTIONS.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex-1">
                <label className="block text-[11px] text-slate-400 mb-0.5">
                  Model B
                </label>
                <select
                  className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-slate-100"
                  value={modelRight}
                  onChange={(e) => setModelRight(e.target.value)}
                >
                  {MODEL_OPTIONS.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.label}
                    </option>
                  ))}
                </select>
              </div>

              <button
                onClick={handleCompare}
                disabled={!canCompare}
                className="sm:self-end inline-flex items-center justify-center rounded-lg px-4 py-2 text-sm font-medium
                           bg-violet-600 hover:bg-violet-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isCompareLoading ? "Comparing..." : "Compare"}
              </button>
            </div>

            {modelLeft === modelRight && (
              <div className="text-[11px] text-amber-400">
                Pick two different models to compare.
              </div>
            )}
          </div>

          <div className="flex-1 overflow-y-auto space-y-4 pr-1">
            {comparisons.length === 0 && (
              <p className="text-sm text-slate-500">
                No comparisons yet. Ask your first side-by-side question above.
              </p>
            )}

            {comparisons.map((cmp) => (
              <div
                key={cmp.id}
                className="border border-slate-800 rounded-lg p-3 bg-slate-950/60"
              >
                <div className="text-xs text-sky-300 mb-1">Question</div>
                <div className="text-sm text-slate-100 mb-2 whitespace-pre-wrap">
                  {cmp.question}
                </div>

                <div className="grid gap-3 lg:grid-cols-2">
                  {/* LEFT */}
                  <div className="border border-slate-800 rounded-lg p-2 bg-slate-950/70">
                    <div className="flex items-center justify-between mb-1">
                      <div className="text-xs text-slate-400">Model A</div>
                      <div className="text-[11px] font-mono text-emerald-300">
                        {cmp.left.model}
                      </div>
                    </div>
                    <div className="text-sm text-slate-100 whitespace-pre-wrap mb-2">
                      {cmp.left.answer}
                    </div>

                    {cmp.left.context.length > 0 && (
                      <button
                        type="button"
                        onClick={() =>
                          setShowCompareContextId((prev) =>
                            prev === cmp.id ? null : cmp.id
                          )
                        }
                        className="text-xs text-sky-400 hover:underline"
                      >
                        {showCompareContextId === cmp.id
                          ? "Hide highlighted context"
                          : "Show highlighted context"}
                      </button>
                    )}

                    {showCompareContextId === cmp.id &&
                      (cmp.left.sources?.length || cmp.left.context.length) > 0 && (
                        <div className="mt-2 text-[11px] text-slate-300 border border-slate-700 rounded p-2 space-y-1 max-h-40 overflow-y-auto">
                          {cmp.left.sources && cmp.left.sources.length > 0 ? (
                            <>
                              <div className="mb-1 text-[10px] text-slate-400">
                                Used documents:{" "}
                                {Array.from(
                                  new Set(cmp.left.sources.map((s) => s.doc_name))
                                ).join(", ")}
                              </div>
                              {Array.from(
                                Object.entries(
                                  cmp.left.sources.reduce((acc, s) => {
                                    if (!acc[s.doc_name]) acc[s.doc_name] = [];
                                    acc[s.doc_name].push(s.text);
                                    return acc;
                                  }, {} as Record<string, string[]>)
                                )
                              ).map(([docName, chunks]) => (
                                <div key={docName} className="mb-1">
                                  <div className="text-[11px] text-sky-300 mb-0.5">{docName}</div>
                                  {chunks.map((c, idx) => (
                                    <pre
                                      key={docName + idx}
                                      className="whitespace-pre-wrap border-b border-slate-800 last:border-none pb-1"
                                    >
                                      {highlightChunk(c, cmp.question)}
                                    </pre>
                                  ))}
                                </div>
                              ))}
                            </>
                          ) : (
                            cmp.left.context.map((c, idx) => (
                              <pre
                                key={idx}
                                className="whitespace-pre-wrap border-b border-slate-800 last:border-none pb-1"
                              >
                                {highlightChunk(c, cmp.question)}
                              </pre>
                            ))
                          )}
                        </div>
                      )}
                  </div>

                  {/* RIGHT */}
                  <div className="border border-slate-800 rounded-lg p-2 bg-slate-950/70">
                    <div className="flex items-center justify-between mb-1">
                      <div className="text-xs text-slate-400">Model B</div>
                      <div className="text-[11px] font-mono text-emerald-300">
                        {cmp.right.model}
                      </div>
                    </div>
                    <div className="text-sm text-slate-100 whitespace-pre-wrap mb-2">
                      {cmp.right.answer}
                    </div>

                    {cmp.right.context.length > 0 && (
                      <button
                        type="button"
                        onClick={() =>
                          setShowCompareContextId((prev) =>
                            prev === cmp.id ? null : cmp.id
                          )
                        }
                        className="text-xs text-sky-400 hover:underline"
                      >
                        {showCompareContextId === cmp.id
                          ? "Hide highlighted context"
                          : "Show highlighted context"}
                      </button>
                    )}

                    {showCompareContextId === cmp.id &&
                      (cmp.right.sources?.length || cmp.right.context.length) > 0 && (
                        <div className="mt-2 text-[11px] text-slate-300 bg-slate-900/80 border border-slate-700 rounded p-2 space-y-1 max-h-40 overflow-y-auto">
                          {cmp.right.sources && cmp.right.sources.length > 0 ? (
                            <>
                              <div className="mb-1 text-[10px] text-slate-400">
                                Used documents:{" "}
                                {Array.from(
                                  new Set(cmp.right.sources.map((s) => s.doc_name))
                                ).join(", ")}
                              </div>
                              {Array.from(
                                Object.entries(
                                  cmp.right.sources.reduce((acc, s) => {
                                    if (!acc[s.doc_name]) acc[s.doc_name] = [];
                                    acc[s.doc_name].push(s.text);
                                    return acc;
                                  }, {} as Record<string, string[]>)
                                )
                              ).map(([docName, chunks]) => (
                                <div key={docName} className="mb-1">
                                  <div className="text-[11px] text-sky-300 mb-0.5">
                                    {docName}
                                  </div>
                                  {chunks.map((c, idx) => (
                                    <pre
                                      key={docName + idx}
                                      className="whitespace-pre-wrap border-b border-slate-800 last:border-none pb-1"
                                    >
                                      {highlightChunk(c, cmp.question)}
                                    </pre>
                                  ))}
                                </div>
                              ))}
                            </>
                          ) : (
                            cmp.right.context.map((c, idx) => (
                              <pre
                                key={idx}
                                className="whitespace-pre-wrap border-b border-slate-800 last:border-none pb-1"
                              >
                                {highlightChunk(c, cmp.question)}
                              </pre>
                            ))
                          )}
                        </div>
                      )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
