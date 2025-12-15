import type {
  KeyboardEvent,
  RefObject,
} from "react";
import type { AutoInsights, SourceChunk } from "../api";

export type HighlightMode = "ai" | "keywords" | "sentences" | "off";

export type Message = {
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

export type Comparison = {
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

export const MODEL_OPTIONS = [
  {
    id: "llama-3.1-8b-instant",
    label: "Llama 3.1 8B Instant (fast, lightweight)",
  },
  {
    id: "llama-3.3-70b-versatile",
    label: "Llama 3.3 70B Versatile (high quality general model)",
  },
  {
    id: "meta-llama/llama-4-scout-17b-16e-instruct",
    label: "Llama 4 Scout 17B 16E (efficient, balanced)",
  },
  {
    id: "meta-llama/llama-4-maverick-17b-128e-instruct",
    label: "Llama 4 Maverick 17B 128E (strong reasoning)",
  },
  {
    id: "openai/gpt-oss-20b",
    label: "GPT OSS 20B (reliable all round model)",
  },
  {
    id: "openai/gpt-oss-120b",
    label: "GPT OSS 120B (high capacity model)",
  },
  {
    id: "meta-llama/llama-guard-4-12b",
    label: "Llama Guard 4 12B (safety check model against disallowed content)",
  },
  {
    id: "openai/gpt-oss-safeguard-20b",
    label: "GPT OSS Safeguard 20B (safety check model against disallowed content)",
  },
  {
    id: "moonshotai/kimi-k2-instruct-0905",
    label: "Kimi K2 Instruct 0905 (large context)",
  },
  {
    id: "qwen/qwen3-32b",
    label: "Qwen3 32B (multilingual & strong general model)",
  },
] as const;

type SimilarityMetric = "cosine" | "dot" | "neg_l2" | "neg_l1" | "hybrid";

const SIMILARITY_LABELS: Record<SimilarityMetric, string> = {
  cosine: "cosine similarity",
  dot: "dot product similarity",
  neg_l2: "negative Euclidean distance (L2)",
  neg_l1: "negative Manhattan distance (L1)",
  hybrid: "hybrid (cosine + Jaccard) similarity",
};

export type AskControlsProps = {
  selectedDoc?: string;
  question: string;
  setQuestion: (v: string) => void;
  topK: number;
  modelId: string;
  setModelId: (id: string) => void;
  canAsk: boolean;
  isLoading: boolean;
  onAsk: () => void;
  onQuestionKeyDown: (e: KeyboardEvent<HTMLTextAreaElement>) => void;
  compareQuestion: string;
  setCompareQuestion: (v: string) => void;
  modelLeft: string;
  setModelLeft: (v: string) => void;
  modelRight: string;
  setModelRight: (v: string) => void;
  canCompare: boolean;
  isCompareLoading: boolean;
  askInputRef?: RefObject<HTMLTextAreaElement | null>;
  similarityMetric: SimilarityMetric;
};

export default function AskControls(props: AskControlsProps) {
  const {
    selectedDoc,
    question,
    setQuestion,
    topK,
    modelId,
    setModelId,
    canAsk,
    isLoading,
    onAsk,
    onQuestionKeyDown,
    askInputRef,
    similarityMetric,
  } = props;

  const scopeText = selectedDoc
    ? `Answering for document: ${selectedDoc}`
    : "Searching across all indexed documents";

  return (
    <div className="flex flex-col gap-4">
      <div className="text-xs text-slate-400 space-y-1">
        <p className="mb-2">
          <span className="font-semibold">{scopeText}</span>, using{" "}
          <span className="font-semibold">Top K = {topK}</span>, and{" "}
          <span className="font-semibold">
            {SIMILARITY_LABELS[similarityMetric]} function
          </span>{" "}
          (configured in section 2).
        </p>

        <div className="mt-1 pt-3 flex flex-wrap items-center gap-3">
          <label className="flex flex-col items-start gap-1 sm:flex-row sm:items-center">
            <span>Choose model:</span>
            <select
              className="w-full sm:w-auto max-w-full bg-slate-800 border border-slate-700 rounded
                       px-2 py-1 text-[11px] sm:text-xs text-slate-100"
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

      <div className="space-y-2">
        <label className="block text-[11px] text-slate-400 mb-0.5">
          Ask Question
        </label>
        <div className="flex flex-col gap-2">
          <textarea
            ref={askInputRef}
            className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm
                 focus:outline-none focus:ring-2 focus:ring-sky-500"
            rows={2}
            placeholder="Ask something about the selected document..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={onQuestionKeyDown}
          />
          <div className="flex justify-end">
            <button
              onClick={onAsk}
              disabled={!canAsk}
              className="h-[40px] inline-flex items-center justify-center rounded-lg px-4 text-sm font-medium
                         bg-sky-600 hover:bg-sky-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? "Thinking..." : "Ask"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
