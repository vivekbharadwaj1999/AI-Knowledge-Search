import type { KeyboardEvent, RefObject } from "react";
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

export type AskControlsProps = {
  selectedDoc?: string;
  question: string;
  setQuestion: (v: string) => void;
  topK: number;
  setTopK: (v: number) => void;
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
};

export default function AskControls({
  selectedDoc,
  question,
  setQuestion,
  topK,
  setTopK,
  modelId,
  setModelId,
  canAsk,
  isLoading,
  onAsk,
  onQuestionKeyDown,
  askInputRef,
}: AskControlsProps) {
  return (
    <div className="flex flex-col gap-4">
      <div className="text-xs text-slate-400 space-y-1">
        <div className="mb-4 font-bold">
          {selectedDoc
            ? `Answering for document: ${selectedDoc}`
            : "Searching across all indexed documents."}
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

      <div className="space-y-2">
        <h3 className="font-semibold text-sm sm:text-base">Ask questions</h3>
        <div className="flex gap-2">
          <textarea
            ref={askInputRef}
            className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm
                       focus:outline-none focus:ring-2 focus:ring-sky-500"
            rows={2}
            placeholder="Ask something about the selected document..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={onQuestionKeyDown}
          />
          <button
            onClick={onAsk}
            disabled={!canAsk}
            className="h-[40px] self-end inline-flex items-center justify-center rounded-lg px-4 text-sm font-medium
                       bg-sky-600 hover:bg-sky-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? "Thinking..." : "Ask"}
          </button>
        </div>
      </div>
    </div>
  );
}
