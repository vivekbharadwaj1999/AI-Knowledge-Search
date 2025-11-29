// src/components/AskPanel.tsx
import { useState } from "react";
import { askQuestion } from "../api";

type AskPanelProps = {
  selectedDoc?: string;
};

type Message = {
  id: number;
  question: string;
  answer: string;
  context: string[];
};

export default function AskPanel({ selectedDoc }: AskPanelProps) {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [topK, setTopK] = useState(5);
  const [showContextFor, setShowContextFor] = useState<number | null>(null);

  const canAsk = question.trim().length > 0 && !isLoading;

  const handleAsk = async () => {
    if (!canAsk) return;
    setIsLoading(true);
    try {
      const res = await askQuestion(question.trim(), topK, selectedDoc);
      setMessages((prev) => [
        ...prev,
        {
          id: prev.length ? prev[prev.length - 1].id + 1 : 1,
          question: question.trim(),
          answer: res.answer,
          context: res.context,
        },
      ]);
      setQuestion("");
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          id: prev.length ? prev[prev.length - 1].id + 1 : 1,
          question: question.trim(),
          answer:
            "Sorry, something went wrong while answering this question.",
          context: [],
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  };

  return (
    <div className="h-full min-h-0 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col gap-3">
      <h2 className="font-semibold text-lg mb-1">2. Ask questions</h2>

      <div className="flex items-center justify-between gap-3 text-xs text-slate-400">
        <span>
          {selectedDoc
            ? `Answering for document: ${selectedDoc}`
            : "No document selected - using latest ingested document."}
        </span>
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
      </div>

      <div className="flex gap-2">
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

      <div className="mt-3 flex-1 overflow-y-auto space-y-3 pr-1">
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
            <div className="text-xs text-sky-300 mb-1">You</div>
            <div className="text-sm text-slate-100 mb-2 whitespace-pre-wrap">
              {m.question}
            </div>
            <div className="text-xs text-emerald-300 mb-1">AI</div>
            <div className="text-sm text-slate-100 whitespace-pre-wrap">
              {m.answer}
            </div>
            {m.context.length > 0 && (
              <button
                type="button"
                onClick={() =>
                  setShowContextFor((prev) =>
                    prev === m.id ? null : m.id
                  )
                }
                className="mt-2 text-xs text-sky-400 hover:underline"
              >
                {showContextFor === m.id
                  ? "Hide context"
                  : `Show context (${m.context.length})`}
              </button>
            )}
            {showContextFor === m.id && m.context.length > 0 && (
              <div className="mt-2 text-[11px] text-slate-300 bg-slate-900 border border-slate-700 rounded p-2 space-y-1 max-h-40 overflow-y-auto">
                {m.context.map((c, idx) => (
                  <pre
                    key={idx}
                    className="whitespace-pre-wrap border-b border-slate-800 last:border-none pb-1"
                  >
                    {c}
                  </pre>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
