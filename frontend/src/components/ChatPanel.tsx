import { useState } from "react";
import { askQuestion } from "../api";

export function ChatPanel() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<string | null>(null);
  const [context, setContext] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onAsk = async () => {
    if (!question.trim()) {
      setError("Please type a question first.");
      return;
    }
    try {
      setError(null);
      setLoading(true);
      const res = await askQuestion(question.trim(), 5);
      setAnswer(res.answer);
      setContext(res.context || []);
    } catch (err) {
      console.error(err);
      setError("Error while asking the question.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4 p-4 rounded-xl bg-slate-900/70 border border-slate-700">
      <h2 className="text-xl font-semibold">2. Ask a question</h2>

      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        rows={3}
        className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm
                   focus:outline-none focus:ring-2 focus:ring-sky-500"
        placeholder='E.g. "What is this document mostly about?" or "Summarize the key points."'
      />

      <button
        onClick={onAsk}
        disabled={loading}
        className="px-4 py-2 rounded-lg bg-sky-600 hover:bg-sky-500 disabled:opacity-60"
      >
        {loading ? "Thinking..." : "Ask"}
      </button>

      {error && <p className="text-sm text-red-400">{error}</p>}

      {answer && (
        <div className="space-y-2">
          <h3 className="font-semibold text-lg">Answer</h3>
          <p className="whitespace-pre-wrap text-sm text-slate-100">
            {answer}
          </p>
        </div>
      )}

      {context.length > 0 && (
        <details className="mt-2">
          <summary className="cursor-pointer text-sm text-slate-300">
            Show context chunks ({context.length})
          </summary>
          <div className="mt-2 space-y-2 text-xs text-slate-300 max-h-64 overflow-y-auto">
            {context.map((chunk, idx) => (
              <div key={idx} className="p-2 rounded bg-slate-800/80">
                <div className="font-mono text-[10px] text-slate-400">
                  Chunk {idx + 1}
                </div>
                <div>{chunk}</div>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
