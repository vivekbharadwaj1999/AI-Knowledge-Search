import { useState, useRef, useEffect } from "react";
import { runBatchEvaluation } from "../api";

interface BatchEvaluationPanelProps {
  documents: string[];
  onClose: () => void;
}

const MODEL_OPTIONS = [
  { value: "llama-3.1-8b-instant", label: "Llama 3.1 8B Instant (fast, lightweight)" },
  { value: "llama-3.3-70b-versatile", label: "Llama 3.3 70B Versatile (high quality general model)" },
  { value: "llama-3.3-70b-specdec", label: "Llama 4 Scout 17B 16E (efficient, balanced)" },
  { value: "mixtral-8x7b-32768", label: "Llama 4 Maverick 17B 128E (strong reasoning)" },
  { value: "gemma2-9b-it", label: "GPT OSS 20B (reliable all round model)" },
  { value: "llama-3.2-11b-vision-preview", label: "GPT OSS 120B (high capacity model)" },
  { value: "llama-3.2-3b-preview", label: "Kimi K2 Instruct 0905 (large context)" },
  { value: "llama-3.2-1b-preview", label: "Qwen3 32B (multilingual & strong general model)" },
];

export default function BatchEvaluationPanel({ documents, onClose }: BatchEvaluationPanelProps) {
  const [questions, setQuestions] = useState<string[]>([]);
  const [questionInput, setQuestionInput] = useState("");
  const [results, setResults] = useState<any>(null);
  const [isRunning, setIsRunning] = useState(false);
  const resultsRef = useRef<HTMLDivElement>(null);
  const [selectedMethods, setSelectedMethods] = useState<string[]>(["cosine", "hybrid"]);
  const [selectedTopK, setSelectedTopK] = useState<number[]>([5, 7]);
  const [selectedDoc, setSelectedDoc] = useState<string>("");
  const [normalizeVectors, setNormalizeVectors] = useState(true);
  const [includeFaithfulness, setIncludeFaithfulness] = useState(true);
  const [enableAsk, setEnableAsk] = useState(false);
  const [enableCompare, setEnableCompare] = useState(false);
  const [enableCritique, setEnableCritique] = useState(false);
  const [askModel, setAskModel] = useState("llama-3.1-8b-instant");
  const [compareModelLeft, setCompareModelLeft] = useState("llama-3.1-8b-instant");
  const [compareModelRight, setCompareModelRight] = useState("llama-3.3-70b-versatile");
  const [critiqueAnswerModel, setCritiqueAnswerModel] = useState("llama-3.1-8b-instant");
  const [critiqueCriticModel, setCritiqueCriticModel] = useState("llama-3.3-70b-versatile");
  const [critiqueSelfCorrect, setCritiqueSelfCorrect] = useState(false);

  const MAX_QUESTIONS = 20;

  useEffect(() => {
    if (results && resultsRef.current) {
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    }
  }, [results]);

  const handleAddQuestion = () => {
    if (questionInput.trim() && questions.length < MAX_QUESTIONS) {
      setQuestions([...questions, questionInput.trim()]);
      setQuestionInput("");
    }
  };

  const canRunBatch = documents.length > 0 &&
    questions.length > 0 &&
    selectedMethods.length > 0 &&
    selectedTopK.length > 0 &&
    (enableAsk || enableCompare || enableCritique);

  const handleRunBatch = async () => {
    if (!canRunBatch) return;

    setIsRunning(true);
    try {
      const operations = [];
      if (enableAsk) {
        operations.push({
          type: "ask",
          model: askModel
        });
      }
      if (enableCompare) {
        operations.push({
          type: "compare",
          models: [compareModelLeft, compareModelRight]
        });
      }
      if (enableCritique) {
        operations.push({
          type: "critique",
          answer_model: critiqueAnswerModel,
          critic_model: critiqueCriticModel,
          self_correct: critiqueSelfCorrect
        });
      }

      const payload = {
        questions: questions,
        similarity_methods: selectedMethods.length > 0 ? selectedMethods : undefined,
        top_k_values: selectedTopK.length > 0 ? selectedTopK : undefined,
        doc_name: selectedDoc || undefined,
        normalize_vectors: normalizeVectors,
        include_faithfulness: includeFaithfulness,
        operations: operations
      };

      const result = await runBatchEvaluation(payload);
      setResults(result);
    } catch (error) {
      console.error("Batch evaluation failed:", error);
      alert("Batch evaluation failed. See console for details.");
    } finally {
      setIsRunning(false);
    }
  };

  const handleExport = async () => {
    if (!results) return;

    try {
      const jsonStr = JSON.stringify(results, null, 2);
      const blob = new Blob([jsonStr], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').split('T')[0] + '_' +
        new Date().toTimeString().split(' ')[0].replace(/:/g, '');
      const filename = `batch_eval_${timestamp}.json`;

      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Export failed:", error);
      alert("Export failed. See console for details.");
    }
  };

  const toggleMethod = (method: string) => {
    if (selectedMethods.includes(method)) {
      setSelectedMethods(selectedMethods.filter(m => m !== method));
    } else {
      setSelectedMethods([...selectedMethods, method]);
    }
  };

  const toggleTopK = (k: number) => {
    if (selectedTopK.includes(k)) {
      setSelectedTopK(selectedTopK.filter(v => v !== k));
    } else {
      setSelectedTopK([...selectedTopK, k]);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-slate-900 rounded-lg p-6 max-w-6xl w-full max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold text-slate-100">Batch Evaluation</h2>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-200 text-2xl leading-none"
          >
            ✕
          </button>
        </div>
        <div className="space-y-4 mb-6">
          <div>
            <label className="block text-sm text-slate-300 mb-2 font-semibold">
              Questions ({questions.length}/{MAX_QUESTIONS})
            </label>
            <div className="text-[11px] text-slate-400 mb-2">
              Add your questions manually (required, max {MAX_QUESTIONS})
            </div>
            <div className="flex gap-2 mb-2">
              <input
                type="text"
                value={questionInput}
                onChange={(e) => setQuestionInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleAddQuestion()}
                placeholder="Enter a question..."
                disabled={questions.length >= MAX_QUESTIONS}
                className="flex-1 bg-slate-800 text-slate-100 px-3 py-2 rounded text-sm border border-slate-700 focus:border-violet-500 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
              />
              <button
                onClick={handleAddQuestion}
                disabled={!questionInput.trim() || questions.length >= MAX_QUESTIONS}
                className="px-4 py-2 bg-sky-600 hover:bg-sky-700 disabled:bg-slate-700 disabled:cursor-not-allowed text-white rounded text-sm font-semibold"
              >
                Add
              </button>
            </div>

            {questions.length >= MAX_QUESTIONS && (
              <div className="text-xs text-amber-400 mb-2">
                Maximum {MAX_QUESTIONS} questions reached
              </div>
            )}

            {questions.length > 0 && (
              <div className="space-y-1 max-h-48 overflow-y-auto bg-slate-800/30 rounded p-2">
                {questions.map((q, idx) => (
                  <div key={idx} className="flex items-center gap-2 bg-slate-800 px-3 py-2 rounded">
                    <span className="text-slate-500 text-xs font-mono">{idx + 1}.</span>
                    <span className="flex-1 text-sm text-slate-200">{q}</span>
                    <button
                      onClick={() => setQuestions(questions.filter((_, i) => i !== idx))}
                      className="text-rose-400 hover:text-rose-300 text-xs"
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="border border-slate-700 rounded-lg p-4 bg-slate-800/30">
            <div className="text-sm text-slate-300 font-semibold mb-3">Operations to Run</div>

            <div className="space-y-3">
              <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-3">
                <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer hover:text-slate-100 sm:min-w-[80px]">
                  <input
                    type="checkbox"
                    checked={enableAsk}
                    onChange={(e) => setEnableAsk(e.target.checked)}
                    className="w-4 h-4 rounded border-slate-600 text-violet-600 focus:ring-violet-500"
                  />
                  <span className="font-semibold">Ask:</span>
                </label>

                <select
                  value={askModel}
                  onChange={(e) => setAskModel(e.target.value)}
                  disabled={!enableAsk}
                  className="flex-1 bg-slate-800 text-slate-100 px-2 py-1 rounded text-xs border border-slate-700 focus:border-violet-500 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {MODEL_OPTIONS.map(opt => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              </div>

              <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-3">
                <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer hover:text-slate-100 sm:min-w-[80px]">
                  <input
                    type="checkbox"
                    checked={enableCompare}
                    onChange={(e) => setEnableCompare(e.target.checked)}
                    className="w-4 h-4 rounded border-slate-600 text-violet-600 focus:ring-violet-500"
                  />
                  <span className="font-semibold">Compare:</span>
                </label>

                <div className="flex-1 grid grid-cols-2 gap-2">
                  <select
                    value={compareModelLeft}
                    onChange={(e) => setCompareModelLeft(e.target.value)}
                    disabled={!enableCompare}
                    className="bg-slate-800 text-slate-100 px-2 py-1 rounded text-xs border border-slate-700 focus:border-violet-500 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {MODEL_OPTIONS.map(opt => (
                      <option key={opt.value} value={opt.value}>{opt.label}</option>
                    ))}
                  </select>

                  <select
                    value={compareModelRight}
                    onChange={(e) => setCompareModelRight(e.target.value)}
                    disabled={!enableCompare}
                    className="bg-slate-800 text-slate-100 px-2 py-1 rounded text-xs border border-slate-700 focus:border-violet-500 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {MODEL_OPTIONS.map(opt => (
                      <option key={opt.value} value={opt.value}>{opt.label}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-3">
                <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer hover:text-slate-100">
                  <input
                    type="checkbox"
                    checked={enableCritique}
                    onChange={(e) => setEnableCritique(e.target.checked)}
                    className="w-4 h-4 rounded border-slate-600 text-violet-600 focus:ring-violet-500"
                  />
                  <span className="font-semibold">Critique:</span>
                </label>

                <div className="flex-1 grid grid-cols-2 gap-2">
                  <select
                    value={critiqueAnswerModel}
                    onChange={(e) => setCritiqueAnswerModel(e.target.value)}
                    disabled={!enableCritique}
                    className="bg-slate-800 text-slate-100 px-2 py-1 rounded text-xs border border-slate-700 focus:border-violet-500 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {MODEL_OPTIONS.map(opt => (
                      <option key={opt.value} value={opt.value}>{opt.label}</option>
                    ))}
                  </select>

                  <select
                    value={critiqueCriticModel}
                    onChange={(e) => setCritiqueCriticModel(e.target.value)}
                    disabled={!enableCritique}
                    className="bg-slate-800 text-slate-100 px-2 py-1 rounded text-xs border border-slate-700 focus:border-violet-500 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {MODEL_OPTIONS.map(opt => (
                      <option key={opt.value} value={opt.value}>{opt.label}</option>
                    ))}
                  </select>
                </div>

                <label className="flex items-center gap-2 text-xs text-slate-400 cursor-pointer hover:text-slate-300">
                  <input
                    type="checkbox"
                    checked={critiqueSelfCorrect}
                    onChange={(e) => setCritiqueSelfCorrect(e.target.checked)}
                    disabled={!enableCritique}
                    className="w-3 h-3 rounded border-slate-600 text-violet-600 focus:ring-violet-500 disabled:opacity-50 disabled:cursor-not-allowed"
                  />
                  <span>Self correct (2 rounds)</span>
                </label>
              </div>
            </div>

          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-slate-300 mb-2 font-semibold">
                Similarity Methods
              </label>
              <div className="space-y-2 bg-slate-800/30 rounded p-3">
                {[
                  { id: "cosine", label: "Cosine" },
                  { id: "dot", label: "Dot Product" },
                  { id: "hybrid", label: "Hybrid" },
                  { id: "neg_l2", label: "L2 Distance" },
                  { id: "neg_l1", label: "L1 Distance" }
                ].map(method => (
                  <label key={method.id} className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer hover:text-slate-100">
                    <input
                      type="checkbox"
                      checked={selectedMethods.includes(method.id)}
                      onChange={() => toggleMethod(method.id)}
                      className="w-4 h-4 rounded border-slate-600 text-violet-600 focus:ring-violet-500"
                    />
                    {method.label}
                  </label>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm text-slate-300 mb-2 font-semibold">
                Top-K Values
              </label>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 bg-slate-800/30 rounded p-3">
                {Array.from({ length: 20 }, (_, i) => i + 1).map(k => (
                  <label key={k} className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer hover:text-slate-100">
                    <input
                      type="checkbox"
                      checked={selectedTopK.includes(k)}
                      onChange={() => toggleTopK(k)}
                      className="w-4 h-4 rounded border-slate-600 text-violet-600 focus:ring-violet-500"
                    />
                    Top-{k}
                  </label>
                ))}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-slate-300 mb-2 font-semibold">
                Document Filter (Optional)
              </label>
              <select
                value={selectedDoc}
                onChange={(e) => setSelectedDoc(e.target.value)}
                className="w-full bg-slate-800 text-slate-100 px-2 py-1 rounded text-[11px] sm:text-xs border border-slate-700 focus:border-violet-500 focus:outline-none"
              >
                <option value="">All Documents</option>
                {documents.map(doc => (
                  <option key={doc} value={doc}>{doc}</option>
                ))}
              </select>
            </div>

            <div className="flex flex-col justify-end space-y-2">
              <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer hover:text-slate-100">
                <input
                  type="checkbox"
                  checked={normalizeVectors}
                  onChange={(e) => setNormalizeVectors(e.target.checked)}
                  className="w-4 h-4 rounded border-slate-600 text-violet-600 focus:ring-violet-500"
                />
                <span className="font-semibold">Normalize vectors</span>
              </label>

              <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer hover:text-slate-100">
                <input
                  type="checkbox"
                  checked={includeFaithfulness}
                  onChange={(e) => setIncludeFaithfulness(e.target.checked)}
                  className="w-4 h-4 rounded border-slate-600 text-violet-600 focus:ring-violet-500"
                />
                <span className="font-semibold">Include faithfulness metrics</span>
                <span className="text-slate-500 text-xs">(slower but more comprehensive)</span>
              </label>
            </div>
          </div>
        </div>

        <button
          onClick={handleRunBatch}
          disabled={!canRunBatch || isRunning}
          title={!documents.length ? "Upload documents first" : !questions.length ? "Add at least one question" : !selectedMethods.length ? "Select at least one similarity method" : !selectedTopK.length ? "Select at least one top-k value" : !(enableAsk || enableCompare || enableCritique) ? "Select at least one operation" : ""}
          className="w-full py-3 bg-violet-600 hover:bg-violet-700 disabled:bg-slate-700 disabled:cursor-not-allowed text-white font-semibold rounded mb-4 text-sm"
        >
          {isRunning ? "Running evaluation..." : !canRunBatch ? (
            !documents.length ? "Upload documents to enable" :
              !questions.length ? "Add questions to enable" :
                !(enableAsk || enableCompare || enableCritique) ? "Select at least one operation" :
                  "Configure settings to enable"
          ) : `Run Batch Evaluation (${questions.length} question${questions.length !== 1 ? 's' : ''})`}
        </button>

        {results && (
          <div ref={resultsRef} className="mt-6 space-y-4 border-t border-slate-700 pt-6">
            <div className="flex justify-between items-center">
              <h3 className="text-lg font-semibold text-slate-100">Results</h3>
              <button
                onClick={handleExport}
                className="px-4 py-2 bg-sky-600 hover:bg-sky-700 text-white rounded text-sm font-semibold"
              >
                Download JSON
              </button>
            </div>

            <div className="bg-slate-800/40 rounded-lg p-3">
              <div className="text-xs text-slate-400 mb-2">Experiment Metadata</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                <div>
                  <span className="text-slate-500">Total runs:</span>
                  <span className="ml-2 text-slate-200 font-semibold">{results.experiment_metadata.total_runs}</span>
                </div>
                <div>
                  <span className="text-slate-500">Successful:</span>
                  <span className="ml-2 text-emerald-400 font-semibold">{results.experiment_metadata.successful_runs}</span>
                </div>
                <div>
                  <span className="text-slate-500">Duration:</span>
                  <span className="ml-2 text-sky-400 font-semibold">{results.experiment_metadata.duration_seconds.toFixed(1)}s</span>
                </div>
                <div>
                  <span className="text-slate-500">Questions:</span>
                  <span className="ml-2 text-violet-400 font-semibold">{results.experiment_metadata.questions_count}</span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="bg-slate-800 rounded p-3">
                <div className="text-xs text-slate-400 mb-1">Avg Latency</div>
                <div className="text-2xl font-bold text-emerald-400">
                  {results.summary.overall.avg_latency_seconds.toFixed(2)}s
                </div>
              </div>

              <div className="bg-slate-800 rounded p-3">
                <div className="text-xs text-slate-400 mb-1">Avg Answer Length</div>
                <div className="text-2xl font-bold text-sky-400">
                  {results.summary.overall.avg_answer_length.toFixed(0)}
                </div>
              </div>

              <div className="bg-slate-800 rounded p-3">
                <div className="text-xs text-slate-400 mb-1">Avg Chunks</div>
                <div className="text-2xl font-bold text-violet-400">
                  {results.summary.overall.avg_chunks_retrieved.toFixed(1)}
                </div>
              </div>

              {results.summary.faithfulness && (
                <div className="bg-slate-800 rounded p-3">
                  <div className="text-xs text-slate-400 mb-1">Avg Evidence Coverage</div>
                  <div className="text-2xl font-bold text-amber-400">
                    {(results.summary.faithfulness.avg_evidence_coverage * 100).toFixed(1)}%
                  </div>
                </div>
              )}
            </div>

            <div className="text-sm text-slate-400 bg-slate-800/40 rounded p-3">
              <div className="font-semibold text-slate-300 mb-1">✓ Batch evaluation complete</div>
              <div>Full results saved. Use export buttons above to download detailed data for analysis.</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}