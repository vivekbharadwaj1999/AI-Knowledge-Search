import { useEffect, useRef, useLayoutEffect, useState, type KeyboardEvent } from "react";
import UploadPanel from "./components/UploadPanel";
import AskControls, { MODEL_OPTIONS } from "./components/AskPanel";
import ReportPanel from "./components/ReportPanel";
import ReactMarkdown from "react-markdown";
import logo from "./assets/logo.webp";
import InstructionsModal from "./components/InstructionsModal";
import UnifiedAnalysisModal from "./components/AdvancedAnalysisModal";
import BatchEvaluationPanel from "./components/BatchEvaluationPanel";
import AuthModal from "./components/AuthModal";
import UserDropdown from "./components/UserDropdown";
import DeleteAccountModal from "./components/DeleteAccountModal";

import {
  fetchDocuments,
  clearDocuments,
  generateReport,
  askQuestion,
  compareModels,
  generateInsights,
  fetchDocumentRelations,
  runCritique,
  type CritiqueResult,
  type PromptIssueTag,
  type CritiqueScores,
  type CritiqueRound,
  analyzeOperation,
  setAuthToken,
  getAuthToken,
  createGuestSession,
  getCurrentUser,
  logout,
  deleteAccount,
  fetchOperationsLog,
  checkOperationsLogExists,
  resetOperationsLog,
} from "./api";


import type {
  DocumentReport,
  CrossDocRelations,
  AutoInsights,
  SourceChunk,
} from "./api";

function MarkdownText({ text }: { text: string }) {
  const cleaned = text.replace(/<br\s*\/?>/gi, "  \n");

  return (
    <div
      className="prose prose-invert prose-sm max-w-none break-words
                 prose-p:my-1 prose-li:my-0.5 prose-ul:ml-4 prose-ol:ml-4">
      <ReactMarkdown
        components={{
          code: ({ children }) => (
            <span className="text-slate-100 break-words">{children}</span>
          ),
          pre: ({ children }) => (
            <div className="whitespace-pre-wrap break-words text-slate-100 font-sans text-xs leading-relaxed my-2">
              {children}
            </div>
          ),
        }}
      >
        {cleaned}
      </ReactMarkdown>
    </div>
  );
}


const PROMPT_TIP_LABELS: Record<PromptIssueTag, string> = {
  missing_context: "Add more context",
  too_vague: "Make your question more specific",
  no_format_specified: "Specify the output format",
  length_unspecified: "Mention how long/short the answer should be",
  ambiguous_audience: "Say who the answer is for",
  multi_question: "Split multiple questions into separate prompts",
};

function PromptTipsChips({ tags }: { tags: PromptIssueTag[] }) {
  if (!tags || tags.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-1 mb-2">
      {tags.map((tag) => (
        <span
          key={tag}
          className="inline-flex items-center gap-1 text-[11px] px-2 py-1 rounded-full
                     bg-amber-500/10 border border-amber-500/40 text-amber-200"
        >
          <span>{PROMPT_TIP_LABELS[tag] ?? tag}</span>
        </span>
      ))}
    </div>
  );
}

function PromptBeforeAfter({
  original,
  improved,
}: {
  original: string;
  improved: string;
}) {
  return (
    <div className="mt-2 space-y-2 text-xs">
      <div className="rounded-lg bg-slate-950/70 border border-slate-800 p-2">
        <div className="text-[10px] font-semibold uppercase text-slate-400 mb-1">
          Before
        </div>
        <div className="whitespace-pre-wrap break-words text-slate-200">
          {original}
        </div>
      </div>

      <div className="rounded-lg bg-emerald-950/40 border border-emerald-700/60 p-2">
        <div className="text-[10px] font-semibold uppercase text-emerald-300 mb-1">
          After (Improved prompt)
        </div>
        <div className="whitespace-pre-wrap break-words text-emerald-50">
          {improved}
        </div>
      </div>
    </div>
  );
}

type ScoreKey = "correctness" | "completeness" | "clarity" | "hallucination_risk";

const SCORE_LABELS: Record<ScoreKey, string> = {
  correctness: "Correctness",
  completeness: "Completeness",
  clarity: "Clarity",
  hallucination_risk: "Hallucination",
};

function ScorePills({ scores }: { scores?: CritiqueScores | null }) {
  if (!scores) return null;

  const entries: [ScoreKey, number][] = Object.entries(
    scores as Record<string, number>
  )
    .filter(([, v]) => typeof v === "number")
    .map(([k, v]) => [k as ScoreKey, v as number]);

  if (!entries.length) return null;

  const getColor = (key: ScoreKey, value: number) => {
    if (key === "hallucination_risk") {
      if (value >= 0.66) return "bg-red-900/60 border-red-600 text-red-300";
      if (value >= 0.33)
        return "bg-yellow-900/40 border-yellow-600 text-yellow-200";
      return "bg-green-900/40 border-green-600 text-green-200";
    }

    if (value >= 0.66) return "bg-green-900/40 border-green-600 text-green-200";
    if (value >= 0.33)
      return "bg-yellow-900/40 border-yellow-600 text-yellow-200";
    return "bg-red-900/60 border-red-600 text-red-300";
  };

  return (
    <div className="flex flex-wrap gap-1 mt-1">
      {entries.map(([key, value]) => (
        <span
          key={key}
          className={
            "inline-flex items-center px-2 py-0.5 rounded-full border text-[10px] font-medium " +
            getColor(key, value)
          }
        >
          {SCORE_LABELS[key]}: {Math.round(value * 100)}%
        </span>
      ))}
    </div>
  );
}

function computeAnswerSimilarity(a: string, b: string): number {
  const tokenize = (s: string) =>
    new Set(
      s
        .toLowerCase()
        .split(/[^a-z0-9]+/i)
        .filter((t) => t.length > 2)
    );

  const setA = tokenize(a);
  const setB = tokenize(b);
  if (setA.size === 0 && setB.size === 0) return 1;
  if (setA.size === 0 || setB.size === 0) return 0;

  let intersection = 0;
  for (const token of setA) {
    if (setB.has(token)) intersection++;
  }
  const union = setA.size + setB.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

function summarizeDrift(rounds: CritiqueRound[]) {
  if (!rounds || rounds.length < 2) return null;

  const first = rounds[0];
  const last = rounds[rounds.length - 1];

  const sim = computeAnswerSimilarity(first.answer, last.answer);
  let label: string;
  if (sim >= 0.8) label = "Low drift (very similar answers)";
  else if (sim >= 0.5) label = "Moderate drift (refined or rephrased)";
  else label = "High drift (substantially changed answer)";

  return {
    similarity: sim,
    label,
  };
}

function summarizeMultiRoundVerdict(rounds: CritiqueRound[]) {
  if (!rounds || rounds.length < 2) return null;

  const first = rounds[0].scores || {};
  const last = rounds[rounds.length - 1].scores || {};

  const c1 = first.correctness ?? null;
  const c2 = last.correctness ?? null;
  const h1 = first.hallucination_risk ?? null;
  const h2 = last.hallucination_risk ?? null;

  if (c1 === null || c2 === null) {
    return {
      verdict: "Not enough score data to compare rounds.",
      deltaCorrectness: 0,
      deltaHallucination: 0,
    };
  }

  const dc = c2 - c1;
  const dh = (h2 ?? 0) - (h1 ?? 0);

  let verdict: string;
  if (dc > 0.1 && dh <= 0.05) {
    verdict = "Second round improved correctness without significantly increasing hallucination risk.";
  } else if (dc < -0.1 && dh >= 0) {
    verdict = "Second round reduced correctness; consider disabling self-correct or adjusting the prompt.";
  } else if (Math.abs(dc) <= 0.1 && Math.abs(dh) <= 0.05) {
    verdict = "Multi-round loop did not significantly change answer quality.";
  } else if (dc > 0.1 && dh > 0.05) {
    verdict = "Second round improved correctness but also increased hallucination risk.";
  } else {
    verdict = "Mixed impact: small changes in correctness and/or hallucination risk.";
  }

  return {
    verdict,
    deltaCorrectness: dc,
    deltaHallucination: dh,
  };
}

function RoundScoreGraph({ rounds }: { rounds: CritiqueRound[] }) {
  if (!rounds || rounds.length === 0) return null;

  const scores = rounds
    .map((r) => r.scores?.correctness)
    .filter((v): v is number => typeof v === "number");

  if (scores.length === 0) return null;

  const first = scores[0];
  const last = scores[scores.length - 1];
  const delta = last - first;

  const formatPct = (v: number | null | undefined) =>
    typeof v === "number" ? `${Math.round(v * 100)}%` : "–";

  const deltaLabel =
    typeof delta === "number" && !Number.isNaN(delta)
      ? `${delta > 0 ? "+" : ""}${Math.round(delta * 100)}%`
      : null;

  return (
    <div className="mt-2">
      <div className="text-[10px] text-slate-400 mb-1">
        Correctness across rounds
      </div>

      <div className="flex flex-col gap-1 text-[11px] text-slate-200">
        <div className="flex flex-wrap gap-2">
          {rounds.map((r) => (
            <div
              key={r.round}
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-slate-900/70 border border-slate-700"
            >
              <span className="text-[10px] text-slate-400">R{r.round}</span>
              <span className="font-medium">
                {formatPct(r.scores?.correctness)}
              </span>
            </div>
          ))}
        </div>

        {deltaLabel && rounds.length > 1 && (
          <div className="text-[10px] text-slate-400">
            Net change from R1 to R{rounds[rounds.length - 1].round}:{" "}
            <span
              className={
                delta > 0
                  ? "text-emerald-300"
                  : delta < 0
                    ? "text-rose-300"
                    : "text-slate-200"
              }
            >
              {deltaLabel}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export type HighlightMode = "ai" | "keywords" | "sentences" | "off";

export type Message = {
  id: number;
  question: string;
  answer: string;
  operation?: "ask" | "compare" | "critique";
  topK?: number;
  docName?: string;
  similarity?: string;
  models?: string[];
  prompt?: string;
  maxRounds?: number;
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

type CritiqueRun = CritiqueResult & {
  id: number;
};

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

  if (mode === "keywords") {
    const phrases =
      insights?.keywords && insights.keywords.length > 0
        ? insights.keywords
        : message.question.split(/\s+/);
    return highlightWithKeywords(text, phrases);
  }

  const sentences = text.split(/(?<=[.!?])\s+/);
  let scores: number[] = new Array(sentences.length).fill(0);

  if (insights?.sentence_importance && insights.sentence_importance.length > 0) {
    for (let i = 0; i < sentences.length; i++) {
      const s = sentences[i];
      const lower = s.toLowerCase();
      let maxScore = 0;

      for (const item of insights.sentence_importance) {
        const sn = item.sentence.toLowerCase().trim();
        if (!sn) continue;
        if (lower.includes(sn) || sn.includes(lower)) {
          if (item.score > maxScore) maxScore = item.score;
        }
      }
      scores[i] = maxScore;
    }
  }

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

  const scoredIndices = scores
    .map((s, i) => ({ score: s, index: i }))
    .filter((x) => x.score > 0);

  if (scoredIndices.length === 0) {
    return highlightWithKeywords(text, tokensForPhrases);
  }

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

  return sentences.map((sentence, idx) => {
    const content = sentence + (idx < sentences.length - 1 ? " " : "");
    const sc = scores[idx];

    if (!keep.has(idx) || sc <= 0) {
      return <span key={idx}>{content}</span>;
    }

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

    let wrapperClass = "";
    if (maxScore <= 0) {
      wrapperClass =
        "bg-yellow-300/10 border-l border-yellow-400/40 pl-1 pr-0.5 rounded-sm";
    } else {
      const norm = sc / maxScore;
      if (norm >= 0.66) {
        wrapperClass =
          "bg-amber-400/25 border-l-4 border-amber-400 pl-[6px] pr-0.5 rounded-sm";
      } else if (norm >= 0.33) {
        wrapperClass =
          "bg-yellow-300/20 border-l-2 border-yellow-400/80 pl-1 pr-0.5 rounded-sm";
      } else {
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

type DocumentSelectorProps = {
  documents: string[];
  selectedDoc?: string;
  onChange: (doc?: string) => void;
  useAllDocs: boolean;
  setUseAllDocs: (v: boolean) => void;
};

function DocumentSelector({
  documents,
  selectedDoc,
  onChange,
  useAllDocs,
  setUseAllDocs,
}: DocumentSelectorProps) {
  const handleSelectChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value || undefined;
    onChange(value);
    if (value) {
      setUseAllDocs(false);
    }
  };

  const toggleAllDocs = () => {
    setUseAllDocs(!useAllDocs);
    if (!useAllDocs) {
      onChange(undefined);
    }
  };

  return (
    <div className="flex flex-col gap-2 text-xs sm:text-sm">
      <div className="text-slate-300 mb-2 font-bold">Search from:</div>
      <div className="flex items-center gap-2">
        <span className="text-slate-300 whitespace-nowrap mb-1">All documents</span>
        <button
          type="button"
          onClick={toggleAllDocs}
          className={`relative inline-flex h-5 w-9 items-center rounded-full transition ${useAllDocs ? "bg-sky-500" : "bg-slate-600"
            }`}
          aria-pressed={useAllDocs}
          aria-label="Toggle search across all documents"
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition ${useAllDocs ? "translate-x-4" : "translate-x-1"
              }`}
          />
        </button>
      </div>

      <div className="flex flex-col sm:flex-row sm:items-center gap-2">
        <span className="text-slate-300 whitespace-nowrap">
          or single document:
        </span>
        <select
          disabled={documents.length === 0 || useAllDocs}
          value={selectedDoc ?? ""}
          onChange={handleSelectChange}
          className="w-full sm:flex-1 min-w-0 rounded-md bg-slate-900 border border-slate-700
                     px-2 py-1 text-xs sm:text-sm text-slate-100
                     disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <option value="">Select a document…</option>
          {documents.map((doc) => (
            <option key={doc} value={doc}>
              {doc}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

type OutputEntryBase =
  | {
    kind: "ask";
    messageId: number;
  }
  | {
    kind: "compare";
    comparisonId: number;
  }
  | {
    kind: "critique";
    critiqueId: number;
  }
  | {
    kind: "report";
    docName?: string;
    report: DocumentReport;
  }
  | {
    kind: "relations";
    relations: CrossDocRelations;
  };


type OutputEntry = OutputEntryBase & {
  id: number;
};

type SimilarityMetric = "cosine" | "dot" | "neg_l2" | "neg_l1" | "hybrid";

const SIMILARITY_LABELS: Record<SimilarityMetric, string> = {
  cosine: "cosine similarity",
  dot: "dot product similarity",
  neg_l2: "negative Euclidean distance (L2)",
  neg_l1: "negative Manhattan distance (L1)",
  hybrid: "hybrid (cosine + Jaccard) similarity",
};

function App() {
  const [currentUsername, setCurrentUsername] = useState<string | null>(null);
  const [isGuestUser, setIsGuestUser] = useState(true);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [documents, setDocuments] = useState<string[]>([]);
  const [selectedDoc, setSelectedDoc] = useState<string | undefined>();
  const [docVersion, setDocVersion] = useState(0);
  const [useAllDocs, setUseAllDocs] = useState<boolean>(false);
  const [outputFeed, setOutputFeed] = useState<OutputEntry[]>([]);
  const askInputRef = useRef<HTMLTextAreaElement | null>(null);
  const leftColumnRef = useRef<HTMLElement | null>(null);
  const [rightMaxHeight, setRightMaxHeight] = useState<number | undefined>();

  const appendOutput = (entry: OutputEntryBase) => {
    setOutputFeed((prev): OutputEntry[] => {
      const maxId = prev.reduce(
        (acc, item) => (item.id > acc ? item.id : acc),
        0
      );
      const nextId = maxId + 1;

      return [
        {
          ...entry,
          id: nextId,
        },
        ...prev,
      ];
    });
  };

  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [topK, setTopK] = useState(5);
  const [modelId, setModelId] = useState<string>("llama-3.1-8b-instant");
  const [highlightMode, setHighlightMode] = useState<HighlightMode>("ai");
  const [showContextFor, setShowContextFor] = useState<number | null>(null);
  const [compareQuestion, setCompareQuestion] = useState("");
  const [modelLeft, setModelLeft] = useState("llama-3.1-8b-instant");
  const [modelRight, setModelRight] = useState("llama-3.3-70b-versatile");
  const [comparisons, setComparisons] = useState<Comparison[]>([]);
  const [isCompareLoading, setIsCompareLoading] = useState(false);
  const [critiqueQuestion, setCritiqueQuestion] = useState("");
  const [answerModelId, setAnswerModelId] = useState("llama-3.1-8b-instant");
  const [criticModelId, setCriticModelId] = useState("llama-3.3-70b-versatile");
  const [critiques, setCritiques] = useState<CritiqueRun[]>([]);
  const [isCritiqueLoading, setIsCritiqueLoading] = useState(false);
  const [relationsLoading, setRelationsLoading] = useState(false);
  const [enableSelfCorrect, setEnableSelfCorrect] = useState(false);
  const [similarityMetric, setSimilarityMetric] = useState<SimilarityMetric>("cosine");
  const [normalizeVectors, setNormalizeVectors] = useState(true);
  const [hasOperationsLog, setHasOperationsLog] = useState(false);
  const [showInstructions, setShowInstructions] = useState(false);
  const [mobileView, setMobileView] = useState<"operations" | "output">(
    "operations"
  );
  const [analysisModalOpen, setAnalysisModalOpen] = useState(false);
  const [analysisModalData, setAnalysisModalData] = useState<any>(null);
  const [showBatchEval, setShowBatchEval] = useState(false);
  const [analysisLoading, setAnalysisLoading] = useState<Record<string, boolean>>({});

  useEffect(() => {
    const initAuth = async () => {
      const token = getAuthToken();
      if (token) {
        const user = await getCurrentUser();
        if (user) {
          setCurrentUsername(user.username);
          setIsGuestUser(user.is_guest);
        } else {
          const guestData = await createGuestSession();
          setAuthToken(guestData.token, true);
          setCurrentUsername(guestData.username);
          setIsGuestUser(true);
        }
      } else {
        const guestData = await createGuestSession();
        setAuthToken(guestData.token, true);
        setCurrentUsername(guestData.username);
        setIsGuestUser(true);
      }
    };
    initAuth();
  }, []);

  const handleAuthSuccess = (token: string, username: string, isGuest: boolean) => {
    setAuthToken(token, isGuest);
    setCurrentUsername(username);
    setIsGuestUser(isGuest);
    setDocVersion((v) => v + 1);
  };

  const handleLogout = async () => {
    await logout();
    const guestData = await createGuestSession();
    setAuthToken(guestData.token, true);
    setCurrentUsername(guestData.username);
    setIsGuestUser(true);
    setDocuments([]);
    setOutputFeed([]);
    window.location.reload();
  };

  const handleDeleteAccount = async () => {
    try {
      await deleteAccount();
      const guestData = await createGuestSession();
      setAuthToken(guestData.token, true);
      setCurrentUsername(guestData.username);
      setIsGuestUser(true);
      setDocuments([]);
      setOutputFeed([]);
      window.location.reload();
      setShowDeleteModal(false);
    } catch (err) {
      throw err;
    }
  };

  useEffect(() => {
    async function checkExistingLogs() {
      try {
        const opsResult = await checkOperationsLogExists();
        setHasOperationsLog(opsResult.exists);
      } catch (err) {
        console.error("Failed to check logs", err);
      }
    }

    checkExistingLogs();
  }, []);

  useEffect(() => {
    fetchDocuments()
      .then((data) => {
        setDocuments(data.documents);

        if (data.documents.length > 0) {
          setSelectedDoc(data.documents[data.documents.length - 1]);
        } else {
          setSelectedDoc(undefined);
        }
      })
      .catch((err) => {
        console.error("Failed to fetch documents", err);
      });
  }, [docVersion]);

  useLayoutEffect(() => {
    function updateHeight() {
      if (window.innerWidth >= 1024 && leftColumnRef.current) {
        setRightMaxHeight(leftColumnRef.current.offsetHeight);
      } else {
        setRightMaxHeight(undefined);
      }
    }

    updateHeight();
    window.addEventListener("resize", updateHeight);
    return () => window.removeEventListener("resize", updateHeight);
  });

  const handleClearAll = async () => {
    if (!window.confirm("Remove all documents and reset the vector store?")) {
      return;
    }
    try {
      await clearDocuments();
      setDocuments([]);
      setSelectedDoc(undefined);
      setUseAllDocs(false);
      setMessages([]);
      setComparisons([]);
      setOutputFeed([]);
      setShowContextFor(null);
      setDocVersion((v) => v + 1);
    } catch (err) {
      console.error("Failed to clear documents", err);
      alert("Failed to remove documents. Check console for details.");
    }
  };

  const handleGenerateReport = async () => {
    if (!selectedDoc || useAllDocs) {
      alert("Please select a single document (disable 'All documents') first.");
      return;
    }

    try {
      setIsGeneratingReport(true);
      const res = await generateReport({ doc_name: selectedDoc });

      appendOutput({
        kind: "report",
        docName: selectedDoc,
        report: res,
      });
    } catch (err) {
      console.error("Failed to generate report", err);
      alert("Failed to generate AI report. Check console for details.");
    } finally {
      setIsGeneratingReport(false);
    }
  };

  const canAsk = question.trim().length > 0 && !isLoading;

  const handleAsk = async () => {
    if (!canAsk) return;
    setIsLoading(true);
    const trimmed = question.trim();

    try {
      const res = await askQuestion(
        trimmed,
        topK,
        selectedDoc,
        modelId,
        similarityMetric
      );

      const nextId = messages.length
        ? messages[messages.length - 1].id + 1
        : 1;
      const newMessage: Message = {
        id: nextId,
        question: trimmed,
        answer: res.answer,
        context: res.context,
        modelUsed: res.model_used || modelId,
        operation: "ask",
        topK: topK,
        docName: selectedDoc || undefined,
        similarity: similarityMetric || "cosine",
        sources: (res.sources || []) as SourceChunk[],
      };

      setMessages([...messages, newMessage]);
      appendOutput({ kind: "ask", messageId: nextId });
      setHasOperationsLog(true);

      setQuestion("");
    } catch (err) {
      console.error(err);

      const nextId = messages.length
        ? messages[messages.length - 1].id + 1
        : 1;
      const newMessage: Message = {
        id: nextId,
        question: trimmed,
        answer: "Sorry, something went wrong while answering this question.",
        context: [],
        modelUsed: modelId,
      };

      setMessages([...messages, newMessage]);
      appendOutput({ kind: "ask", messageId: nextId });
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuestionKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  };

  const triggerInsights = async (id: number) => {
    setMessages((prev) =>
      prev.map((m) =>
        m.id === id ? { ...m, insightsLoading: true, insightsError: null } : m
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

  const canCompare =
    compareQuestion.trim().length > 0 &&
    !isCompareLoading &&
    modelLeft !== modelRight;

  const handleCompare = async () => {
    if (!canCompare) return;
    setIsCompareLoading(true);
    const trimmed = compareQuestion.trim();

    try {
      const result = await compareModels(
        trimmed,
        modelLeft,
        modelRight,
        topK,
        useAllDocs ? undefined : selectedDoc,
        similarityMetric,
        normalizeVectors
      );

      const nextId = comparisons.length
        ? comparisons[comparisons.length - 1].id + 1
        : 1;

      const newComparison: Comparison = {
        id: nextId,
        question: trimmed,
        left: {
          model: result.left.model,
          answer: result.left.answer,
          context: result.left.context,
          sources: result.left.sources as SourceChunk[],
        },
        right: {
          model: result.right.model,
          answer: result.right.answer,
          context: result.right.context,
          sources: result.right.sources as SourceChunk[],
        },
      };

      setComparisons([...comparisons, newComparison]);
      appendOutput({ kind: "compare", comparisonId: nextId });
      setHasOperationsLog(true);

      setCompareQuestion("");
    } catch (err) {
      console.error("Compare failed", err);
      alert("Compare failed. Check console for details.");
    } finally {
      setIsCompareLoading(false);
    }
  };

  const canCritique =
    critiqueQuestion.trim().length > 0 &&
    !isCritiqueLoading &&
    documents.length > 0;

  const handleCritique = async () => {
    if (!canCritique) return;
    setIsCritiqueLoading(true);
    const trimmed = critiqueQuestion.trim();

    try {
      const result = await runCritique({
        question: trimmed,
        answer_model: answerModelId,
        critic_model: criticModelId,
        top_k: topK,
        doc_name: useAllDocs ? undefined : selectedDoc,
        self_correct: enableSelfCorrect,
        similarity: similarityMetric,
        normalize_vectors: normalizeVectors,
      });

      const nextId =
        critiques.length > 0 ? critiques[critiques.length - 1].id + 1 : 1;
      const run: CritiqueRun = { ...result, id: nextId };

      setCritiques((prev) => [...prev, run]);
      appendOutput({ kind: "critique", critiqueId: nextId });
      setHasOperationsLog(true);

    } catch (err) {
      console.error("Critique failed", err);
      alert("Critique failed. Check console for details.");
    } finally {
      setIsCritiqueLoading(false);
    }
  };

  const handleExportOperationsLog = async () => {
    try {
      const entries = await fetchOperationsLog();
      setHasOperationsLog(entries.length > 0);

      if (!entries.length) {
        alert("No operations logged yet.");
        return;
      }

      const json = JSON.stringify(entries, null, 2);
      const blob = new Blob([json], {
        type: "application/json;charset=utf-8;",
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "vivbot_logs.json");
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Failed to export operations log", err);
      alert("Failed to export logs. Check console for details.");
    }
  };

  const handleResetOperationsLog = async () => {
    if (!confirm("Are you sure you want to delete all logs? This cannot be undone.")) {
      return;
    }
    try {
      await resetOperationsLog();
      setHasOperationsLog(false);
      alert("Logs reset successfully.");
    } catch (err) {
      console.error("Failed to reset operations log", err);
      alert("Failed to reset logs. Check console for details.");
    }
  };

  const decreaseTopK = () => setTopK((k) => Math.max(1, k - 1));
  const increaseTopK = () => setTopK((k) => Math.min(20, k + 1));

  const handleTopKChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value.trim();
    if (raw === "") {
      setTopK(1);
      return;
    }
    const parsed = parseInt(raw, 10);
    if (!Number.isNaN(parsed)) {
      setTopK(Math.min(20, Math.max(1, parsed)));
    }
  };

  const handleAnalyzeRelations = async () => {
    if (!useAllDocs) {
      alert("Turn on 'All documents' to analyze relations between them.");
      return;
    }
    if (documents.length < 2) {
      alert("You need at least two documents to analyze relations.");
      return;
    }

    try {
      setRelationsLoading(true);
      const data = await fetchDocumentRelations({
        similarity: similarityMetric,
        normalize_vectors: normalizeVectors,
      });
      appendOutput({ kind: "relations", relations: data });
    } catch (err) {
      console.error("Failed to analyze relations", err);
      alert("Failed to analyze relations. Check console for details.");
    } finally {
      setRelationsLoading(false);
    }
  };

  const handleAdvancedAnalysis = async (msgId: number) => {
    const msg = messages.find(m => m.id === msgId);
    if (!msg || !msg.operation) return;

    setAnalysisLoading(prev => ({ ...prev, [msgId]: true }));

    try {
      let params: any = {
        operation: msg.operation,
        top_k: msg.topK || 7,
        doc_name: msg.docName || undefined,
        normalize_vectors: normalizeVectors,
      };

      if (msg.operation === "ask") {
        params.question = msg.question;
        params.model = msg.modelUsed;
      } else if (msg.operation === "compare") {
        params.question = msg.question;
        params.models = msg.models;
      } else if (msg.operation === "critique") {
        params.prompt = msg.prompt;
        params.max_rounds = msg.maxRounds || 2;
      }

      const result = await analyzeOperation(params);

      setAnalysisModalData({
        ...result,
        selectedMethod: msg.similarity || "cosine",
      });
      setAnalysisModalOpen(true);
      setHasOperationsLog(true);
    } catch (error) {
      console.error("Analysis failed:", error);
      alert("Failed to run advanced analysis. Please try again.");
    } finally {
      setAnalysisLoading(prev => ({ ...prev, [msgId]: false }));
    }
  };

  const handleAdvancedAnalysisCompare = async (comparisonId: number) => {
    const cmp = comparisons.find(c => c.id === comparisonId);
    if (!cmp) return;

    setAnalysisLoading(prev => ({ ...prev, ["cmp_" + comparisonId]: true }));

    try {
      const params = {
        operation: "compare" as const,
        question: cmp.question,
        models: [cmp.left.model, cmp.right.model],
        top_k: topK,
        doc_name: useAllDocs ? undefined : selectedDoc,
        normalize_vectors: normalizeVectors,
      };

      const result = await analyzeOperation(params);

      setAnalysisModalData({
        ...result,
        selectedMethod: similarityMetric || "cosine",
      });
      setAnalysisModalOpen(true);
      setHasOperationsLog(true);
    } catch (error) {
      console.error("Compare analysis failed:", error);
      alert("Failed to run advanced analysis. Please try again.");
    } finally {
      setAnalysisLoading(prev => ({ ...prev, ["cmp_" + comparisonId]: false }));
    }
  };

  const handleAdvancedAnalysisCritique = async (critiqueId: number) => {
    const crt = critiques.find(c => c.id === critiqueId);
    if (!crt) return;

    setAnalysisLoading(prev => ({ ...prev, ["crt_" + critiqueId]: true }));

    try {
      const params = {
        operation: "critique" as const,
        question: crt.question,
        answer_model: crt.answer_model,
        critic_model: crt.critic_model,
        top_k: topK,
        doc_name: useAllDocs ? undefined : selectedDoc,
        max_rounds: crt.rounds?.length ?? 1,
        normalize_vectors: normalizeVectors,
      };

      const result = await analyzeOperation(params);

      setAnalysisModalData({
        ...result,
        selectedMethod: similarityMetric || "cosine",
      });
      setAnalysisModalOpen(true);
      setHasOperationsLog(true);
    } catch (error) {
      console.error("Critique analysis failed:", error);
      alert("Failed to run advanced analysis. Please try again.");
    } finally {
      setAnalysisLoading(prev => ({ ...prev, ["crt_" + critiqueId]: false }));
    }
  };

  return (
    <div className="flex flex-col bg-slate-950 text-slate-100 min-h-screen overflow-hidden">
      <header className="shrink-0 border-b border-slate-800 px-4 sm:px-6 py-3 flex items-center justify-between gap-3">
        <div className="flex items-center gap-1 sm:gap-3">
          <img
            src={logo}
            alt="VivBot logo"
            className="h-10 w-10 sm:h-10 sm:w-10 object-contain rounded-xl"
          />
          <div className="flex flex-col ml-3">
            <h1 className="text-base sm:text-xl font-semibold">
              VivBot - An AI Document Knowledge Search
            </h1>
            <p className="text-[11px] sm:text-xs text-slate-400">
              Designed and developed by Vivek
            </p>
          </div>
        </div>
        <div className="flex flex-col items-end gap-2">
          <div
            className="
      flex flex-col items-end gap-2
      sm:flex-row sm:items-center sm:gap-3">
            <button
              onClick={() => setShowInstructions(true)}
              className="
        rounded-lg border border-slate-600 px-3 py-1.5
        text-xs sm:text-sm font-medium text-slate-100 hover:bg-slate-800">
              Instructions
            </button>
            {isGuestUser ? (
              <button
                onClick={() => setShowAuthModal(true)}
                className="
          rounded-lg border border-sky-600 bg-sky-600/10 px-3 py-1.5
          text-xs sm:text-sm font-medium text-sky-400 hover:bg-sky-600/20">
                Login/Signup
              </button>
            ) : (
              currentUsername && (
                <UserDropdown
                  username={currentUsername}
                  isGuest={isGuestUser}
                  onLogout={handleLogout}
                  onDeleteAccount={() => setShowDeleteModal(true)}
                />
              )
            )}
          </div>
          {isGuestUser && (
            <p className="text-[9px] sm:text-2xs text-slate-400 text-right max-w-[260px] sm:max-w-none">
              Guest mode - data will be deleted when you close the browser. Login/signup
              to save uploaded docs and logs.
            </p>
          )}
        </div>
      </header>

      <main className="flex-1 flex flex-col lg:flex-row overflow-hidden pb-20 lg:pb-0">
        <section
          ref={leftColumnRef}
          className={`w-full lg:w-1/2 border-b-0 lg:border-r border-slate-800 flex flex-col ${mobileView === "output" ? "hidden lg:flex" : ""
            }`}>
          <div className="px-4 sm:px-6 pt-4 pb-5 border-b border-slate-800">
            <h2 className="text-sm sm:text-base font-semibold mb-3">
              1. Upload & index a document
              <span className="text-[11px] text-slate-400 ml-2">
                (Currently supports PDF, TXT, CSV, DOCX, PPTX, XSLX file formats).
              </span>
            </h2>
            <UploadPanel onIndexed={() => setDocVersion((v) => v + 1)} />
          </div>

          <div className="px-4 sm:px-6 py-4 border-b border-slate-800 space-y-3">
            <h2 className="text-sm sm:text-base font-semibold mb-1">
              2. Documents & search scope
            </h2>

            <DocumentSelector
              documents={documents}
              selectedDoc={selectedDoc}
              onChange={setSelectedDoc}
              useAllDocs={useAllDocs}
              setUseAllDocs={setUseAllDocs}
            />

            <div className="mt-4 pb-2 pt-4 grid grid-cols-1 sm:grid-cols-3 gap-4 items-start">
              <div className="space-y-2">
                <h3 className="text-xs font-semibold text-slate-300 tracking-wide uppercase">
                  Similarity function
                </h3>
                <p className="text-[11px] text-slate-400">
                  Choose how document chunks are ranked for all operations below
                  (Relations between documents, Ask, Compare, Critique).
                </p>

                <select
                  className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs text-slate-100"
                  value={similarityMetric}
                  onChange={(e) => setSimilarityMetric(e.target.value as SimilarityMetric)}
                >
                  <option value="cosine">Cosine (default)</option>
                  <option value="dot">Dot product</option>
                  <option value="neg_l2">Negative Euclidean distance (L2)</option>
                  <option value="neg_l1">Negative Manhattan distance (L1)</option>
                  <option value="hybrid">Hybrid (Cosine + Jaccard keyword overlap)</option>
                </select>
              </div>

              <div className="space-y-2">
                <h3 className="text-xs font-semibold text-slate-300 tracking-wide uppercase">
                  Retrieval Top K
                </h3>
                <p className="text-[11px] text-slate-400">
                  Used for Ask, Compare, and Critique.
                </p>

                <label className="flex items-center gap-2 text-xs text-slate-300 pt-4 sm:pt-8">
                  <span>Top&nbsp;K:</span>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={decreaseTopK}
                      className="flex h-7 w-7 items-center justify-center rounded-full
                   border border-slate-600 bg-slate-900
                   text-xs text-slate-100 hover:bg-slate-800"
                    >
                      <span className="text-xl pb-1">–</span>
                    </button>

                    <input
                      type="text"
                      inputMode="numeric"
                      value={topK}
                      onChange={handleTopKChange}
                      className="w-12 rounded-md border border-slate-700 bg-slate-800
                   px-2 py-1 text-xs text-slate-100 text-center
                   focus:outline-none focus:ring-2 focus:ring-sky-500"
                    />

                    <button
                      type="button"
                      onClick={increaseTopK}
                      className="flex h-7 w-7 items-center justify-center rounded-full
                   border border-slate-600 bg-slate-900
                   text-xs text-slate-100 hover:bg-slate-800"
                    >
                      <span className="text-xl pb-1">+</span>
                    </button>
                  </div>
                </label>
              </div>

              <div className="space-y-2">
                <h3 className="text-xs font-semibold text-slate-300 tracking-wide uppercase">
                  Vector normalization
                </h3>
                <p className="text-[11px] text-slate-400">
                  If ON, embeddings are L2 normalized before scoring (dot ≈ cosine).
                </p>

                <div className="pt-2 sm:pt-4">
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setNormalizeVectors((v) => !v)}
                      className={`relative inline-flex h-5 w-9 items-center rounded-full transition ${normalizeVectors ? "bg-emerald-500" : "bg-slate-600"
                        }`}
                      aria-pressed={normalizeVectors}
                      aria-label="Toggle vector normalization"
                    >
                      <span
                        className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition ${normalizeVectors ? "translate-x-4" : "translate-x-1"
                          }`}
                      />
                    </button>
                    <span className="ml-1 text-base text-slate-300">
                      {normalizeVectors ? "ON" : "OFF"}
                    </span>
                  </div>
                </div>
              </div>
            </div>
            <div className="flex flex-wrap gap-2 mt-2">
              <button
                onClick={handleGenerateReport}
                disabled={!selectedDoc || useAllDocs || isGeneratingReport}
                className="px-3 py-1.5 text-xs sm:text-sm rounded-md border border-sky-500/70 bg-sky-600/80 hover:bg-sky-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isGeneratingReport ? "Generating report…" : "Generate AI Report"}
              </button>

              <button
                onClick={handleAnalyzeRelations}
                disabled={!useAllDocs || documents.length < 2 || relationsLoading}
                className="px-3 py-1.5 text-xs sm:text-sm rounded-md border border-purple-500/70 bg-purple-600/80 hover:bg-purple-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {relationsLoading
                  ? "Analyzing relations…"
                  : "Relations between these documents"}
              </button>

              <button
                onClick={() => setShowBatchEval(true)}
                disabled={documents.length === 0}
                className="px-3 py-1.5 text-xs sm:text-sm rounded-md border border-violet-500/70 bg-violet-600/80 hover:bg-violet-500 disabled:bg-slate-700 disabled:border-slate-600 disabled:cursor-not-allowed disabled:opacity-50 flex items-center gap-1.5"
                title={documents.length === 0 ? "Upload documents first to enable batch evaluation" : "Run batch experiments across multiple configurations"}
              >
                <span>Batch Evaluation</span>
              </button>

              <button
                onClick={handleClearAll}
                disabled={documents.length === 0}
                className="px-3 py-1.5 text-xs sm:text-sm rounded-md border border-slate-600 hover:border-rose-500 hover:text-rose-300 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Remove all documents
              </button>
            </div>
          </div>

          <div className="px-4 sm:px-6 py-4 border-b border-slate-800">
            <h2 className="text-sm sm:text-base font-semibold mb-3">
              3. Ask questions
            </h2>
            <AskControls
              selectedDoc={useAllDocs ? undefined : selectedDoc}
              question={question}
              setQuestion={setQuestion}
              topK={topK}
              modelId={modelId}
              setModelId={setModelId}
              canAsk={canAsk}
              isLoading={isLoading}
              onAsk={handleAsk}
              onQuestionKeyDown={handleQuestionKeyDown}
              compareQuestion={"" as any}
              setCompareQuestion={() => { }}
              modelLeft={modelLeft}
              setModelLeft={setModelLeft}
              modelRight={modelRight}
              setModelRight={setModelRight}
              canCompare={false}
              isCompareLoading={false}
              askInputRef={askInputRef}
              similarityMetric={similarityMetric}

            />
          </div>

          <div className="px-4 sm:px-6 py-4">
            <h2 className="text-sm sm:text-base font-semibold mb-3">
              4. Compare models
            </h2>
            <div className="mb-4 text-xs text-slate-400">
              <span className="font-semibold">
                {selectedDoc
                  ? `Answering for document: ${selectedDoc}`
                  : "Searching across all indexed documents"}
              </span>
              {", using "}
              <span className="font-semibold">Top K = {topK}</span>
              {", and "}
              <span className="font-semibold">
                {SIMILARITY_LABELS[similarityMetric]} function
              </span>
              {" (configured in section 2)."}
            </div>
            <div className="space-y-2">
              <label className="block text-[11px] text-slate-400 mb-0.5">
                Ask Question
              </label>
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
              </div>

              <div className="mt-3 flex justify-end">
                <button
                  onClick={handleCompare}
                  disabled={!canCompare}
                  className="inline-flex items-center justify-center rounded-lg px-4 py-2 text-sm font-medium
                   bg-violet-600 hover:bg-violet-500 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isCompareLoading ? "Comparing…" : "Compare models"}
                </button>
              </div>
            </div>
          </div>
          <div className="px-4 sm:px-6 py-4 border-t border-slate-800">
            <h2 className="text-sm sm:text-base font-semibold mb-2">
              5. Critique answer & prompt
            </h2>
            <div className="mb-4 text-xs text-slate-400">
              <span className="font-semibold">
                {selectedDoc
                  ? `Answering for document: ${selectedDoc}`
                  : "Searching across all indexed documents"}
              </span>
              {", using "}
              <span className="font-semibold">Top K = {topK}</span>
              {", and "}
              <span className="font-semibold">
                {SIMILARITY_LABELS[similarityMetric]} function
              </span>
              {" (configured in section 2)."}
            </div>
            <div className="space-y-2">
              <label className="block text-[11px] text-slate-400 mb-0.5">
                Ask Question
              </label>
              <textarea
                className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm
                           focus:outline-none focus:ring-2 focus:ring-emerald-500"
                rows={2}
                placeholder="Ask something to critique the answer and prompt..."
                value={critiqueQuestion}
                onChange={(e) => setCritiqueQuestion(e.target.value)}
              />

              <div className="flex flex-col sm:flex-row gap-2">
                <div className="flex-1">
                  <label className="block text-[11px] text-slate-400 mb-0.5">
                    Answer model
                  </label>
                  <select
                    className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-slate-100"
                    value={answerModelId}
                    onChange={(e) => setAnswerModelId(e.target.value)}
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
                    Critic model
                  </label>
                  <select
                    className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-slate-100"
                    value={criticModelId}
                    onChange={(e) => setCriticModelId(e.target.value)}
                  >
                    {MODEL_OPTIONS.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              <div className="mt-4 pt-4 border-slate-800 space-y-2">
                <h3 className="text-xs font-semibold text-slate-300 tracking-wide uppercase">
                  Self correcting loop
                </h3>
                <p className="text-[11px] text-slate-400">
                  When enabled, the system will run up to two critique & repair
                  rounds: answer → critique → improved prompt → answer again
                  (logged for research).
                </p>

                <label className="inline-flex items-center gap-2 text-xs text-slate-100 pt-4">
                  <button
                    type="button"
                    onClick={() => setEnableSelfCorrect(!enableSelfCorrect)}
                    className={`relative inline-flex h-5 w-9 items-center rounded-full transition ${enableSelfCorrect ? "bg-sky-500" : "bg-slate-600"
                      }`}
                    aria-pressed={enableSelfCorrect}
                    aria-label="Toggle self-correcting critique loop"
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition ${enableSelfCorrect ? "translate-x-4" : "translate-x-1"
                        }`}
                    />
                  </button>

                  <span>Enable self correcting critique loop (max 2 rounds)</span>
                </label>
              </div>
              <div className="flex justify-end">
                <button
                  onClick={handleCritique}
                  disabled={!canCritique}
                  className="inline-flex items-center justify-center rounded-lg px-4 py-2 pb-2 text-sm font-medium
                             bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isCritiqueLoading ? "Critiquing…" : "Critique answer & prompt"}
                </button>
              </div>
            </div>
            <div className="mt-4 pt-6 pb-2 flex flex-col sm:flex-row justify-between gap-2 border-t border-slate-800">
              <button
                type="button"
                onClick={handleExportOperationsLog}
                disabled={!hasOperationsLog}
                className="inline-flex items-center justify-center rounded-lg px-3 py-1.5 text-[11px] font-medium
             border border-emerald-600 text-emerald-200 bg-slate-900 hover:bg-emerald-800/40
             disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Export logs (JSON)
              </button>
              <button
                type="button"
                onClick={handleResetOperationsLog}
                disabled={!hasOperationsLog}
                className="inline-flex items-center justify-center rounded-lg px-3 py-1.5 text-[11px] font-medium
               border border-rose-600 text-rose-200 bg-slate-900 hover:bg-rose-800/40
               disabled:opacity-50 disabled:cursor-not-allowed">
                Reset logs
              </button>
            </div>

          </div>
        </section>

        <section
          className={`w-full lg:w-1/2 flex flex-col ${mobileView === "operations" ? "hidden lg:flex" : ""
            }`}
          style={
            rightMaxHeight
              ? { maxHeight: rightMaxHeight, overflowY: "auto" }
              : undefined}>
          <div className="flex-1 px-4 sm:px-6 py-4">
            <h2 className="text-sm sm:text-base font-semibold mb-3">Output</h2>

            {outputFeed.length === 0 ? (
              <p className="text-xs sm:text-sm text-slate-400">
                Run an operation on the left. Each result will appear here as a
                separate card.
              </p>
            ) : (
              <div className="space-y-4">
                {outputFeed.map((entry) => {
                  if (entry.kind === "ask") {
                    const msg = messages.find((m) => m.id === entry.messageId);
                    if (!msg) return null;

                    return (
                      <div
                        key={entry.id}
                        className="border border-slate-800 rounded-xl bg-slate-900/40 p-3 sm:p-4 space-y-2"
                      >
                        <div className="flex items-center justify-between text-[11px] text-slate-400">
                          <span className="font-semibold text-slate-100">
                            Ask question
                          </span>
                          {msg.modelUsed && <span>Model: {msg.modelUsed}</span>}
                        </div>
                        <div className="mt-2 flex justify-end">
                          <button
                            type="button"
                            onClick={() => handleAdvancedAnalysis(msg.id)}
                            disabled={analysisLoading[msg.id]}
                            className="inline-flex items-center gap-1 rounded-md border border-sky-500/70 bg-sky-600 hover:bg-sky-500 px-2 py-1 text-[11px] text-white disabled:opacity-50 transition"
                            title="Compare all 5 similarity methods"
                          >
                            {analysisLoading[msg.id] ? (
                              <>
                                <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                                </svg>
                                <span>Analyzing...</span>
                              </>
                            ) : (
                              <>
                                <span>Advanced Analysis</span>
                              </>
                            )}
                          </button>
                        </div>
                        <div className="mt-1">
                          <div className="text-[11px] text-sky-300">
                            Question
                          </div>
                          <div className="text-xs sm:text-[13px] text-slate-100 leading-relaxed">
                            <MarkdownText text={msg.question} />
                          </div>
                        </div>

                        <div className="mt-2">
                          <div className="text-[11px] text-emerald-300">
                            Answer
                          </div>
                          <div className="text-xs sm:text-[13px] text-slate-100 leading-relaxed">
                            <MarkdownText text={msg.answer} />
                          </div>
                        </div>

                        <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px]">
                          <button
                            type="button"
                            onClick={() => triggerInsights(msg.id)}
                            className="inline-flex items-center gap-1 rounded-md border border-violet-500/70 bg-violet-600 hover:bg-violet-500 px-2 py-1 text-[11px] text-white disabled:opacity-50"
                            disabled={msg.insightsLoading}
                          >
                            <span>Auto insights</span>
                            {msg.insightsLoading && (
                              <span className="opacity-80">(thinking...)</span>
                            )}
                          </button>

                          {msg.insightsError && (
                            <span className="text-red-400">
                              {msg.insightsError}
                            </span>
                          )}
                        </div>

                        {msg.insights && (
                          <div className="mt-2 border border-violet-700 bg-violet-950/40 rounded p-2 text-[11px] text-slate-100 space-y-1">
                            {msg.insights.summary && (
                              <div>
                                <div className="font-semibold text-violet-200 mb-0.5">
                                  Summary
                                </div>
                                <div className="whitespace-pre-wrap">
                                  {msg.insights.summary}
                                </div>
                              </div>
                            )}

                            {msg.insights.key_points?.length > 0 && (
                              <div>
                                <div className="font-semibold text-violet-200 mb-0.5">
                                  Key points
                                </div>
                                <ul className="list-disc list-inside space-y-0.5">
                                  {msg.insights.key_points.map((p, i) => (
                                    <li key={i}>{p}</li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            {msg.insights.entities?.length > 0 && (
                              <div>
                                <div className="font-semibold text-violet-200 mb-0.5">
                                  Entities
                                </div>
                                <div className="flex flex-wrap gap-1">
                                  {msg.insights.entities.map((e, i) => (
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

                            {msg.insights.keywords?.length > 0 && (
                              <div>
                                <div className="font-semibold text-violet-200 mb-0.5">
                                  Keywords
                                </div>
                                <div className="flex flex-wrap gap-1">
                                  {msg.insights.keywords.map((kw, i) => (
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

                            {msg.insights.suggested_questions?.length > 0 && (
                              <div>
                                <div className="font-semibold text-violet-200 mb-0.5">
                                  Suggested questions
                                </div>
                                <ul className="list-disc list-inside space-y-0.5">
                                  {msg.insights.suggested_questions.map(
                                    (q, i) => (
                                      <li key={i}>{q}</li>
                                    )
                                  )}
                                </ul>
                              </div>
                            )}

                            {msg.insights.mindmap && (
                              <div>
                                <div className="font-semibold text-violet-200 mb-0.5">
                                  Mindmap (text)
                                </div>
                                <pre className="whitespace-pre-wrap font-mono">
                                  {msg.insights.mindmap}
                                </pre>
                              </div>
                            )}
                          </div>
                        )}

                        {(msg.context.length > 0 ||
                          (msg.sources && msg.sources.length > 0)) && (
                            <div className="mt-2 text-xs">
                              <button
                                type="button"
                                disabled={!msg.insights}
                                onClick={() =>
                                  msg.insights &&
                                  setShowContextFor(
                                    showContextFor === msg.id ? null : msg.id
                                  )
                                }
                                className={`inline-flex items-center gap-1 rounded-md px-2 py-1 border ${msg.insights
                                  ? "bg-slate-800 hover:bg-slate-700 text-sky-200 border-slate-600"
                                  : "bg-slate-900 text-slate-500 border-slate-700 cursor-not-allowed opacity-60"
                                  }`}
                              >
                                {showContextFor === msg.id
                                  ? "Hide highlighted context"
                                  : "Show highlighted context"}
                              </button>

                              {!msg.insights && (
                                <span className="ml-2 text-[10px] text-slate-500">
                                  Run Auto insights first to enable AI
                                  highlighting.
                                </span>
                              )}

                              {showContextFor === msg.id && (
                                <div className="mt-2 text-[11px] text-slate-300 bg-slate-900/80 border border-slate-700 rounded max-h-80 overflow-y-auto relative">
                                  <div className="sticky top-0 z-10 bg-slate-900/95 border-b border-slate-700 px-2 py-1 flex items-center justify-between">
                                    <span className="text-[10px] text-slate-400">
                                      Highlighted context
                                    </span>
                                    <div className="flex items-center gap-1 text-[10px]">
                                      <span className="mr-1 text-slate-500">
                                        Highlight Mode:
                                      </span>
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

                                  <div className="px-2 py-1 space-y-1">
                                    {msg.sources && msg.sources.length > 0 ? (
                                      <>
                                        <div className="text-[10px] text-slate-400 mb-1">
                                          Used documents:{" "}
                                          {Array.from(
                                            new Set(
                                              msg.sources.map((s) => s.doc_name)
                                            )
                                          ).join(", ")}
                                        </div>
                                        {Array.from(
                                          Object.entries(
                                            msg.sources.reduce((acc, s) => {
                                              if (!acc[s.doc_name])
                                                acc[s.doc_name] = [];
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
                                                {renderHighlightedChunk(
                                                  c,
                                                  msg,
                                                  highlightMode
                                                )}
                                              </pre>
                                            ))}
                                          </div>
                                        ))}
                                      </>
                                    ) : (
                                      msg.context.map((c, idx) => (
                                        <pre
                                          key={idx}
                                          className="whitespace-pre-wrap border-b border-slate-800 last:border-none pb-1"
                                        >
                                          {renderHighlightedChunk(
                                            c,
                                            msg,
                                            highlightMode
                                          )}
                                        </pre>
                                      ))
                                    )}
                                  </div>
                                </div>
                              )}
                            </div>
                          )}
                      </div>
                    );
                  }

                  if (entry.kind === "compare") {
                    const cmp = comparisons.find(
                      (c) => c.id === entry.comparisonId
                    );
                    if (!cmp) return null;

                    return (
                      <div
                        key={entry.id}
                        className="border border-slate-800 rounded-xl bg-slate-900/40 p-3 sm:p-4 space-y-3"
                      >
                        <div className="flex items-center justify-between text-[11px] text-slate-400">
                          <span className="font-semibold text-slate-100">
                            Model comparison
                          </span>
                          <span>
                            {cmp.left.model} vs {cmp.right.model}
                          </span>
                        </div>
                        <div className="mt-2 flex justify-end">
                          <button
                            type="button"
                            onClick={() => handleAdvancedAnalysisCompare(cmp.id)}
                            disabled={analysisLoading["cmp_" + cmp.id]}
                            className="inline-flex items-center gap-1 rounded-md border border-sky-500/70 bg-sky-600 hover:bg-sky-500 px-2 py-1 text-[11px] text-white disabled:opacity-50 transition"
                          >
                            {analysisLoading["cmp_" + cmp.id] ? <>
                              <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                              </svg>
                              <span>Analyzing...</span>
                            </> : "Advanced Analysis"}
                          </button>
                        </div>

                        <div>
                          <div className="text-[11px] text-sky-300">
                            Question
                          </div>
                          <div className="text-xs sm:text-[13px] text-slate-100 leading-relaxed">
                            <MarkdownText text={cmp.question} />
                          </div>
                        </div>

                        <div className="mt-2 grid gap-3 sm:grid-cols-2">
                          <div className="space-y-1">
                            <div className="text-[11px] text-slate-400">
                              {cmp.left.model}
                            </div>
                            <div className="text-xs sm:text-[13px] text-slate-100 leading-relaxed">
                              <MarkdownText text={cmp.left.answer} />
                            </div>
                          </div>
                          <div className="space-y-1">
                            <div className="text-[11px] text-slate-400">
                              {cmp.right.model}
                            </div>
                            <div className="text-xs sm:text-[13px] text-slate-100 leading-relaxed">
                              <MarkdownText text={cmp.right.answer} />
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  }

                  if (entry.kind === "critique") {
                    const crt = critiques.find((c) => c.id === entry.critiqueId);
                    if (!crt) return null;

                    const hasTwoRounds = !!(crt.rounds && crt.rounds.length > 1);
                    const round1 = hasTwoRounds ? crt.rounds[0] : undefined;

                    return (
                      <div
                        key={entry.id}
                        className="border border-slate-800 rounded-xl bg-slate-900/40 p-3 sm:p-4 space-y-3"
                      >
                        <div className="flex items-center justify-between text-[11px] text-slate-400">
                          <span className="font-semibold text-slate-100">
                            Critique: answer & prompt
                          </span>
                          <span>
                            {crt.answer_model} (answer) · {crt.critic_model} (critic)
                          </span>
                        </div>
                        <div className="mt-2 flex justify-end">
                          <button
                            type="button"
                            onClick={() => handleAdvancedAnalysisCritique(crt.id)}
                            disabled={analysisLoading["crt_" + crt.id]}
                            className="inline-flex items-center gap-1 rounded-md border border-sky-500/70 bg-sky-600 hover:bg-sky-500 px-2 py-1 text-[11px] text-white disabled:opacity-50 transition"
                          >
                            {analysisLoading["crt_" + crt.id] ? <>
                              <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                              </svg>
                              <span>Analyzing...</span>
                            </> : "Advanced Analysis"}
                          </button>
                        </div>
                        <div>
                          <div className="text-[11px] text-sky-300">Question</div>
                          <div className="text-xs sm:text-[13px] text-slate-100 leading-relaxed">
                            <MarkdownText text={crt.question} />
                          </div>
                        </div>

                        {crt.prompt_issue_tags && crt.prompt_issue_tags.length > 0 && (
                          <PromptTipsChips
                            tags={crt.prompt_issue_tags as PromptIssueTag[]}
                          />
                        )}

                        {hasTwoRounds && round1 && (
                          <div className="mt-1 border-t border-slate-800 pt-3 space-y-2">
                            <div className="text-[11px] uppercase tracking-wide text-slate-400">
                              Self-correcting loop – Round 1
                            </div>

                            <div className="grid gap-3 md:grid-cols-2">
                              <div className="space-y-1">
                                <div className="text-[11px] font-semibold text-slate-300">
                                  Round 1 answer
                                </div>
                                <div className="rounded-lg bg-slate-950/60 border border-slate-800 p-2 text-xs break-words overflow-auto">
                                  <MarkdownText text={round1.answer} />
                                </div>
                              </div>

                              <div className="space-y-1">
                                <div className="text-[11px] font-semibold text-slate-300">
                                  Round 1 critique
                                </div>
                                <div className="rounded-lg bg-slate-950/60 border border-amber-900 p-2 text-xs break-words overflow-auto">
                                  <MarkdownText
                                    text={round1.answer_critique_markdown || ""}
                                  />
                                </div>
                              </div>
                            </div>

                            {round1.improved_prompt && (
                              <div className="space-y-1 mt-2">
                                <div className="text-[11px] font-semibold text-emerald-300">
                                  Improved prompt for Round 2
                                </div>
                                <div className="rounded-lg bg-slate-950/60 border border-emerald-900 p-2 text-xs break-words overflow-auto">
                                  <MarkdownText text={round1.improved_prompt} />
                                </div>
                              </div>
                            )}
                          </div>
                        )}

                        <div className="space-y-2">
                          <div className="text-[11px] uppercase tracking-wide text-slate-400 border-b border-slate-800 pb-1">
                            {hasTwoRounds ? "Answer quality – Round 2 (final)" : "Answer quality"}
                          </div>

                          <div className="grid gap-3 md:grid-cols-2">
                            <div className="space-y-1">
                              <div className="text-[11px] font-semibold text-slate-300">
                                Answer (base model)
                              </div>
                              <div className="rounded-lg bg-slate-950/60 border border-slate-800 p-2 text-xs break-words overflow-auto">
                                <MarkdownText text={crt.answer} />
                              </div>
                            </div>

                            <div className="space-y-1">
                              <div className="text-[11px] font-semibold text-slate-300">
                                Answer critique
                              </div>
                              <ScorePills scores={crt.scores || undefined} />
                              <div className="rounded-lg bg-slate-950/60 border border-amber-900 p-2 text-xs break-words overflow-auto">
                                <MarkdownText text={crt.answer_critique_markdown} />
                              </div>
                            </div>
                          </div>
                        </div>

                        <div className="space-y-2 border-t border-slate-800 pt-3">
                          <div className="text-[11px] uppercase tracking-wide text-slate-400">
                            Prompt issues & improved prompt
                          </div>

                          <div className="text-[11px] font-semibold text-emerald-300">
                            Prompt coaching
                          </div>

                          <div className="rounded-lg bg-slate-950/60 border border-emerald-900 p-2 text-xs break-words max-h-64 overflow-auto">
                            <MarkdownText text={crt.prompt_feedback_markdown} />
                          </div>

                          <PromptBeforeAfter
                            original={crt.question}
                            improved={crt.improved_prompt}
                          />

                          <div className="flex justify-end">
                            <button
                              type="button"
                              onClick={() => {
                                setQuestion(crt.improved_prompt);
                                askInputRef.current?.focus();
                                askInputRef.current?.scrollIntoView({
                                  behavior: "smooth",
                                  block: "center",
                                });
                              }}
                              className="inline-flex items-center gap-1 px-3 py-1.5 rounded-full border border-emerald-500/60
                       text-[11px] text-emerald-100 hover:bg-emerald-500/10"
                            >
                              Use this prompt in Ask
                            </button>
                          </div>
                        </div>

                        {hasTwoRounds && crt.rounds && (() => {
                          const verdict = summarizeMultiRoundVerdict(crt.rounds);
                          const drift = summarizeDrift(crt.rounds);

                          return (
                            <div className="mt-2 border border-slate-800 rounded-lg bg-slate-950/70 p-3 space-y-2">
                              <div className="text-[11px] uppercase tracking-wide text-slate-400">
                                Multi-round verdict
                              </div>

                              {verdict && (
                                <div className="text-[11px] text-slate-200 whitespace-pre-wrap">
                                  {verdict.verdict}
                                  {typeof verdict.deltaCorrectness === "number" &&
                                    typeof verdict.deltaHallucination === "number" && (
                                      <div className="mt-1 text-[10px] text-slate-400">
                                        Δ correctness:{" "}
                                        {Math.round(verdict.deltaCorrectness * 100)}% · Δ
                                        hallucination:{" "}
                                        {Math.round(verdict.deltaHallucination * 100)}%
                                      </div>
                                    )}
                                </div>
                              )}

                              {drift && (
                                <div className="mt-1 text-[11px] text-slate-200">
                                  <div>{drift.label}</div>
                                  <div className="text-[10px] text-slate-400">
                                    Overlap similarity between Round 1 and final answer:{" "}
                                    {Math.round(drift.similarity * 100)}%
                                  </div>
                                </div>
                              )}

                              <RoundScoreGraph rounds={crt.rounds} />
                            </div>
                          );
                        })()}
                      </div>
                    );
                  }

                  if (entry.kind === "report") {
                    return (
                      <div
                        key={entry.id}
                        className="border border-slate-800 rounded-xl bg-slate-900/40 p-3 sm:p-4 space-y-2"
                      >
                        <div className="flex items-center justify-between text-[11px] text-slate-400">
                          <span className="font-semibold text-slate-100">
                            AI Study Report
                          </span>
                          {entry.docName && <span>{entry.docName}</span>}
                        </div>
                        <ReportPanel
                          report={entry.report}
                          isGenerating={false}
                          lastDocName={entry.docName}
                        />
                      </div>
                    );
                  }

                  return (
                    <div
                      key={entry.id}
                      className="border border-slate-800 rounded-xl bg-slate-900/40 p-3 sm:p-4 space-y-2 text-sm"
                    >
                      <div className="flex items-center justify-between text-[11px] text-slate-400">
                        <span className="font-semibold text-slate-100">
                          Relations between documents
                        </span>
                      </div>

                      {entry.relations.global_themes?.length ? (
                        <div>
                          <div className="text-xs font-semibold text-slate-200 mb-1">
                            Global themes
                          </div>
                          <ul className="list-disc list-inside text-xs sm:text-sm text-slate-300 space-y-0.5">
                            {entry.relations.global_themes.map((t, i) => (
                              <li key={i}>{t}</li>
                            ))}
                          </ul>
                        </div>
                      ) : null}

                      {entry.relations.relations?.length ? (
                        <div>
                          <div className="text-xs font-semibold text-slate-200 mb-1">
                            Pairwise relations
                          </div>
                          <div className="space-y-2">
                            {entry.relations.relations.map((rel, idx) => (
                              <div
                                key={idx}
                                className="border border-slate-800 rounded-lg p-2 bg-slate-950/60"
                              >
                                <div className="text-[11px] text-slate-300 mb-0.5">
                                  {rel.doc_a} ↔ {rel.doc_b}
                                </div>
                                <div className="text-xs text-slate-300 whitespace-pre-wrap">
                                  {rel.relationship}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      ) : null}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </section>
      </main>
      <div className="lg:hidden fixed bottom-5 left-0 right-0 px-4 pointer-events-none">
        <div className="flex justify-between">
          {mobileView === "output" ? (
            <button
              type="button"
              onClick={() => setMobileView("operations")}
              className="inline-flex items-center justify-center rounded-full px-3 py-3 text-xs font-medium
                   bg-emerald-500 text-white shadow-lg shadow-emerald-500/40 border border-emerald-400
                   active:scale-95 transition-transform pointer-events-auto"
            >
              <span className="mr-1">◀</span>
              <span>OPERATIONS</span>
            </button>
          ) : (
            <span />
          )}

          {mobileView === "operations" && (
            <button
              type="button"
              onClick={() => setMobileView("output")}
              className="inline-flex items-center justify-center rounded-full px-3 py-3 text-xs font-medium
                   bg-sky-500 text-white shadow-lg shadow-sky-500/40 border border-sky-400
                   active:scale-95 transition-transform pointer-events-auto"
            >
              <span>OUTPUTS</span>
              <span className="ml-1">▶</span>
            </button>
          )}
        </div>
      </div>

      <InstructionsModal
        open={showInstructions}
        onClose={() => setShowInstructions(false)}
      />

      <AuthModal
        open={showAuthModal}
        onClose={() => setShowAuthModal(false)}
        onSuccess={handleAuthSuccess}
      />

      {showDeleteModal && currentUsername && (
        <DeleteAccountModal
          username={currentUsername}
          onConfirm={handleDeleteAccount}
          onCancel={() => setShowDeleteModal(false)}
        />
      )}

      <UnifiedAnalysisModal
        isOpen={analysisModalOpen}
        onClose={() => {
          setAnalysisModalOpen(false);
          setAnalysisModalData(null);
        }}
        data={analysisModalData}
        selectedMethod={analysisModalData?.selectedMethod}
      />

      {showBatchEval && (
        <BatchEvaluationPanel
          documents={documents}
          onClose={() => setShowBatchEval(false)}
        />
      )}
    </div>
  );
}

export default App;