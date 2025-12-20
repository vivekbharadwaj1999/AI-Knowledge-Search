import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const oldIsGuest = localStorage.getItem("is_guest");
if (oldIsGuest === "true") {
  localStorage.removeItem("auth_token");
  localStorage.removeItem("is_guest");
  console.log("Cleared old guest session from localStorage");
}

let authToken: string | null = sessionStorage.getItem("auth_token") || localStorage.getItem("auth_token");
let isGuestMode: boolean = sessionStorage.getItem("is_guest") === "true" || localStorage.getItem("is_guest") === "true";

export function setAuthToken(token: string | null, isGuest: boolean = false) {
  authToken = token;
  isGuestMode = isGuest;
  
  if (token) {
    if (isGuest) {
      sessionStorage.setItem("auth_token", token);
      sessionStorage.setItem("is_guest", "true");
      localStorage.removeItem("auth_token");
      localStorage.removeItem("is_guest");
    } else {
      localStorage.setItem("auth_token", token);
      localStorage.setItem("is_guest", "false");
      sessionStorage.removeItem("auth_token");
      sessionStorage.removeItem("is_guest");
    }
    axios.defaults.headers.common["Authorization"] = `Bearer ${token}`;
  } else {
    localStorage.removeItem("auth_token");
    localStorage.removeItem("is_guest");
    sessionStorage.removeItem("auth_token");
    sessionStorage.removeItem("is_guest");
    delete axios.defaults.headers.common["Authorization"];
  }
}

export function getAuthToken(): string | null {
  return authToken;
}

export function isGuest(): boolean {
  return isGuestMode;
}

if (authToken) {
  axios.defaults.headers.common["Authorization"] = `Bearer ${authToken}`;
}

export async function createGuestSession(): Promise<{ token: string; username: string; is_guest: boolean }> {
  const res = await axios.post(`${API_BASE}/auth/guest`);
  return res.data;
}

export async function signup(username: string, password: string): Promise<{ token: string; username: string; is_guest: boolean }> {
  const res = await axios.post(`${API_BASE}/auth/signup`, { username, password });
  return res.data;
}

export async function login(username: string, password: string): Promise<{ token: string; username: string; is_guest: boolean }> {
  const res = await axios.post(`${API_BASE}/auth/login`, { username, password });
  return res.data;
}

export async function getCurrentUser(): Promise<{ username: string; is_guest: boolean } | null> {
  try {
    const res = await axios.get(`${API_BASE}/auth/me`);
    return res.data;
  } catch {
    return null;
  }
}

export async function logout(): Promise<void> {
  try {
    await axios.post(`${API_BASE}/auth/logout`);
  } catch (err) {
    console.error("Logout error:", err);
  }
}

export async function deleteAccount(): Promise<void> {
  await axios.delete(`${API_BASE}/auth/account`);
}

export type AskResult = {
  answer: string;
  context: string[];
  model_used?: string;
  sources?: SourceChunk[];
};

export type SentenceImportance = {
  sentence: string;
  score: number;
};

export type SourceChunk = {
  doc_name: string;
  text: string;
  score: number;
};

export type AutoInsights = {
  summary: string;
  key_points: string[];
  entities: string[];
  suggested_questions: string[];
  mindmap: string;
  reading_difficulty: string;
  sentiment: string;
  keywords: string[];
  highlights?: string[][];
  sentence_importance?: SentenceImportance[];
};

export type EmbeddingModel = {
  id: string;
  label: string;
  type: "local" | "openai";
  dimension: number;
  description: string;
};

export async function fetchEmbeddingModels(): Promise<EmbeddingModel[]> {
  const res = await axios.get(`${API_BASE}/embedding-models`);
  return res.data.models as EmbeddingModel[];
}

export async function uploadFile(
  file: File,
  chunk_size: number,
  chunk_overlap: number,
  embedding_model?: string 
) {
  const form = new FormData();
  form.append("file", file);
  form.append("chunk_size", String(chunk_size));
  form.append("chunk_overlap", String(chunk_overlap));
  if (embedding_model) {
    form.append("embedding_model", embedding_model);
  }

  const res = await axios.post(`${API_BASE}/ingest`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data as { 
    status: string; 
    chunks_indexed: number;
    embedding_model?: string;  
    embedding_dimension?: number;
  };
}

export async function fetchDocuments() {
  const res = await axios.get(`${API_BASE}/documents`);
  return res.data as { documents: string[] };
}

export async function askQuestion(
  question: string,
  top_k = 5,
  docName?: string,
  model?: string,
  similarity?: "cosine" | "dot" | "neg_l2" | "neg_l1" | "hybrid",
  normalizeVectors?: boolean,
  embeddingModel?: string 
): Promise<AskResult> {
  const payload: any = { question, top_k };
  if (docName) payload.doc_name = docName;
  if (model) payload.model = model;
  if (similarity) payload.similarity = similarity;
  if (normalizeVectors !== undefined) payload.normalize_vectors = normalizeVectors;
  if (embeddingModel) payload.embedding_model = embeddingModel; 
  const res = await axios.post(`${API_BASE}/ask`, payload);
  return res.data as AskResult;
}

export type CompareResult = {
  model: string;
  answer: string;
  context: string[];
  sources: SourceChunk[];
};

export type CompareResponse = {
  left: CompareResult;
  right: CompareResult;
};

export async function compareModels(
  question: string,
  modelLeft: string,
  modelRight: string,
  top_k = 5,
  docName?: string,
  similarity?: "cosine" | "dot" | "neg_l2" | "neg_l1" | "hybrid",
  normalizeVectors?: boolean,
  embeddingModel?: string
): Promise<CompareResponse> {
  const payload: any = {
    question,
    model_left: modelLeft,
    model_right: modelRight,
    top_k
  };
  if (docName) payload.doc_name = docName;
  if (similarity) payload.similarity = similarity;
  if (normalizeVectors !== undefined) payload.normalize_vectors = normalizeVectors;
  if (embeddingModel) payload.embedding_model = embeddingModel;
  const res = await axios.post(`${API_BASE}/compare`, payload);
  return res.data as CompareResponse;
}

export async function generateInsights(params: {
  question: string;
  answer: string;
  context: string[];
  model?: string;
}): Promise<AutoInsights> {
  const res = await axios.post(`${API_BASE}/insights`, params);
  return res.data as AutoInsights;
}

export async function clearDocuments() {
  const res = await axios.delete(`${API_BASE}/documents`);
  return res.data as { status: string; message: string };
}

export type KnowledgeGraphEdge = {
  source: string;
  relation: string;
  target: string;
};

export type QAItem = {
  question: string;
  answer: string;
};

export type DocumentReport = {
  doc_name: string;
  title?: string;
  executive_summary: string;
  sections: { heading: string; content: string }[];
  key_concepts: string[];
  concept_explanations: string[];
  relationships: string[];
  knowledge_graph: KnowledgeGraphEdge[];
  practice_questions: QAItem[];
  difficulty_level: "beginner" | "intermediate" | "advanced" | string;
  difficulty_explanation: string;
  study_path: string[];
  explain_like_im_5: string;
  cheat_sheet: string[];
};

export async function generateReport(params: {
  doc_name: string;
  model?: string;
}): Promise<DocumentReport> {
  const res = await axios.post(`${API_BASE}/report`, params);
  return res.data as DocumentReport;
}

export type DocPairRelation = {
  doc_a: string;
  doc_b: string;
  similarity: number;
  relationship: string;
};

export type CrossDocRelations = {
  documents: string[];
  global_themes: string[];
  relations: DocPairRelation[];
};

export async function fetchDocumentRelations(params?: {
  model?: string;
  similarity?: "cosine" | "dot" | "neg_l2" | "neg_l1" | "hybrid";
  normalize_vectors?: boolean;
}): Promise<CrossDocRelations> {
  const res = await axios.post(`${API_BASE}/document-relations`, params || {});
  return res.data as CrossDocRelations;
}

export async function runCritique(params: {
  question: string;
  answer_model: string;
  critic_model?: string;
  top_k?: number;
  doc_name?: string;
  self_correct?: boolean;
  similarity?: "cosine" | "dot" | "neg_l2" | "neg_l1" | "hybrid";
  normalize_vectors?: boolean;
  embedding_model?: string;  
}): Promise<CritiqueResult> {
  const res = await axios.post(`${API_BASE}/critique`, params);
  return res.data as CritiqueResult;
}

export type CritiqueLogRow = {
  timestamp?: string;
  question?: string;
  answer_model: string;
  critic_model: string;
  doc_name?: string | null;
  self_correct: boolean;
  similarity?: string | null;
  num_rounds: number;
  r1_correctness?: number | null;
  rN_correctness?: number | null;
  r1_hallucination?: number | null;
  rN_hallucination?: number | null;
  delta_correctness?: number | null;
  delta_hallucination?: number | null;
};

export async function fetchCritiqueLogRows(): Promise<CritiqueLogRow[]> {
  const res = await axios.get(`${API_BASE}/critique-log-rows`);
  return (res.data.rows || []) as CritiqueLogRow[];
}

export async function checkCritiqueLogsExist(): Promise<{ exists: boolean; count: number }> {
  const res = await axios.get(`${API_BASE}/critique-log-exists`);
  return res.data;
}

export type PromptIssueTag =
  | "missing_context"
  | "too_vague"
  | "no_format_specified"
  | "length_unspecified"
  | "ambiguous_audience"
  | "multi_question";

export type CritiqueScores = {
  correctness?: number | null;
  completeness?: number | null;
  clarity?: number | null;
  hallucination_risk?: number | null;
  prompt_quality?: number | null;
};

export type CritiqueRound = {
  round: number;
  question: string;
  answer: string;
  context: string[];
  sources?: SourceChunk[];
  answer_critique_markdown: string;
  prompt_feedback_markdown: string;
  improved_prompt: string;
  prompt_issue_tags: string[];
  scores?: CritiqueScores | null;
};

export type CritiqueResult = {
  question: string;
  answer_model: string;
  critic_model: string;
  answer: string;
  context: string[];
  sources?: SourceChunk[];
  answer_critique_markdown: string;
  prompt_feedback_markdown: string;
  improved_prompt: string;
  prompt_issue_tags?: PromptIssueTag[];
  scores?: CritiqueScores | null;
  rounds: CritiqueRound[];
};

export async function analyzeOperation(params: {
  operation: "ask" | "compare" | "critique";
  question?: string;
  model?: string;
  models?: string[];
  answer_model?: string;
  critic_model?: string;
  top_k?: number;
  doc_name?: string;
  max_rounds?: number;
  normalize_vectors?: boolean;
  embedding_model?: string;  
  temperature?: number;
}): Promise<any> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  
  if (authToken) {
    headers["Authorization"] = `Bearer ${authToken}`;
  }

  const response = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    headers,
    body: JSON.stringify(params),
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`Analysis failed (${response.status}): ${text}`);
  }
  return response.json();
}


export async function resetCritiqueLog(): Promise<void> {
  const res = await fetch(`${API_BASE}/reset-critique-log`, {
    method: "POST",
  });
  if (!res.ok) {
    throw new Error("Failed to reset critique log");
  }
}

export interface OperationsLogEntry {
  operation: string;
  timestamp: string;
  parameters: Record<string, any>;
  results: Record<string, any>;
}

export async function fetchOperationsLog(): Promise<OperationsLogEntry[]> {
  const res = await axios.get(`${API_BASE}/operations-log`);
  return res.data.entries || [];
}

export async function checkOperationsLogExists(): Promise<{ exists: boolean; count: number }> {
  const res = await axios.get(`${API_BASE}/operations-log-exists`);
  return res.data;
}

export async function resetOperationsLog(): Promise<void> {
  const res = await fetch(`${API_BASE}/reset-operations-log`, {
    method: "POST",
    headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
  });
  if (!res.ok) {
    throw new Error("Failed to reset operations log");
  }
}

export async function runBatchEvaluation(params: {
  questions?: string[];
  question_count?: number;
  similarity_methods?: string[];
  embedding_models?: string[];
  top_k_values?: number[];
  llm_models?: string[];
  doc_name?: string;
  temperature?: number;
  include_faithfulness?: boolean;
}): Promise<any> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  
  if (authToken) {
    headers["Authorization"] = `Bearer ${authToken}`;
  }

  const response = await fetch(`${API_BASE}/batch-evaluate`, {
    method: "POST",
    headers,
    body: JSON.stringify(params),
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`Batch evaluation failed (${response.status}): ${text}`);
  }
  return response.json();
}

export async function exportBatchResults(params: {
  results: any;
  format: "json";
}): Promise<{ status: string; filename: string; path: string; format: string }> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  
  if (authToken) {
    headers["Authorization"] = `Bearer ${authToken}`;
  }

  const response = await fetch(`${API_BASE}/batch-evaluate/export`, {
    method: "POST",
    headers,
    body: JSON.stringify(params),
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`Export failed (${response.status}): ${text}`);
  }
  return response.json();
}

export async function runCounterfactualAnalysis(params: {
  question: string;
  original_chunks: any[];
  counterfactual_type: string;
  top_k?: number;
  doc_name?: string;
  model?: string;
  similarity?: string;
  embedding_model?: string;
  temperature?: number;
  original_answer?: string; 
}): Promise<any> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  
  if (authToken) {
    headers["Authorization"] = `Bearer ${authToken}`;
  }

  const response = await fetch(`${API_BASE}/counterfactual-analysis`, {
    method: "POST",
    headers,
    body: JSON.stringify(params),
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`Counterfactual analysis failed (${response.status}): ${text}`);
  }
  return response.json();
}

