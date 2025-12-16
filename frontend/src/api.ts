import axios from "axios";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "/api";

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
  const response = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
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
