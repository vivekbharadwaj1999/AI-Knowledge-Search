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
  score: number; // 0–5
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

export async function uploadFile(file: File) {
  const form = new FormData();
  form.append("file", file);
  const res = await axios.post(`${API_BASE}/ingest`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data as { status: string; chunks_indexed: number };
}

export async function fetchDocuments() {
  const res = await axios.get(`${API_BASE}/documents`);
  return res.data as { documents: string[] };
}

export async function askQuestion(
  question: string,
  top_k = 5,
  docName?: string,
  model?: string
): Promise<AskResult> {
  const payload: any = { question, top_k };
  if (docName) payload.doc_name = docName;
  if (model) payload.model = model;

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
}): Promise<CritiqueResult> {
  const res = await axios.post(`${API_BASE}/critique`, params);
  return res.data as CritiqueResult;
}

export type PromptIssueTag =
  | "missing_context"
  | "too_vague"
  | "no_format_specified"
  | "length_unspecified"
  | "ambiguous_audience"
  | "multi_question";

export type CritiqueScores = {
  correctness?: number;
  completeness?: number;
  clarity?: number;
  hallucination_risk?: number;
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
  scores?: CritiqueScores;
};
