// src/api.ts
import axios from "axios";

const API_BASE = "/api";

export type AskResult = {
  answer: string;
  context: string[];
  model_used?: string;
};

export type SentenceImportance = {
  sentence: string;
  score: number; // 0–5
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
