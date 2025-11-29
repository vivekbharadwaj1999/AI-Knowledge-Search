// src/api.ts
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

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
  docName?: string
) {
  const payload: any = { question, top_k };
  if (docName) payload.doc_name = docName;

  const res = await axios.post(`${API_BASE}/ask`, payload);
  return res.data as { answer: string; context: string[] };
}
