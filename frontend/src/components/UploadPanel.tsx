// src/components/UploadPanel.tsx
import { useState } from "react";
import { uploadFile } from "../api";

type UploadPanelProps = {
  onIndexed?: () => void;
};

export default function UploadPanel({ onIndexed }: UploadPanelProps) {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] || null;
    setFile(f);
    setStatus(null);
  };

  const handleUpload = async () => {
    if (!file) return;
    setIsUploading(true);
    setStatus(null);
    try {
      const res = await uploadFile(file);
      setStatus(`Indexed ${res.chunks_indexed} chunks from "${file.name}".`);
      onIndexed?.(); // 👈 notify parent to refresh doc list
    } catch (err: any) {
      console.error(err);
      setStatus(err?.response?.data?.detail || "Upload failed");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="flex flex-col gap-3">
      <input
        type="file"
        accept=".pdf,.txt,.csv,.docx,.pptx,.xlsx"
        onChange={handleChange}
        className="block w-full text-sm text-slate-200 file:mr-4 file:py-2 file:px-4 file:rounded-lg
                 file:border-0 file:text-sm file:font-semibold file:bg-sky-600 file:text-white
                 hover:file:bg-sky-500"
      />
      <button
        onClick={handleUpload}
        disabled={!file || isUploading}
        className="inline-flex items-center justify-center rounded-lg px-4 py-2 text-sm font-medium
                 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isUploading ? "Indexing..." : "Upload & Index"}
      </button>
      {status && (
        <p className="text-xs text-slate-300 whitespace-pre-line">{status}</p>
      )}
    </div>
  );
}
