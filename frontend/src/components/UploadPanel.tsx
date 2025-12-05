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
      onIndexed?.();
    } catch (err: any) {
      console.error(err);
      setStatus(err?.response?.data?.detail || "Upload failed");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-3">
        <label className="inline-flex items-center justify-center rounded-lg px-4 py-2 text-sm font-medium
                     bg-sky-600 hover:bg-sky-500 cursor-pointer">
          Choose File
          <input
            type="file"
            accept=".pdf,.txt,.csv,.docx,.pptx,.xlsx"
            onChange={handleChange}
            className="hidden"
          />
        </label>

        <span className="text-sm text-slate-200 truncate">
          {file ? file.name : "No file chosen"}
        </span>
      </div>
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
