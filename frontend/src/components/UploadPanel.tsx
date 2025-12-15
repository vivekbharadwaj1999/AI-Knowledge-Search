import { useState } from "react";
import { uploadFile } from "../api";

type UploadPanelProps = {
  onIndexed?: () => void;
};

export default function UploadPanel({ onIndexed }: UploadPanelProps) {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunkOverlap, setChunkOverlap] = useState(200);

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
      const res = await uploadFile(file, chunkSize, chunkOverlap);
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

      <div className="my-3">
        <div className="text-xs font-semibold text-slate-300 tracking-wide uppercase mb-2">
          Chunk size
        </div>

        <label className="flex items-center gap-2 text-xs text-slate-300">
          <span>Chars:</span>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setChunkSize((s) => Math.max(200, s - 200))}
              className="flex h-7 w-7 items-center justify-center rounded-full
           border border-slate-600 bg-slate-900
           text-xs text-slate-100 hover:bg-slate-800"
              aria-label="Decrease chunk size"
            >
              <span className="text-xl pb-1">–</span>
            </button>

            <input
              type="text"
              inputMode="numeric"
              value={chunkSize}
              onChange={(e) => {
                const raw = parseInt(e.target.value.replace(/\D/g, ""), 10);
                if (Number.isNaN(raw)) return;

                const clamped = Math.min(4000, Math.max(200, raw));
                const snapped = Math.round(clamped / 200) * 200;

                setChunkSize(snapped);
              }}
              className="w-14 rounded-md border border-slate-700 bg-slate-800
           px-2 py-1 text-xs text-slate-100 text-center
           focus:outline-none focus:ring-2 focus:ring-sky-500"
            />

            <button
              type="button"
              onClick={() => setChunkSize((s) => Math.min(4000, s + 200))}
              className="flex h-7 w-7 items-center justify-center rounded-full
           border border-slate-600 bg-slate-900
           text-xs text-slate-100 hover:bg-slate-800"
              aria-label="Increase chunk size"
            >
              <span className="text-xl pb-1">+</span>
            </button>
          </div>
          <span className="text-[11px] text-slate-400">
            Changing chunk size requires re-indexing (Upload & Index again).
          </span>
        </label>
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
