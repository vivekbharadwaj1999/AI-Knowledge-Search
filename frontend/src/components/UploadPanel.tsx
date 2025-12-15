import { useState, useEffect } from "react";
import { uploadFile, fetchEmbeddingModels, type EmbeddingModel } from "../api";

type UploadPanelProps = {
  onIndexed?: () => void;
};

export default function UploadPanel({ onIndexed }: UploadPanelProps) {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunkOverlap, setChunkOverlap] = useState(60);
  const [embeddingModels, setEmbeddingModels] = useState<EmbeddingModel[]>([]);
  const [selectedEmbeddingModel, setSelectedEmbeddingModel] =
    useState<string>("all-MiniLM-L6-v2");
  const [loadingModels, setLoadingModels] = useState(true);
  const getModelShortLabel = (model: EmbeddingModel): string => {
    const baseLabel = model.label.split("–")[0].trim();

    if (model.id === "all-MiniLM-L6-v2") return `${baseLabel} (fast, free)`;
    if (model.id === "bge-base-en-v1.5") return `${baseLabel} (reliable, free)`;
    if (model.id === "Alibaba-NLP/gte-large-en-v1.5") return `GTE-large (best quality, free)`;
    if (model.id === "jinaai/jina-embeddings-v2-base-en") return `Jina v2 (long docs, free)`;
    if (model.id === "intfloat/e5-base") return `E5 (efficient, free)`;
    if (model.id === "intfloat/multilingual-e5-base") return `E5-multilingual (70+ langs, free)`;
    if (model.id === "hkunlp/instructor-large") return `INSTRUCTOR (high quality, free)`;
    if (model.id === "text-embedding-3-small") return `OpenAI Small (fast, paid)`;
    if (model.id === "text-embedding-3-large") return `OpenAI Large (powerful, paid)`;

    return baseLabel;
  };

  useEffect(() => {
    const loadModels = async () => {
      try {
        const models = await fetchEmbeddingModels();
        setEmbeddingModels(models);

        if (models.length > 0) {
          const defaultModel =
            models.find((m) => m.id === "all-MiniLM-L6-v2") || models[0];
          setSelectedEmbeddingModel(defaultModel.id);
        }
      } catch (err) {
        console.error("Failed to load embedding models:", err);
        setSelectedEmbeddingModel("all-MiniLM-L6-v2");
      } finally {
        setLoadingModels(false);
      }
    };
    loadModels();
  }, []);

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
      const res = await uploadFile(file, chunkSize, chunkOverlap, selectedEmbeddingModel);

      let statusMsg = `Indexed ${res.chunks_indexed} chunks from "${file.name}".`;
      if (res.embedding_model) {
        const modelInfo = embeddingModels.find((m) => m.id === res.embedding_model);
        const modelLabel = modelInfo?.label || res.embedding_model;
        statusMsg += `\nEmbedding model: ${modelLabel}`;
        if (res.embedding_dimension) statusMsg += ` (${res.embedding_dimension}D)`;
      }
      setStatus(statusMsg);

      onIndexed?.();
    } catch (err: any) {
      console.error(err);
      const errorDetail = err?.response?.data?.detail || "Upload failed";
      setStatus(errorDetail);
    } finally {
      setIsUploading(false);
    }
  };

  const selectedModelInfo = embeddingModels.find((m) => m.id === selectedEmbeddingModel);

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-3">
        <label
          className="inline-flex items-center justify-center rounded-lg px-4 py-2 text-sm font-medium
                     bg-sky-600 hover:bg-sky-500 cursor-pointer"
        >
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
          Index settings
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-[minmax(340px,1fr)_auto_auto] gap-4 items-end">
          <div className="min-w-0">
            <div className="text-[11px] font-semibold text-slate-300 tracking-wide uppercase mb-2">
              Embedding model
            </div>

            <select
              value={selectedEmbeddingModel}
              onChange={(e) => setSelectedEmbeddingModel(e.target.value)}
              disabled={loadingModels || isUploading}
              className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs text-slate-100
                         disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loadingModels ? (
                <option>Loading models...</option>
              ) : (
                <>
                  <optgroup label="Free Local Models">
                    {embeddingModels
                      .filter((m) => m.type === "local")
                      .map((model) => (
                        <option key={model.id} value={model.id}>
                          {getModelShortLabel(model)}
                        </option>
                      ))}
                  </optgroup>

                  {embeddingModels.some((m) => m.type === "openai") && (
                    <optgroup label="OpenAI (Paid - Only use if absolutely necessary)">
                      {embeddingModels
                        .filter((m) => m.type === "openai")
                        .map((model) => (
                          <option key={model.id} value={model.id}>
                            {getModelShortLabel(model)}
                          </option>
                        ))}
                    </optgroup>
                  )}
                </>
              )}
            </select>
          </div>

          <div className="w-fit">
            <div className="text-[11px] font-semibold sm:pl-8 text-slate-300 tracking-wide uppercase mb-2">
              Chunk size
            </div>

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
          </div>

          <div className="w-fit">
            <div className="text-[11px] font-semibold sm:pl-10 text-slate-300 tracking-wide uppercase mb-2">
              Overlap
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setChunkOverlap((o) => Math.max(0, o - 20))}
                className="flex h-7 w-7 items-center justify-center rounded-full
                           border border-slate-600 bg-slate-900
                           text-xs text-slate-100 hover:bg-slate-800"
                aria-label="Decrease overlap"
              >
                <span className="text-xl pb-1">–</span>
              </button>

              <input
                type="text"
                inputMode="numeric"
                value={chunkOverlap}
                onChange={(e) => {
                  const raw = parseInt(e.target.value.replace(/\D/g, ""), 10);
                  if (Number.isNaN(raw)) return;
                  const clamped = Math.min(200, Math.max(0, raw));
                  const snapped = Math.round(clamped / 20) * 20;
                  setChunkOverlap(snapped);
                }}
                className="w-14 rounded-md border border-slate-700 bg-slate-800
                           px-2 py-1 text-xs text-slate-100 text-center
                           focus:outline-none focus:ring-2 focus:ring-sky-500"
              />

              <button
                type="button"
                onClick={() => setChunkOverlap((o) => Math.min(200, o + 20))}
                className="flex h-7 w-7 items-center justify-center rounded-full
                           border border-slate-600 bg-slate-900
                           text-xs text-slate-100 hover:bg-slate-800"
                aria-label="Increase overlap"
              >
                <span className="text-xl pb-1">+</span>
              </button>
            </div>
          </div>
        </div>

        {selectedModelInfo && (
          <p className="mt-2 text-[11px] text-slate-400">
            {selectedModelInfo.description}
            {selectedModelInfo.type === "openai"}
          </p>
        )}

        <p className="mt-2 text-[11px] text-slate-400">
          Chunk size controls how much text goes into each chunk. Chunk overlap repeats part of the
          previous chunk to preserve context across boundaries. Changing any setting requires re-indexing
          (Upload &amp; Index again).
        </p>
      </div>

      <button
        onClick={handleUpload}
        disabled={!file || isUploading || loadingModels}
        className="inline-flex items-center justify-center rounded-lg px-4 py-2 text-sm font-medium
                   bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isUploading ? "Indexing..." : "Upload & Index"}
      </button>

      {status && <p className="text-xs text-slate-300 whitespace-pre-line">{status}</p>}
    </div>
  );
}
