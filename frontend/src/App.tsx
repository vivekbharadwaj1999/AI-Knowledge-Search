// src/App.tsx
import { useEffect, useState } from "react";
import UploadPanel from "./components/UploadPanel";
import AskPanel from "./components/AskPanel";
import { fetchDocuments, clearDocuments } from "./api";
import logo from "./assets/logo.webp";

type DocSelectorProps = {
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
}: DocSelectorProps) {
  if (documents.length === 0) {
    return (
      <div className="text-sm text-slate-400">
        No documents indexed yet. Upload one to get started.
      </div>
    );
  }

  const toggleAllDocs = () => {
    const next = !useAllDocs;
    setUseAllDocs(next);
    if (next) {
      // when switching to "all documents", clear specific doc
      onChange(undefined);
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-3 text-sm">
      {/* Label */}
      <span className="text-slate-300">Search from:</span>

      {/* "All documents" + switch */}
      <div className="flex items-center gap-2">
        <span className="text-slate-300">All documents</span>
        <button
          type="button"
          onClick={toggleAllDocs}
          className={`relative inline-flex h-5 w-9 items-center rounded-full transition
            ${useAllDocs ? "bg-sky-500" : "bg-slate-600"}`}
          aria-pressed={useAllDocs}
          aria-label="Toggle search across all documents"
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition
              ${useAllDocs ? "translate-x-4" : "translate-x-1"}`}
          />
        </button>
      </div>

      {/* Dropdown for single-doc mode */}
      <select
        value={selectedDoc ?? ""}
        disabled={useAllDocs}
        onChange={(e) => {
          const v = e.target.value || undefined;
          onChange(v);
        }}
        className={`bg-slate-800 border border-slate-600 rounded-md px-2 py-1 text-sm text-slate-100
          min-w-[220px]
          ${useAllDocs ? "opacity-50 cursor-not-allowed" : ""}`}
      >
        <option value="" disabled>
          Select a document…
        </option>
        {documents.map((doc) => (
          <option key={doc} value={doc}>
            {doc}
          </option>
        ))}
      </select>
    </div>
  );
}

function App() {
  const [documents, setDocuments] = useState<string[]>([]);
  const [selectedDoc, setSelectedDoc] = useState<string | undefined>();
  const [docVersion, setDocVersion] = useState(0);
  const [useAllDocs, setUseAllDocs] = useState<boolean>(true);

  useEffect(() => {
    fetchDocuments()
      .then((data) => {
        setDocuments(data.documents);
        if (data.documents.length > 0) {
          setSelectedDoc((prev) =>
            prev && data.documents.includes(prev)
              ? prev
              : data.documents[data.documents.length - 1]
          );
        } else {
          setSelectedDoc(undefined);
        }
      })
      .catch((err) => {
        console.error("Failed to fetch documents", err);
      });
  }, [docVersion]);

  const handleClearAll = async () => {
    if (documents.length === 0) return;

    const ok = window.confirm(
      "Are you sure you want to remove ALL documents and their embeddings?"
    );
    if (!ok) return;

    try {
      await clearDocuments();
      // reset local state
      setDocuments([]);
      setSelectedDoc(undefined);
      // bump version in case something else depends on refetch
      setDocVersion((v) => v + 1);
    } catch (err) {
      console.error("Failed to clear documents", err);
      alert("Failed to remove documents. Check console for details.");
    }
  };

  return (
    <div className="flex flex-col overflow-hidden bg-slate-950 text-slate-100">
      <header className="shrink-0 border-b border-slate-800 px-6 py-2 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <img
            src={logo}
            alt="VivBot logo"
            className="h-12 w-12 object-contain rounded-xl"
          />
          <h1 className="text-xl font-semibold">
            VivBot - A document AI Knowledge Search
          </h1>
        </div>

        <div className="flex items-center gap-4">
          <DocumentSelector
            documents={documents}
            selectedDoc={selectedDoc}
            onChange={setSelectedDoc}
            useAllDocs={useAllDocs}
            setUseAllDocs={setUseAllDocs}
          />
          <button
            onClick={handleClearAll}
            disabled={documents.length === 0}
            className={`px-3 py-1 text-sm rounded-md border border-red-500 text-red-400 hover:bg-red-500/10 transition ${documents.length === 0 ? "opacity-50 cursor-not-allowed" : ""
              }`}
          >
            Remove all documents
          </button>
        </div>
      </header>

      {/* Column layout: upload on top, ask/chat below */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Top: Upload panel, small-ish height */}
        <div className="shrink-0 p-6 pb-3">
          <UploadPanel onIndexed={() => setDocVersion((v) => v + 1)} />
        </div>

        {/* Bottom: Ask/chat panel fills remaining height */}
        <div className="flex-1 p-6 pt-3 overflow-hidden">
          <AskPanel selectedDoc={useAllDocs ? undefined : selectedDoc} />
        </div>
      </main>
    </div>
  );
}

export default App;
