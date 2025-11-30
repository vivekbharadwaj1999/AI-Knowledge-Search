// src/App.tsx
import { useEffect, useState } from "react";
import UploadPanel from "./components/UploadPanel";
import AskPanel from "./components/AskPanel";
import { fetchDocuments, clearDocuments } from "./api";

type DocSelectorProps = {
  documents: string[];
  selectedDoc?: string;
  onChange: (doc?: string) => void;
};

function DocumentSelector({ documents, selectedDoc, onChange }: DocSelectorProps) {
  if (documents.length === 0) {
    return (
      <div className="text-sm text-slate-400">
        No documents indexed yet. Upload one to get started.
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="text-slate-300">Document:</span>
      <select
        value={selectedDoc ?? ""}
        onChange={(e) => {
          const v = e.target.value || undefined;
          onChange(v);
        }}
        className="bg-slate-800 border border-slate-600 rounded-md px-2 py-1 text-sm text-slate-100"
      >
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
      <header className="shrink-0 border-b border-slate-800 px-6 py-4 flex items-center justify-between">
        <h1 className="text-xl font-semibold">
          VivBot - A document AI Knowledge Search
        </h1>

        <div className="flex items-center gap-4">
          <DocumentSelector
            documents={documents}
            selectedDoc={selectedDoc}
            onChange={setSelectedDoc}
          />
          <button
            onClick={handleClearAll}
            disabled={documents.length === 0}
            className={`px-3 py-1 text-sm rounded-md border border-red-500 text-red-400 hover:bg-red-500/10 transition ${
              documents.length === 0 ? "opacity-50 cursor-not-allowed" : ""
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
          <AskPanel selectedDoc={selectedDoc} />
        </div>
      </main>
    </div>
  );
}

export default App;
