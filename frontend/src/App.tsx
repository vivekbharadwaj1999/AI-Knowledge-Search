// src/App.tsx
import { useEffect, useState } from "react";
import UploadPanel from "./components/UploadPanel";
import AskPanel from "./components/AskPanel";
import { fetchDocuments } from "./api";

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

  return (
    <div className="flex flex-col overflow-hidden bg-slate-950 text-slate-100">
      <header className="shrink-0 border-b border-slate-800 px-6 py-4 flex items-center justify-between">
        <h1 className="text-xl font-semibold">
          VivBot - A document AI Knowledge Search
        </h1>
        <DocumentSelector
          documents={documents}
          selectedDoc={selectedDoc}
          onChange={setSelectedDoc}
        />
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
