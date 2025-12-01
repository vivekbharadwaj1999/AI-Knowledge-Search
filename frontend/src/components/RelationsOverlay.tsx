import type { CrossDocRelations } from "../api";

type RelationsOverlayProps = {
  open: boolean;
  onClose: () => void;
  data: CrossDocRelations | null;
  loading: boolean;
};

export default function RelationsOverlay({
  open,
  onClose,
  data,
  loading,
}: RelationsOverlayProps) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70">
      <div className="w-full max-w-3xl max-h-[80vh] bg-slate-950 border border-slate-800 rounded-2xl shadow-xl overflow-hidden flex flex-col">
        <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold">Relations between documents</h2>
            <p className="text-[11px] text-slate-400">
              High-level themes and pairwise relationships between your uploaded
              documents.
            </p>
          </div>
          <button
            className="text-xs px-2 py-1 rounded-md border border-slate-600 hover:border-slate-400 text-slate-200"
            onClick={onClose}
          >
            Close
          </button>
        </div>

        <div className="flex-1 overflow-auto px-4 py-3 text-sm">
          {loading && (
            <div className="h-full flex items-center justify-center text-slate-300">
              Analyzing relations between your documents…
            </div>
          )}

          {!loading && !data && (
            <div className="text-slate-400 text-sm">
              No data yet. Click the button again to generate relations.
            </div>
          )}

          {!loading && data && (
            <div className="space-y-4">
              {data.global_themes && data.global_themes.length > 0 && (
                <section>
                  <h3 className="text-xs font-semibold mb-1">Global themes</h3>
                  <ul className="list-disc list-inside text-[13px] text-slate-100 space-y-1">
                    {data.global_themes.map((t, i) => (
                      <li key={i}>{t}</li>
                    ))}
                  </ul>
                </section>
              )}

              {data.relations && data.relations.length > 0 && (
                <section>
                  <h3 className="text-xs font-semibold mb-2">
                    Pairwise document relations
                  </h3>
                  <div className="space-y-2">
                    {data.relations.map((rel, i) => (
                      <div
                        key={i}
                        className="rounded-lg border border-slate-800 bg-slate-900/60 p-3"
                      >
                        <div className="flex items-center justify-between gap-2">
                          <div className="text-[13px] font-semibold text-slate-100">
                            {rel.doc_a}{" "}
                            <span className="text-slate-500">↔</span>{" "}
                            {rel.doc_b}
                          </div>
                        </div>
                        <p className="mt-1 text-[13px] text-slate-100">
                          {rel.relationship}
                        </p>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {!loading &&
                data.relations &&
                data.relations.length === 0 && (
                  <p className="text-[13px] text-slate-400">
                    No strong relations detected between your documents.
                  </p>
                )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
