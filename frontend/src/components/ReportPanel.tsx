import type { DocumentReport } from "../api";

type ReportPanelProps = {
  report: DocumentReport | null;
  isGenerating: boolean;
  lastDocName?: string;
};

export default function ReportPanel({
  report,
  isGenerating,
  lastDocName,
}: ReportPanelProps) {
  if (isGenerating) {
    return (
      <div className="h-full flex items-center justify-center text-sm text-slate-300">
        Generating AI report for your document…
      </div>
    );
  }

  if (!report) {
    return (
      <div className="h-full flex items-center justify-center text-center text-sm text-slate-400 px-6">
        <div>
          <h2 className="text-base font-semibold mb-1">AI Study Report</h2>
          <p className="text-xs text-slate-400">
            Select a document above and click{" "}
            <span className="font-semibold">&quot;Generate AI Report&quot;</span>{" "}
            to turn it into a full interactive study guide.
          </p>
        </div>
      </div>
    );
  }

  const difficultyBadge =
    report.difficulty_level?.toLowerCase() === "beginner"
      ? "bg-emerald-500/15 text-emerald-300 border border-emerald-500/40"
      : report.difficulty_level?.toLowerCase() === "advanced"
      ? "bg-rose-500/15 text-rose-300 border border-rose-500/40"
      : "bg-amber-500/15 text-amber-200 border border-amber-500/40";

  return (
    <div className="h-full overflow-auto px-6 py-4 space-y-5 text-sm lg:text-[13px]">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-base lg:text-lg font-semibold">
            {report.title || lastDocName || "AI Study Report"}
          </h2>
          <p className="mt-1 text-[11px] uppercase tracking-wide text-slate-400">
            Auto generated from your uploaded document
          </p>
        </div>
        <span
          className={`px-3 py-1 rounded-full text-[10px] font-medium ${difficultyBadge}`}
        >
          {report.difficulty_level?.toUpperCase() || "INTERMEDIATE"}
        </span>
      </div>

      <section className="bg-slate-900/70 rounded-2xl p-4 border border-slate-800/80 shadow-sm">
        <h3 className="text-xs font-semibold mb-2">Executive summary</h3>
        <p className="text-slate-100 leading-relaxed">
          {report.executive_summary}
        </p>
      </section>

      {report.sections && report.sections.length > 0 && (
        <section className="space-y-2">
          <h3 className="text-xs font-semibold">Important sections</h3>
          <div className="grid gap-2">
            {report.sections.map((s, idx) => (
              <article
                key={idx}
                className="bg-slate-900/40 rounded-xl p-3 border border-slate-800/80"
              >
                <h4 className="text-xs font-semibold mb-1">{s.heading}</h4>
                <p className="text-slate-100 text-[13px] leading-relaxed">
                  {s.content}
                </p>
              </article>
            ))}
          </div>
        </section>
      )}

      {(report.key_concepts?.length ?? 0) > 0 && (
        <section className="grid lg:grid-cols-2 gap-4">
          <div>
            <h3 className="text-xs font-semibold mb-2">Key concepts</h3>
            <ul className="space-y-1 text-[13px]">
              {report.key_concepts.map((c, i) => (
                <li key={i} className="flex gap-2">
                  <span className="mt-1 h-1.5 w-1.5 rounded-full bg-sky-400" />
                  <span>{c}</span>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h3 className="text-xs font-semibold mb-2">Concept explanations</h3>
            <ul className="space-y-2 text-[13px] text-slate-100">
              {report.concept_explanations.map((text, i) => (
                <li key={i}>
                  <span className="font-semibold">
                    {report.key_concepts[i] || `Concept ${i + 1}`}:
                  </span>{" "}
                  {text}
                </li>
              ))}
            </ul>
          </div>
        </section>
      )}

      {(report.relationships?.length ?? 0) > 0 ||
      (report.knowledge_graph?.length ?? 0) > 0 ? (
        <section className="grid lg:grid-cols-2 gap-4">
          {report.relationships && report.relationships.length > 0 && (
            <div>
              <h3 className="text-xs font-semibold mb-2">
                Relationships between ideas
              </h3>
              <ul className="space-y-1 text-[13px] text-slate-100 list-disc list-inside">
                {report.relationships.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </div>
          )}
          {report.knowledge_graph && report.knowledge_graph.length > 0 && (
            <div>
              <h3 className="text-xs font-semibold mb-2">Mini knowledge graph</h3>
              <div className="text-[11px] bg-slate-900/40 rounded-xl p-3 border border-slate-800/80 space-y-1">
                {report.knowledge_graph.map((edge, i) => (
                  <div
                    key={i}
                    className="flex flex-wrap items-center gap-1 text-slate-100"
                  >
                    <span className="font-semibold">{edge.source}</span>
                    <span className="text-slate-400">
                      — {edge.relation} →
                    </span>
                    <span className="font-semibold">{edge.target}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      ) : null}

      {report.practice_questions && report.practice_questions.length > 0 && (
        <section>
          <h3 className="text-xs font-semibold mb-2">Practice questions</h3>
          <div className="space-y-2">
            {report.practice_questions.map((qa, i) => (
              <details
                key={i}
                className="bg-slate-900/40 rounded-xl p-3 border border-slate-800/80"
              >
                <summary className="cursor-pointer text-[13px] font-medium text-slate-100">
                  Q{i + 1}. {qa.question}
                </summary>
                <p className="mt-2 text-[13px] text-emerald-300">
                  <span className="font-semibold">Answer:</span> {qa.answer}
                </p>
              </details>
            ))}
          </div>
        </section>
      )}

      {(report.difficulty_explanation &&
        report.difficulty_explanation.length > 0) ||
      (report.study_path?.length ?? 0) > 0 ? (
        <section className="grid lg:grid-cols-2 gap-4">
          {report.difficulty_explanation && (
            <div>
              <h3 className="text-xs font-semibold mb-2">Difficulty breakdown</h3>
              <p className="text-[13px] text-slate-100">
                {report.difficulty_explanation}
              </p>
            </div>
          )}
          {report.study_path && report.study_path.length > 0 && (
            <div>
              <h3 className="text-xs font-semibold mb-2">
                Suggested study path
              </h3>
              <ol className="list-decimal list-inside text-[13px] text-slate-100 space-y-1">
                {report.study_path.map((step, i) => (
                  <li key={i}>{step}</li>
                ))}
              </ol>
            </div>
          )}
        </section>
      ) : null}

      {(report.explain_like_im_5 &&
        report.explain_like_im_5.length > 0) ||
      (report.cheat_sheet?.length ?? 0) > 0 ? (
        <section className="grid gap-4">
          {report.cheat_sheet && report.cheat_sheet.length > 0 && (
            <div>
              <h3 className="text-xs font-semibold mb-2">Cheat sheet</h3>
              <ul className="text-[13px] text-slate-100 space-y-1 list-disc list-inside">
                {report.cheat_sheet.map((item, i) => (
                  <li key={i}>{item}</li>
                ))}
              </ul>
            </div>
          )}
        </section>
      ) : null}
    </div>
  );
}
