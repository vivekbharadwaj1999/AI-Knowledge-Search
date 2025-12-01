// src/components/OutputPanel.tsx
import type { DocumentReport } from "../api";
import ReportPanel from "./ReportPanel";

type ChatMessage = {
  role: "user" | "assistant";
  text: string;
};

type ComparisonResult = {
  question: string;
  modelA: string;
  modelB: string;
  answerA: string;
  answerB: string;
};

type OutputPanelProps = {
  chat: ChatMessage[];
  comparisons: ComparisonResult[];
  report: DocumentReport | null;
  isGeneratingReport: boolean;
  lastReportDoc?: string;
};

export default function OutputPanel({
  chat,
  comparisons,
  report,
  isGeneratingReport,
  lastReportDoc,
}: OutputPanelProps) {
  return (
    <div className="space-y-6 text-sm sm:text-[15px]">
      {/* Chat output */}
      <section>
        <h2 className="text-sm sm:text-base font-semibold mb-2">
          Answers & chat
        </h2>
        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-3 sm:p-4 max-h-[260px] sm:max-h-[320px] overflow-y-auto">
          {chat.length === 0 ? (
            <p className="text-xs sm:text-sm text-slate-400">
              Ask a question on the left. Answers will appear here.
            </p>
          ) : (
            <div className="space-y-3">
              {chat.map((m, i) => (
                <div key={i} className="flex gap-2">
                  <span className="mt-[2px] text-[11px] uppercase tracking-wide text-slate-500">
                    {m.role === "user" ? "You" : "VivBot"}
                  </span>
                  <p className="flex-1 text-xs sm:text-sm whitespace-pre-wrap">
                    {m.text}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>

      {/* Model comparison output */}
      <section>
        <h2 className="text-sm sm:text-base font-semibold mb-2">
          Model comparison
        </h2>
        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-3 sm:p-4 max-h-[260px] sm:max-h-[320px] overflow-y-auto">
          {comparisons.length === 0 ? (
            <p className="text-xs sm:text-sm text-slate-400">
              Enter a comparison question on the left to see side-by-side
              answers.
            </p>
          ) : (
            <div className="space-y-4">
              {comparisons.map((c, idx) => (
                <div key={idx} className="space-y-2">
                  <p className="text-xs font-semibold text-slate-300">
                    Q: {c.question}
                  </p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-2.5">
                      <p className="text-[11px] font-semibold mb-1 text-slate-400">
                        {c.modelA}
                      </p>
                      <p className="text-xs sm:text-sm whitespace-pre-wrap">
                        {c.answerA}
                      </p>
                    </div>
                    <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-2.5">
                      <p className="text-[11px] font-semibold mb-1 text-slate-400">
                        {c.modelB}
                      </p>
                      <p className="text-xs sm:text-sm whitespace-pre-wrap">
                        {c.answerB}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>

      {/* AI Study Report */}
      <section>
        <h2 className="text-sm sm:text-base font-semibold mb-2">
          AI Study Report
        </h2>
        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-3 sm:p-4 max-h-[360px] sm:max-h-[420px] overflow-y-auto">
          {isGeneratingReport ? (
            <p className="text-xs sm:text-sm text-slate-400">
              Generating report…
            </p>
          ) : !report ? (
            <p className="text-xs sm:text-sm text-slate-400">
              Select a document on the left and click{" "}
              <span className="font-semibold">Generate AI Report</span> to turn
              it into a study guide.
            </p>
          ) : (
            <ReportPanel
              report={report}
              isGenerating={isGeneratingReport}
              lastDocName={lastReportDoc}
            />
          )}
        </div>
      </section>
    </div>
  );
}
