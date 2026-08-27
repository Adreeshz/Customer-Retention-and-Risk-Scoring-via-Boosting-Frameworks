import { useMemo } from "react";
import type { Adapter } from "../lib/adapter";
import { confusionAt, fmtPct } from "../lib/analytics";
import { ConfusionGrid, CurveChart, ImportanceBars, TenureChart } from "./charts";
import { CountUp, IconPulse, SectionHead, SliderRow } from "./ui";

/* ============================================================================ */
/* SECTION: METRIC TILES — headline evaluation numbers with count-up animation  */
/* ============================================================================ */
function MetricTile({ label, value, color, note }: { label: string; value: number; color: string; note: string }) {
  return (
    <div className="panel relative overflow-hidden px-4 py-4">
      <span className="absolute inset-x-0 top-0 h-[2.5px]" style={{ background: `linear-gradient(90deg, ${color}, transparent 70%)` }} />
      <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-faint">{label}</p>
      <p className="mt-1.5 font-display text-[27px] font-bold leading-none tabular-nums" style={{ color }}>
        <CountUp value={value} format={(v) => fmtPct(v)} />
      </p>
      <p className="mt-1.5 text-[11px] leading-snug text-muted">{note}</p>
    </div>
  );
}

/* ============================================================================ */
/* SECTION: OVERVIEW WORKSPACE — evaluation suite: curves, threshold-tunable    */
/* confusion matrix, feature importance, tenure EDA, training facts and the     */
/* boosting-framework comparison table produced by the Python backend           */
/* ============================================================================ */
export function Overview({ adapter, threshold, onThreshold }: { adapter: Adapter; threshold: number; onThreshold: (t: number) => void }) {
  const { summary } = adapter;
  const api = adapter.info.mode === "api";

  /* threshold-dependent confusion counts are recomputed client-side in both    */
  /* modes from the held-out test scores, so the slider behaves identically     */
  const cm = useMemo(() => confusionAt(adapter.testScores.labels, adapter.testScores.scores, threshold), [adapter, threshold]);

  const importanceRows = useMemo(() => {
    const rows = summary.importance.map((value, i) => ({
      name: (adapter.featureNames[i] ?? `feature_${i}`).replace(/_/g, " "),
      value,
    }));
    rows.sort((a, b) => b.value - a.value);
    return rows.filter((r) => r.value > 0.004).slice(0, 9);
  }, [adapter, summary]);

  const imbalance = summary.churnRate > 0 ? (1 - summary.churnRate) / summary.churnRate : 0;

  const facts: [string, string][] = [
    ["Algorithm", "Gradient boosting"],
    ["Loss", "Logistic (log-loss)"],
    ["Trees", `${summary.trees} × depth 4`],
    ["Learning rate", "0.085"],
    ["Subsampling", "85% per tree"],
    ["L2 regularizer", api ? "1.2 (XGBoost comparison)" : "λ = 1.2"],
    ["Dataset", adapter.info.datasetSource],
    ["Engine", adapter.info.engine],
    ["Train rows", `${summary.trainRows.toLocaleString()}`],
    ["Test rows", `${summary.testRows.toLocaleString()}`],
    ["Fit time", `${(summary.trainedMs / 1000).toFixed(2)} s`],
    ["Random seed", `#${summary.seed}`],
    ["Churn rate", fmtPct(summary.churnRate)],
    ["Imbalance", `${imbalance.toFixed(1)} : 1`],
  ];

  return (
    <div className="anim-fade-up space-y-4">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-5">
        <MetricTile label="ROC · AUC" value={summary.curves.rocAuc} color="#33d6ae" note="Ranking quality across every threshold" />
        <MetricTile label="PR · AUC" value={summary.curves.prAuc} color="#5fa8f2" note="Imbalance-aware precision–recall area" />
        <MetricTile label="F1 Score" value={cm.f1} color="#f2b33d" note={`Harmonic mean at ${fmtPct(threshold, 0)} cut-off`} />
        <MetricTile label="Precision" value={cm.precision} color="#f2637a" note="Share of flagged customers who churn" />
        <MetricTile label="Recall" value={cm.recall} color="#e8eefa" note="Share of churners caught before they leave" />
      </div>

      <div className="grid gap-3 lg:grid-cols-3">
        <div className="panel p-4">
          <SectionHead title="ROC Curve" sub="Held-out test partition (20%)" />
          <CurveChart
            points={summary.curves.roc}
            color="#33d6ae"
            diagonal
            areaFrom="curve"
            xLabel="false positive rate"
            yLabel="true positive rate"
            metric="AUC"
            metricValue={summary.curves.rocAuc}
          />
        </div>
        <div className="panel p-4">
          <SectionHead title="Precision–Recall" sub="The honest view under class imbalance" />
          <CurveChart
            points={summary.curves.pr}
            color="#5fa8f2"
            areaFrom="curve"
            xLabel="recall"
            yLabel="precision"
            metric="PR-AUC"
            metricValue={summary.curves.prAuc}
          />
        </div>
        <div className="panel p-4">
          <SectionHead title="Confusion Matrix" sub={`Test partition · cut-off ${fmtPct(threshold, 0)}`} />
          <ConfusionGrid tp={cm.tp} fp={cm.fp} tn={cm.tn} fn={cm.fn} />
          <div className="mt-4">
            <SliderRow
              label="Decision threshold"
              value={threshold}
              min={0.15}
              max={0.75}
              step={0.01}
              display={fmtPct(threshold, 0)}
              onChange={onThreshold}
            />
          </div>
          <p className="mt-3 grid grid-cols-3 gap-2 text-center font-mono text-[11px] text-muted">
            <span>
              ACC <span className="text-ink">{fmtPct(cm.accuracy)}</span>
            </span>
            <span>
              F1 <span className="text-ink">{fmtPct(cm.f1)}</span>
            </span>
            <span>
              REC <span className="text-ink">{fmtPct(cm.recall)}</span>
            </span>
          </p>
        </div>
      </div>

      <div className="grid gap-3 lg:grid-cols-3">
        <div className="panel p-4">
          <SectionHead title="Feature Importance" sub={`Normalized split-gain across ${summary.trees} trees`} />
          <ImportanceBars rows={importanceRows} />
        </div>
        <div className="panel p-4">
          <SectionHead title="Churn Rate by Tenure" sub="Exploratory analysis — % labels above bars" />
          <TenureChart buckets={summary.tenure} />
          <div className="mt-2 flex items-center gap-4 font-mono text-[10px] text-faint">
            <span className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-sm bg-rose" /> churned
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-sm bg-teal/40" /> stayed
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-0.5 w-4 border-t border-dashed border-amber" /> rate trend
            </span>
          </div>
        </div>
        <div className="panel flex flex-col p-4">
          <SectionHead title="Training Facts" sub="Everything the model knows about itself" />
          <dl className="grid grid-cols-2 content-start gap-x-4 gap-y-2">
            {facts.map(([k, v]) => (
              <div key={k} className="rounded-md border border-linesoft bg-canvas/50 px-2.5 py-1.5">
                <dt className="font-mono text-[9.5px] uppercase tracking-wider text-faint">{k}</dt>
                <dd className="mt-0.5 truncate font-mono text-[11.5px] font-medium text-ink" title={v}>
                  {v}
                </dd>
              </div>
            ))}
          </dl>

          <div className="mt-3 rounded-md border border-line bg-canvas/50 py-2">
            <p className="px-3 font-mono text-[9.5px] uppercase tracking-[0.12em] text-faint">Boosting frameworks</p>
            <table className="mt-1 w-full text-left font-mono text-[11px]">
              <thead>
                <tr className="text-[9.5px] uppercase tracking-wider text-faint">
                  <th className="px-3 py-1 font-medium">Engine</th>
                  <th className="py-1 font-medium">ROC</th>
                  <th className="py-1 font-medium">PR</th>
                  <th className="py-1 pr-3 text-right font-medium">Role</th>
                </tr>
              </thead>
              <tbody>
                {adapter.info.frameworks.map((f) => (
                  <tr key={f.name} className={f.active ? "text-teal" : "text-muted"}>
                    <td className="px-3 py-1">{f.name}</td>
                    <td className="py-1 tabular-nums">{fmtPct(f.rocAuc)}</td>
                    <td className="py-1 tabular-nums">{fmtPct(f.prAuc)}</td>
                    <td className="py-1 pr-3 text-right">{f.active ? "● scoring" : "compared"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-3 flex items-center gap-2 rounded-md border border-teal/25 bg-teal/8 px-3 py-2 text-[11px] text-teal">
            <IconPulse size={14} />
            {api
              ? "Scores and exact tree-path contributions are served by the Python backend."
              : "Model re-fits live in your browser — run “python backend/run.py” for the full Telco pipeline."}
          </div>
        </div>
      </div>
    </div>
  );
}
