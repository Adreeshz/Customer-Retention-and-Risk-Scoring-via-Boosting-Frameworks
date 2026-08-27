import { useCallback, useEffect, useRef, useState } from "react";
import { useModel } from "./hooks/useModel";
import { Overview } from "./components/Overview";
import { Explorer } from "./components/Explorer";
import { Simulator } from "./components/Simulator";
import { IconFlask, IconGrid, IconRadar, IconRefresh, IconUsers, ToastHost } from "./components/ui";
import type { ToastItem, ToastKind } from "./components/ui";
import { fmtPct } from "./lib/analytics";

/* ============================================================================ */
/* SECTION: PERSISTED PREFERENCES — active tab and decision threshold survive   */
/* reloads via localStorage with safe fallbacks for corrupt entries             */
/* ============================================================================ */
type Tab = "overview" | "explorer" | "simulator";

interface Prefs {
  tab: Tab;
  threshold: number;
}

const PREFS_KEY = "churnlens:prefs:v1";

function loadPrefs(): Prefs {
  try {
    const raw = localStorage.getItem(PREFS_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<Prefs>;
      const tab: Tab = parsed.tab === "explorer" || parsed.tab === "simulator" ? parsed.tab : "overview";
      const threshold = typeof parsed.threshold === "number" && parsed.threshold >= 0.15 && parsed.threshold <= 0.75 ? parsed.threshold : 0.5;
      return { tab, threshold };
    }
  } catch {
    /* corrupt storage falls through to defaults */
  }
  return { tab: "overview", threshold: 0.5 };
}

/* ============================================================================ */
/* SECTION: APP SHELL — header, tab navigation, pipeline status surfaces and    */
/* the mounted workspace for the trained model                                  */
/* ============================================================================ */
export default function App() {
  const { state, seed, retrain } = useModel(42);
  const [prefs, setPrefs] = useState<Prefs>(loadPrefs);
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const toastId = useRef(0);
  const lastReportedSeed = useRef<number | null>(null);

  useEffect(() => {
    try {
      localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
    } catch {
      /* private-mode storage failures are non-fatal */
    }
  }, [prefs]);

  const notify = useCallback((message: string, kind: ToastKind = "success") => {
    const id = ++toastId.current;
    setToasts((t) => [...t.slice(-3), { id, kind, message }]);
    window.setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 4000);
  }, []);

  const dismissToast = useCallback((id: number) => setToasts((t) => t.filter((x) => x.id !== id)), []);

  useEffect(() => {
    if (state.status === "ready" && state.model) {
      const isFirstRun = lastReportedSeed.current === null;
      if (!isFirstRun && state.model.seed !== lastReportedSeed.current) {
        notify(`Model retrained with seed #${state.model.seed} — ROC-AUC ${fmtPct(state.model.curves.rocAuc)}.`, "success");
      }
      lastReportedSeed.current = state.model.seed;
    }
  }, [state.status, state.model, notify]);

  const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: "overview", label: "Model Performance", icon: <IconGrid size={15} /> },
    { id: "explorer", label: "Risk Explorer", icon: <IconUsers size={15} /> },
    { id: "simulator", label: "What-If Lab", icon: <IconFlask size={15} /> },
  ];

  return (
    <div className="relative min-h-screen">
      <div className="ambient-grid" aria-hidden />
      <div className="ambient-glow glow-teal" aria-hidden />
      <div className="ambient-glow glow-amber" aria-hidden />

      <div className="relative z-10 mx-auto max-w-[1380px] px-4 pb-16 pt-5 sm:px-6">
        <Header
          status={state.status}
          seed={seed}
          rocAuc={state.model?.curves.rocAuc ?? null}
          trees={state.model?.gbdt.trees.length ?? null}
          onRetrain={() => {
            notify(`Re-fitting ensemble with seed #${seed + 1}…`, "info");
            retrain(seed + 1);
          }}
        />

        <nav className="mt-5 flex flex-wrap items-center gap-1 border-b border-line" aria-label="Workspaces">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setPrefs((p) => ({ ...p, tab: t.id }))}
              className={`relative flex items-center gap-2 rounded-t-lg px-4 py-2.5 font-display text-[13.5px] font-semibold transition-colors ${
                prefs.tab === t.id ? "text-teal" : "text-muted hover:text-ink"
              }`}
              aria-current={prefs.tab === t.id ? "page" : undefined}
            >
              {t.icon}
              {t.label}
              {prefs.tab === t.id && <span className="absolute inset-x-3 -bottom-px h-[2.5px] rounded-full bg-teal" />}
            </button>
          ))}
          <div className="ml-auto hidden items-center gap-2 pb-2 font-mono text-[11px] text-faint lg:flex">
            <span className="rounded border border-line bg-panel px-2 py-1">dataset · telco-churn mirror</span>
            <span className="rounded border border-line bg-panel px-2 py-1">n = 2,400</span>
            {state.model && <span className="rounded border border-line bg-panel px-2 py-1">churn {fmtPct(state.model.churnRate, 0)}</span>}
          </div>
        </nav>

        <main className="mt-5">
          {state.status === "error" ? (
            <ErrorPanel message={state.error ?? "Unknown failure"} onRetry={() => retrain(seed)} />
          ) : state.status !== "ready" || !state.model ? (
            <LoadingPanel progress={state.progress} message={state.message} />
          ) : prefs.tab === "overview" ? (
            <Overview model={state.model} threshold={prefs.threshold} onThreshold={(t) => setPrefs((p) => ({ ...p, threshold: t }))} />
          ) : prefs.tab === "explorer" ? (
            <Explorer model={state.model} notify={notify} />
          ) : (
            <Simulator model={state.model} notify={notify} />
          )}
        </main>

        <footer className="mt-10 flex flex-col items-start justify-between gap-2 border-t border-linesoft pt-4 font-mono text-[11px] text-faint sm:flex-row sm:items-center">
          <p>
            ChurnLens · gradient boosting trained in-browser · contributions via exact tree-path decomposition
          </p>
          <p>seed #{seed} · synthetic cohort mirroring IBM Telco Customer Churn distributions</p>
        </footer>
      </div>

      <ToastHost toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}

/* ============================================================================ */
/* SECTION: HEADER — product mark, live model status chip and retrain control   */
/* ============================================================================ */
function Header({
  status,
  seed,
  rocAuc,
  trees,
  onRetrain,
}: {
  status: string;
  seed: number;
  rocAuc: number | null;
  trees: number | null;
  onRetrain: () => void;
}) {
  const busy = status !== "ready";
  return (
    <header className="flex flex-wrap items-center justify-between gap-3">
      <div className="flex items-center gap-3">
        <span className="grid h-11 w-11 place-items-center rounded-xl border border-teal/35 bg-teal/10 text-teal shadow-[0_0_28px_-6px_rgba(51,214,174,0.5)]">
          <IconRadar size={24} />
        </span>
        <div>
          <h1 className="font-display text-[22px] font-bold leading-tight tracking-tight text-ink">
            Churn<span className="text-teal">Lens</span>
          </h1>
          <p className="font-mono text-[11px] text-faint">boosted retention & risk scoring · machine learning project</p>
        </div>
      </div>

      <div className="flex items-center gap-2">
        {rocAuc !== null && (
          <span className="hidden items-center gap-2 rounded-lg border border-line bg-panel px-3 py-2 font-mono text-[11px] text-muted sm:flex">
            <span className="live-dot h-2 w-2 rounded-full bg-teal" />
            GBDT{trees ? ` · ${trees} trees` : ""} · ROC-AUC <span className="font-semibold text-teal">{fmtPct(rocAuc)}</span>
          </span>
        )}
        <span className="hidden rounded-lg border border-line bg-panel px-3 py-2 font-mono text-[11px] text-faint md:block">seed #{seed}</span>
        <button
          onClick={onRetrain}
          disabled={busy}
          className="flex items-center gap-2 rounded-lg border border-teal/45 bg-teal/12 px-3.5 py-2 font-mono text-[12px] font-semibold text-teal transition-all hover:bg-teal/22 active:scale-[0.97] disabled:opacity-50"
        >
          <IconRefresh size={14} className={busy ? "anim-spin" : ""} />
          {busy ? "Training…" : "Retrain model"}
        </button>
      </div>
    </header>
  );
}

/* ============================================================================ */
/* SECTION: PIPELINE STATUS PANELS — phased loading meter and failure recovery  */
/* ============================================================================ */
function LoadingPanel({ progress, message }: { progress: number; message: string }) {
  return (
    <div className="anim-fade-up mx-auto mt-16 max-w-md">
      <div className="panel px-6 py-8">
        <div className="flex items-center gap-3">
          <span className="grid h-10 w-10 place-items-center rounded-lg border border-teal/35 bg-teal/10 text-teal">
            <IconRefresh size={18} className="anim-spin" />
          </span>
          <div>
            <p className="font-display text-[16px] font-bold text-ink">Training pipeline</p>
            <p className="font-mono text-[11.5px] text-muted">{message}</p>
          </div>
        </div>
        <div className="mt-5 h-2 overflow-hidden rounded-full bg-canvas">
          <div
            className="h-full rounded-full bg-gradient-to-r from-teal via-amber to-rose transition-[width] duration-300 ease-out"
            style={{ width: `${Math.round(progress * 100)}%` }}
          />
        </div>
        <div className="mt-2 flex justify-between font-mono text-[10.5px] text-faint">
          <span>synthesize → fit → score → calibrate</span>
          <span>{Math.round(progress * 100)}%</span>
        </div>
      </div>
    </div>
  );
}

function ErrorPanel({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="anim-fade-up mx-auto mt-16 max-w-md">
      <div className="panel border-rose/40 px-6 py-8 text-center">
        <p className="font-display text-[17px] font-bold text-rose">Pipeline failed</p>
        <p className="mt-2 text-[13px] leading-relaxed text-muted">{message}</p>
        <button
          onClick={onRetry}
          className="mt-5 inline-flex items-center gap-2 rounded-lg border border-rose/50 bg-rose/12 px-4 py-2 font-mono text-[12px] font-semibold text-rose transition-all hover:bg-rose/22 active:scale-[0.97]"
        >
          <IconRefresh size={14} /> Retry training run
        </button>
      </div>
    </div>
  );
}


+++ src/App.tsx (修改后)
import { useCallback, useEffect, useRef, useState } from "react";
import { useSystem } from "./hooks/useSystem";
import { Overview } from "./components/Overview";
import { Explorer } from "./components/Explorer";
import { Simulator } from "./components/Simulator";
import { IconFlask, IconGrid, IconRadar, IconRefresh, IconUsers, ToastHost } from "./components/ui";
import type { ToastItem, ToastKind } from "./components/ui";
import { fmtPct } from "./lib/analytics";
import type { Adapter } from "./lib/adapter";

/* ============================================================================ */
/* SECTION: PERSISTED PREFERENCES — active tab and decision threshold survive   */
/* reloads via localStorage with safe fallbacks for corrupt entries             */
/* ============================================================================ */
type Tab = "overview" | "explorer" | "simulator";

interface Prefs {
  tab: Tab;
  threshold: number;
}

const PREFS_KEY = "churnlens:prefs:v1";

function loadPrefs(): Prefs {
  try {
    const raw = localStorage.getItem(PREFS_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<Prefs>;
      const tab: Tab = parsed.tab === "explorer" || parsed.tab === "simulator" ? parsed.tab : "overview";
      const threshold = typeof parsed.threshold === "number" && parsed.threshold >= 0.15 && parsed.threshold <= 0.75 ? parsed.threshold : 0.5;
      return { tab, threshold };
    }
  } catch {
    /* corrupt storage falls through to defaults */
  }
  return { tab: "overview", threshold: 0.5 };
}

/* ============================================================================ */
/* SECTION: APP SHELL — boots the two-tier system (Python backend first, then   */
/* the in-browser fallback), renders the tab navigation and the active workspace */
/* ============================================================================ */
export default function App() {
  const [prefs, setPrefs] = useState<Prefs>(loadPrefs);
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const toastId = useRef(0);

  const notify = useCallback((message: string, kind: ToastKind = "success") => {
    const id = ++toastId.current;
    setToasts((t) => [...t.slice(-3), { id, kind, message }]);
    window.setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 4200);
  }, []);

  const dismissToast = useCallback((id: number) => setToasts((t) => t.filter((x) => x.id !== id)), []);

  const { phase, adapter, seed, retrain, retry, fallbackToBrowser } = useSystem(notify);

  useEffect(() => {
    try {
      localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
    } catch {
      /* private-mode storage failures are non-fatal */
    }
  }, [prefs]);

  const busy = phase.kind !== "ready";

  const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: "overview", label: "Model Performance", icon: <IconGrid size={15} /> },
    { id: "explorer", label: "Risk Explorer", icon: <IconUsers size={15} /> },
    { id: "simulator", label: "What-If Lab", icon: <IconFlask size={15} /> },
  ];

  return (
    <div className="relative min-h-screen">
      <div className="ambient-grid" aria-hidden />
      <div className="ambient-glow glow-teal" aria-hidden />
      <div className="ambient-glow glow-amber" aria-hidden />
      <div className="ambient-glow glow-sky" aria-hidden />

      <div className="relative z-10 mx-auto max-w-[1380px] px-4 pb-16 pt-5 sm:px-6">
        <Header
          busy={busy}
          adapter={adapter}
          seed={seed}
          onRetrain={() => {
            notify(`Re-fitting the ${adapter?.info.mode === "api" ? "Python backend" : "browser engine"} with seed #${seed + 1}…`, "info");
            void retrain(seed + 1);
          }}
          onProbe={retry}
        />

        <nav className="mt-5 flex flex-wrap items-center gap-1 border-b border-line" aria-label="Workspaces">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setPrefs((p) => ({ ...p, tab: t.id }))}
              className={`relative flex items-center gap-2 rounded-t-lg px-4 py-2.5 font-display text-[13.5px] font-semibold transition-colors ${
                prefs.tab === t.id ? "text-teal" : "text-muted hover:text-ink"
              }`}
              aria-current={prefs.tab === t.id ? "page" : undefined}
            >
              {t.icon}
              {t.label}
              {prefs.tab === t.id && <span className="absolute inset-x-3 -bottom-px h-[2.5px] rounded-full bg-teal" />}
            </button>
          ))}
          <div className="ml-auto hidden items-center gap-2 pb-2 font-mono text-[11px] text-faint lg:flex">
            {adapter ? (
              <>
                <span className="max-w-[340px] truncate rounded border border-line bg-panel px-2 py-1" title={adapter.info.datasetSource}>
                  {adapter.info.datasetSource}
                </span>
                <span className="rounded border border-line bg-panel px-2 py-1">n = {adapter.info.datasetRows.toLocaleString()}</span>
                <span className="rounded border border-line bg-panel px-2 py-1">churn {fmtPct(adapter.summary.churnRate, 0)}</span>
              </>
            ) : (
              <span className="rounded border border-line bg-panel px-2 py-1">connecting to engine…</span>
            )}
          </div>
        </nav>

        <main className="mt-5">
          {phase.kind === "error" ? (
            <ErrorPanel message={phase.message} onRetry={retry} onFallback={fallbackToBrowser} />
          ) : phase.kind === "probing" ? (
            <LoadingPanel progress={0.12} message="Contacting the Python backend at 127.0.0.1:8000…" note="When it is running, the full Telco pipeline is served from Python." />
          ) : phase.kind === "api-loading" ? (
            <LoadingPanel progress={0.62} pulse message="Loading scored customers from the FastAPI service…" note="The backend downloads the Telco dataset on first launch, then serves instantly." />
          ) : phase.kind === "browser-training" ? (
            <LoadingPanel progress={phase.progress} message={phase.message} note="Fallback engine — the browser is training its own boosted ensemble." />
          ) : adapter && prefs.tab === "overview" ? (
            <Overview adapter={adapter} threshold={prefs.threshold} onThreshold={(t) => setPrefs((p) => ({ ...p, threshold: t }))} />
          ) : adapter && prefs.tab === "explorer" ? (
            <Explorer adapter={adapter} threshold={prefs.threshold} notify={notify} />
          ) : adapter ? (
            <Simulator adapter={adapter} notify={notify} />
          ) : null}
        </main>

        <footer className="mt-10 flex flex-col items-start justify-between gap-2 border-t border-linesoft pt-4 font-mono text-[11px] text-faint sm:flex-row sm:items-center">
          <p>
            {adapter?.info.mode === "api"
              ? `Python backend · FastAPI + scikit-learn GradientBoosting · ${adapter.info.datasetSource}`
              : "In-browser fallback engine · run “python backend/run.py” to activate the full Telco pipeline"}
          </p>
          <p>seed #{seed} · contributions sum exactly to every risk score</p>
        </footer>
      </div>

      <ToastHost toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}

/* ============================================================================ */
/* SECTION: HEADER — product mark, engine status pill, seed chip and retrain    */
/* control; clicking the pill re-probes the Python backend                      */
/* ============================================================================ */
function Header({
  busy,
  adapter,
  seed,
  onRetrain,
  onProbe,
}: {
  busy: boolean;
  adapter: Adapter | null;
  seed: number;
  onRetrain: () => void;
  onProbe: () => void;
}) {
  const api = adapter?.info.mode === "api";
  return (
    <header className="flex flex-wrap items-center justify-between gap-3">
      <div className="flex items-center gap-3">
        <span className="grid h-11 w-11 place-items-center rounded-xl border border-teal/35 bg-teal/10 text-teal shadow-[0_0_28px_-6px_rgba(51,214,174,0.5)]">
          <IconRadar size={24} />
        </span>
        <div>
          <h1 className="font-display text-[22px] font-bold leading-tight tracking-tight text-ink">
            Churn<span className="text-teal">Lens</span>
          </h1>
          <p className="font-mono text-[11px] text-faint">boosted retention & risk scoring · machine learning project</p>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={onProbe}
          title={api ? "Connected to the Python backend — click to re-probe" : "In-browser fallback engine — click to retry the Python backend"}
          className={`flex items-center gap-2 rounded-lg border px-3 py-2 font-mono text-[11px] transition-all active:scale-[0.97] ${
            !adapter
              ? "border-line text-faint"
              : api
                ? "border-teal/40 bg-teal/8 text-teal hover:bg-teal/15"
                : "border-amber/40 bg-amber/8 text-amber hover:bg-amber/15"
          }`}
        >
          <span className={`live-dot h-2 w-2 rounded-full ${!adapter ? "bg-faint" : api ? "bg-teal" : "bg-amber"}`} />
          {!adapter ? "connecting…" : api ? `Python API · ${adapter.info.datasetRows.toLocaleString()} rows` : "Browser engine · demo cohort"}
        </button>
        {adapter && (
          <span className="hidden items-center gap-2 rounded-lg border border-line bg-panel px-3 py-2 font-mono text-[11px] text-muted sm:flex">
            ROC-AUC <span className="font-semibold text-teal">{fmtPct(adapter.summary.curves.rocAuc)}</span>
          </span>
        )}
        <span className="hidden rounded-lg border border-line bg-panel px-3 py-2 font-mono text-[11px] text-faint md:block">seed #{seed}</span>
        <button
          onClick={onRetrain}
          disabled={busy || !adapter}
          className="flex items-center gap-2 rounded-lg border border-teal/45 bg-teal/12 px-3.5 py-2 font-mono text-[12px] font-semibold text-teal transition-all hover:bg-teal/22 active:scale-[0.97] disabled:opacity-50"
        >
          <IconRefresh size={14} className={busy ? "anim-spin" : ""} />
          {busy ? "Working…" : "Retrain model"}
        </button>
      </div>
    </header>
  );
}

/* ============================================================================ */
/* SECTION: PIPELINE STATUS PANELS — phased loading meter with context notes    */
/* and a failure surface offering both recovery paths                           */
/* ============================================================================ */
function LoadingPanel({ progress, message, note, pulse }: { progress: number; message: string; note?: string; pulse?: boolean }) {
  return (
    <div className="anim-fade-up mx-auto mt-16 max-w-md">
      <div className="panel px-6 py-8">
        <div className="flex items-center gap-3">
          <span className="grid h-10 w-10 place-items-center rounded-lg border border-teal/35 bg-teal/10 text-teal">
            <IconRefresh size={18} className="anim-spin" />
          </span>
          <div>
            <p className="font-display text-[16px] font-bold text-ink">Prediction pipeline</p>
            <p className="font-mono text-[11.5px] text-muted">{message}</p>
          </div>
        </div>
        <div className="mt-5 h-2 overflow-hidden rounded-full bg-canvas">
          <div
            className={`h-full rounded-full bg-gradient-to-r from-teal via-amber to-rose transition-[width] duration-300 ease-out ${pulse ? "animate-pulse" : ""}`}
            style={{ width: `${Math.round(progress * 100)}%` }}
          />
        </div>
        <ol className="mt-5 space-y-1.5 font-mono text-[11px]">
          {["Acquire dataset", "Clean & engineer features", "Fit boosted ensemble", "Score & decompose"].map((step, i) => {
            const done = progress >= (i + 1) / 4;
            const active = !done && progress >= i / 4;
            return (
              <li key={step} className={`flex items-center gap-2 ${done ? "text-teal" : active ? "text-ink" : "text-faint"}`}>
                <span className={`grid h-4 w-4 place-items-center rounded-full border text-[9px] ${done ? "border-teal/60 bg-teal/15" : active ? "border-amber/60 bg-amber/15 text-amber" : "border-line"}`}>
                  {done ? "✓" : i + 1}
                </span>
                {step}
              </li>
            );
          })}
        </ol>
        {note && <p className="mt-4 text-[11.5px] leading-relaxed text-faint">{note}</p>}
      </div>
    </div>
  );
}

function ErrorPanel({ message, onRetry, onFallback }: { message: string; onRetry: () => void; onFallback: () => void }) {
  return (
    <div className="anim-fade-up mx-auto mt-16 max-w-md">
      <div className="panel border-rose/35 px-6 py-8">
        <p className="font-display text-[16px] font-bold text-rose">Pipeline failed</p>
        <p className="mt-2 text-[13px] leading-relaxed text-muted">{message}</p>
        <div className="mt-5 flex flex-wrap gap-2">
          <button
            onClick={onRetry}
            className="rounded-md border border-teal/45 bg-teal/12 px-3.5 py-2 font-mono text-[12px] font-semibold text-teal transition-all hover:bg-teal/22 active:scale-[0.97]"
          >
            Retry backend probe
          </button>
          <button
            onClick={onFallback}
            className="rounded-md border border-line px-3.5 py-2 font-mono text-[12px] text-muted transition-all hover:border-faint hover:text-ink active:scale-[0.97]"
          >
            Run in-browser engine
          </button>
        </div>
      </div>
    </div>
  );
}
