import { useCallback, useEffect, useRef, useState } from "react";
import type { Adapter } from "../lib/adapter";
import { createBrowserAdapter } from "../lib/adapter";
import { trainBrowserModel } from "../lib/browserTraining";
import { createApiAdapter, fetchMetrics, fetchStatus, probeBackend } from "../lib/api";
import { fmtPct } from "../lib/analytics";
import type { ToastKind } from "../components/ui";

/* ============================================================================ */
/* SECTION: SYSTEM PHASES — every state the two-tier bootstrap can be in, so    */
/* the shell always renders an honest surface (probing, loading, training,      */
/* failed or ready) instead of a blank screen                                   */
/* ============================================================================ */
export type SystemPhase =
  | { kind: "probing" }
  | { kind: "api-loading" }
  | { kind: "browser-training"; progress: number; message: string }
  | { kind: "error"; message: string }
  | { kind: "ready" };

/* ============================================================================ */
/* SECTION: SYSTEM HOOK — boots against the Python backend first and degrades   */
/* gracefully to the in-browser engine; also owns retraining for both modes     */
/* ============================================================================ */
export function useSystem(onEvent: (message: string, kind?: ToastKind) => void) {
  const [phase, setPhase] = useState<SystemPhase>({ kind: "probing" });
  const [adapter, setAdapter] = useState<Adapter | null>(null);
  const [seed, setSeed] = useState(42);
  const runId = useRef(0);

  /* ------------------------------------------------------------------------ */
  /* SECTION: BROWSER TRAINING RUNNER — reports staged progress to the phase   */
  /* ------------------------------------------------------------------------ */
  const trainInBrowser = useCallback(
    async (activeSeed: number, id: number): Promise<Adapter> => {
      const alive = () => runId.current === id;
      setPhase({ kind: "browser-training", progress: 0.04, message: "Preparing workspace" });
      const model = await trainBrowserModel(activeSeed, (progress, message) => {
        if (alive()) setPhase({ kind: "browser-training", progress, message });
      });
      if (!alive()) throw new Error("superseded");
      return createBrowserAdapter(model);
    },
    [],
  );

  /* ------------------------------------------------------------------------ */
  /* SECTION: BOOT SEQUENCE — probe the FastAPI service, load its artifacts,    */
  /* and fall back to local training when the backend is absent or failing     */
  /* ------------------------------------------------------------------------ */
  const boot = useCallback(
    async (forceBrowser: boolean) => {
      const id = ++runId.current;
      const alive = () => runId.current === id;

      if (!forceBrowser) {
        setPhase({ kind: "probing" });
        const reachable = await probeBackend();
        if (!alive()) return;

        if (reachable) {
          try {
            setPhase({ kind: "api-loading" });
            const [status, metrics] = await Promise.all([fetchStatus(), fetchMetrics()]);
            if (!alive()) return;
            setAdapter(createApiAdapter(status, metrics));
            setSeed(status.seed);
            setPhase({ kind: "ready" });
            onEvent(
              `Python backend connected — ${status.datasetRows.toLocaleString()} customers scored by ${status.engine}.`,
              "success",
            );
            return;
          } catch (err) {
            const detail = err instanceof Error ? err.message : "unknown failure";
            onEvent(`Backend reachable but failed to load (${detail}). Falling back to the in-browser engine.`, "error");
          }
        } else {
          onEvent("Python backend not reachable at 127.0.0.1:8000 — running the in-browser fallback engine.", "info");
        }
      }

      try {
        const next = await trainInBrowser(42, id);
        if (!alive()) return;
        setAdapter(next);
        setSeed(42);
        setPhase({ kind: "ready" });
      } catch (err) {
        if (err instanceof Error && err.message === "superseded") return;
        if (alive()) setPhase({ kind: "error", message: err instanceof Error ? err.message : "Training failed" });
      }
    },
    [onEvent, trainInBrowser],
  );

  useEffect(() => {
    void boot(false);
  }, [boot]);

  /* ------------------------------------------------------------------------ */
  /* SECTION: RETRAIN — refreshes whichever engine is active with a fresh seed */
  /* ------------------------------------------------------------------------ */
  const retrain = useCallback(
    async (nextSeed: number) => {
      const current = adapter;
      if (!current) return;
      const id = ++runId.current;
      const alive = () => runId.current === id;

      try {
        if (current.info.mode === "api") {
          setPhase({ kind: "api-loading" });
          const next = await current.retrain(nextSeed);
          if (!alive()) return;
          setAdapter(next);
          setSeed(nextSeed);
          setPhase({ kind: "ready" });
          onEvent(`Backend retrained with seed #${nextSeed} — ROC-AUC ${fmtPct(next.summary.curves.rocAuc)}.`, "success");
        } else {
          const next = await trainInBrowser(nextSeed, id);
          if (!alive()) return;
          setAdapter(next);
          setSeed(nextSeed);
          setPhase({ kind: "ready" });
          onEvent(`Browser engine retrained with seed #${nextSeed} — ROC-AUC ${fmtPct(next.summary.curves.rocAuc)}.`, "success");
        }
      } catch (err) {
        if (err instanceof Error && err.message === "superseded") return;
        if (alive()) setPhase({ kind: "error", message: err instanceof Error ? err.message : "Retraining failed" });
      }
    },
    [adapter, onEvent, trainInBrowser],
  );

  const retry = useCallback(() => void boot(false), [boot]);
  const fallbackToBrowser = useCallback(() => void boot(true), [boot]);

  return { phase, adapter, seed, retrain, retry, fallbackToBrowser };
}
