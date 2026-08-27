import { useCallback, useEffect, useRef, useState } from "react";
import { buildFeatureMatrix, generateCustomers, featureVector } from "../lib/dataset";
import type { Customer, FeatureMatrix } from "../lib/dataset";
import { fitGBDT, normalizedImportance, stratifiedSplit } from "../lib/boosting";
import type { GBDT } from "../lib/boosting";
import { confusionAt, rankScores, tenureBuckets } from "../lib/analytics";
import type { Confusion, Point, RankedScores, TenureBucket } from "../lib/analytics";

/* ============================================================================ */
/* SECTION: PIPELINE STATE — the full status surface exposed to the UI layer    */
/* ============================================================================ */
export type PipelineStatus = "generating" | "training" | "scoring" | "ready" | "error";

export interface TrainedModel {
  customers: Customer[];
  matrix: FeatureMatrix;
  gbdt: GBDT;
  probabilities: Float64Array;
  trainIndex: number[];
  testIndex: number[];
  curves: RankedScores;
  confusion: Confusion;
  importance: number[];
  tenure: TenureBucket[];
  churnRate: number;
  trainedMs: number;
  seed: number;
}

export interface ModelState {
  status: PipelineStatus;
  progress: number;
  message: string;
  error: string | null;
  model: TrainedModel | null;
}

/* ============================================================================ */
/* SECTION: PIPELINE HOOK — generates the cohort, fits the boosted ensemble in  */
/* UI-friendly chunks, then scores every customer and precomputes evaluation    */
/* ============================================================================ */
export function useModel(initialSeed = 42) {
  const [state, setState] = useState<ModelState>({
    status: "generating",
    progress: 0,
    message: "Preparing workspace",
    error: null,
    model: null,
  });
  const [seed, setSeed] = useState(initialSeed);
  const runId = useRef(0);

  const run = useCallback(async (activeSeed: number) => {
    const id = ++runId.current;
    const alive = () => runId.current === id;
    const patch = (partial: Partial<ModelState>) => {
      if (alive()) setState((prev) => ({ ...prev, ...partial }));
    };

    try {
      patch({ status: "generating", progress: 0.04, message: "Synthesizing 2,400 customer records", error: null });
      await tick();
      const customers = generateCustomers(activeSeed, 2400);
      const matrix = buildFeatureMatrix(customers);
      const labels = Uint8Array.from(customers, (c) => (c.churned ? 1 : 0));

      patch({ status: "training", progress: 0.1, message: "Fitting 150 boosted trees (depth 4)" });
      const split = stratifiedSplit(labels, 0.2, activeSeed);
      const trainCols = matrix.columns.map((col) => Float64Array.from(split.train, (i) => col[i]));
      const trainY = Uint8Array.from(split.train, (i) => labels[i]);

      const started = performance.now();
      const gbdt = await fitGBDT(
        trainCols,
        trainY,
        { estimators: 150, learningRate: 0.085, maxDepth: 4, minLeaf: 10, subsample: 0.85, lambda: 1.2, seed: activeSeed },
        (fraction) => patch({ progress: 0.1 + fraction * 0.72, message: `Fitting boosted trees — tree ${Math.round(fraction * 150)} of 150` }),
      );
      const trainedMs = Math.round(performance.now() - started);

      patch({ status: "scoring", progress: 0.88, message: "Scoring customers and calibrating curves" });
      await tick();

      const probabilities = new Float64Array(customers.length);
      for (let i = 0; i < customers.length; i++) {
        const vector = featureVector(matrix, i);
        let logit = gbdt.base;
        for (const tree of gbdt.trees) {
          let idx = 0;
          for (;;) {
            const node = tree.nodes[idx];
            if (node.feature < 0) break;
            idx = vector[node.feature] <= node.threshold ? node.left : node.right;
          }
          logit += tree.nodes[idx].out * gbdt.config.learningRate;
        }
        probabilities[i] = 1 / (1 + Math.exp(-logit));
      }

      const testLabels = split.test.map((i) => labels[i]);
      const testScores = split.test.map((i) => probabilities[i]);
      const curves = rankScores(testLabels, testScores);
      const confusion = confusionAt(testLabels, testScores, 0.5);
      const importance = normalizedImportance(gbdt);
      const tenure = tenureBuckets(customers);
      const churnRate = customers.filter((c) => c.churned).length / Math.max(1, customers.length);

      patch({
        status: "ready",
        progress: 1,
        message: "Model ready",
        model: {
          customers,
          matrix,
          gbdt,
          probabilities,
          trainIndex: split.train,
          testIndex: split.test,
          curves,
          confusion,
          importance,
          tenure,
          churnRate,
          trainedMs,
          seed: activeSeed,
        },
      });
    } catch (err) {
      patch({
        status: "error",
        message: "Pipeline failed",
        error: err instanceof Error ? err.message : "Unknown training failure",
      });
    }
  }, []);

  useEffect(() => {
    void run(seed);
  }, [seed, run]);

  const retrain = useCallback((nextSeed: number) => setSeed(nextSeed), []);

  return { state, seed, retrain };
}

function tick(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 30));
}

/* ============================================================================ */
/* SECTION: CURVE RESAMPLING — helper so charts can render fixed-size paths     */
/* without re-deriving them on every paint                                      */
/* ============================================================================ */
export function curvePath(points: Point[], width: number, height: number, pad = 6): string {
  if (points.length === 0) return "";
  return points
    .map((p, i) => {
      const x = pad + p.x * (width - pad * 2);
      const y = height - pad - p.y * (height - pad * 2);
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}
