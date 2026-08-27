import { buildFeatureMatrix, featureVector, generateCustomers } from "./dataset";
import type { Customer, FeatureMatrix } from "./dataset";
import { fitGBDT, normalizedImportance, sigmoid, stratifiedSplit } from "./boosting";
import type { GBDT } from "./boosting";
import { confusionAt, rankScores, tenureBuckets } from "./analytics";
import type { Confusion, Point, RankedScores, TenureBucket } from "./analytics";

/* ============================================================================ */
/* SECTION: TRAINED MODEL SHAPE — everything the in-browser engine produces in  */
/* one pipeline run: cohort, matrices, ensemble and evaluation artifacts        */
/* ============================================================================ */
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

/* ============================================================================ */
/* SECTION: IN-BROWSER TRAINING PIPELINE — generates the synthetic Telco mirror */
/* cohort, fits the boosted ensemble in UI-friendly chunks and scores every     */
/* customer; this engine is the offline fallback when the Python backend is     */
/* not running, and it mirrors backend/features.py feature-for-feature          */
/* ============================================================================ */
export async function trainBrowserModel(
  seed: number,
  onStage: (progress: number, message: string) => void,
): Promise<TrainedModel> {
  onStage(0.04, "Synthesizing 2,400 customer records");
  await tick();
  const customers = generateCustomers(seed, 2400);
  const matrix = buildFeatureMatrix(customers);
  const labels = Uint8Array.from(customers, (c) => (c.churned ? 1 : 0));

  onStage(0.1, "Fitting 150 boosted trees (depth 4)");
  const split = stratifiedSplit(labels, 0.2, seed);
  const trainCols = matrix.columns.map((col) => Float64Array.from(split.train, (i) => col[i]));
  const trainY = Uint8Array.from(split.train, (i) => labels[i]);

  const started = performance.now();
  const gbdt = await fitGBDT(
    trainCols,
    trainY,
    { estimators: 150, learningRate: 0.085, maxDepth: 4, minLeaf: 10, subsample: 0.85, lambda: 1.2, seed },
    (fraction) => onStage(0.1 + fraction * 0.72, `Fitting boosted trees — tree ${Math.round(fraction * 150)} of 150`),
  );
  const trainedMs = Math.round(performance.now() - started);

  onStage(0.88, "Scoring customers and calibrating curves");
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
    probabilities[i] = sigmoid(logit);
  }

  const testLabels = split.test.map((i) => labels[i]);
  const testScores = split.test.map((i) => probabilities[i]);
  const curves = rankScores(testLabels, testScores);
  const confusion = confusionAt(testLabels, testScores, 0.5);
  const importance = normalizedImportance(gbdt);
  const tenure = tenureBuckets(customers);
  const churnRate = customers.filter((c) => c.churned).length / Math.max(1, customers.length);

  return {
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
    seed,
  };
}

function tick(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 30));
}

/* ============================================================================ */
/* SECTION: CURVE RESAMPLING — maps normalized curve points onto SVG space so   */
/* chart components render fixed-size paths without re-deriving geometry        */
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
