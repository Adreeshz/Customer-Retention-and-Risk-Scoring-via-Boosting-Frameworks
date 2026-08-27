import { createRng, randInt, clamp } from "./random";

/* ========================================================================= */
/* SECTION: Types & Interfaces                                               */
/* ========================================================================= */

export interface GBDTConfig {
  estimators: number;
  learningRate: number;
  maxDepth: number;
  minLeaf: number;
  subsample: number;
  lambda: number;
  seed: number;
}

export interface Node {
  feature: number;
  threshold: number;
  left: number;
  right: number;
  out: number;
  gain?: number; // Internal usage for feature importance
  val?: number;  // Internal usage for telescoping explanations
}

export interface Tree {
  nodes: Node[];
}

export interface GBDT {
  base: number;
  trees: Tree[];
  config: GBDTConfig;
}

/* ========================================================================= */
/* SECTION: Math Utils                                                       */
/* ========================================================================= */

export function sigmoid(x: number): number {
  if (x < -20) return 0.0;
  if (x > 20) return 1.0;
  return 1 / (1 + Math.exp(-x));
}

export function stratifiedSplit(labels: Uint8Array, testFraction: number, seed: number): { train: number[], test: number[] } {
  const rng = createRng(seed);
  const pos: number[] = [];
  const neg: number[] = [];

  for (let i = 0; i < labels.length; i++) {
    if (labels[i] === 1) pos.push(i);
    else neg.push(i);
  }

  // Fisher-Yates shuffle
  const shuffle = (arr: number[]) => {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = randInt(rng, 0, i);
      const temp = arr[i];
      arr[i] = arr[j];
      arr[j] = temp;
    }
  };

  shuffle(pos);
  shuffle(neg);

  const posTestCount = Math.floor(pos.length * testFraction);
  const negTestCount = Math.floor(neg.length * testFraction);

  const test = [...pos.slice(0, posTestCount), ...neg.slice(0, negTestCount)];
  const train = [...pos.slice(posTestCount), ...neg.slice(negTestCount)];

  shuffle(test);
  shuffle(train);

  return { train, test };
}

/* ========================================================================= */
/* SECTION: Training                                                         */
/* ========================================================================= */

export async function fitGBDT(
  columns: Float64Array[],
  labels: Uint8Array,
  config: GBDTConfig,
  onProgress?: (fraction: number) => void
): Promise<GBDT> {
  const nSamples = labels.length;
  const rng = createRng(config.seed);

  let posCount = 0;
  for (let i = 0; i < nSamples; i++) posCount += labels[i];
  
  const p = clamp(posCount / nSamples, 1e-6, 1 - 1e-6);
  const base = Math.log(p / (1 - p));

  const preds = new Float64Array(nSamples);
  for (let i = 0; i < nSamples; i++) preds[i] = base;

  const trees: Tree[] = [];
  const allIndices = new Int32Array(nSamples);
  for (let i = 0; i < nSamples; i++) allIndices[i] = i;

  const sampleSize = Math.max(1, Math.floor(nSamples * config.subsample));
  const grads = new Float64Array(nSamples);
  const hesss = new Float64Array(nSamples);

  for (let t = 0; t < config.estimators; t++) {
    // 1. Calculate gradients and hessians for logistic loss
    for (let i = 0; i < nSamples; i++) {
      const prob = sigmoid(preds[i]);
      grads[i] = prob - labels[i]; 
      hesss[i] = prob * (1 - prob);
    }

    // 2. Subsample
    for (let i = 0; i < sampleSize; i++) {
      const j = randInt(rng, i, nSamples - 1);
      const temp = allIndices[i];
      allIndices[i] = allIndices[j];
      allIndices[j] = temp;
    }
    const sampleIndices = allIndices.slice(0, sampleSize);

    // 3. Build tree
    const tree = buildTree(columns, grads, hesss, sampleIndices, config);
    trees.push(tree);

    // 4. Update predictions with learning rate
    for (let i = 0; i < nSamples; i++) {
      preds[i] += predictTree(tree, columns, i) * config.learningRate;
    }

    // 5. Yield progress
    if (onProgress && (t + 1) % 10 === 0) {
      onProgress((t + 1) / config.estimators);
      await new Promise(r => setTimeout(r, 0));
    }
  }

  if (onProgress) onProgress(1.0);
  return { base, trees, config };
}

interface BuildQueueItem {
  indices: Int32Array;
  depth: number;
  nodeIdx: number;
}

function buildTree(
  columns: Float64Array[],
  grads: Float64Array,
  hesss: Float64Array,
  indices: Int32Array,
  config: GBDTConfig
): Tree {
  const nodes: Node[] = [];
  const q: BuildQueueItem[] = [{ indices, depth: 0, nodeIdx: 0 }];
  nodes.push({ feature: -1, threshold: 0, left: -1, right: -1, out: 0 });

  while (q.length > 0) {
    const { indices: nodeIndices, depth, nodeIdx } = q.shift()!;
    let sumG = 0, sumH = 0;
    
    for (let i = 0; i < nodeIndices.length; i++) {
      sumG += grads[nodeIndices[i]];
      sumH += hesss[nodeIndices[i]];
    }

    // Leaf / internal value according to Newton step
    const val = -sumG / (sumH + config.lambda);
    nodes[nodeIdx].val = val;

    if (depth >= config.maxDepth || nodeIndices.length < config.minLeaf * 2) {
      nodes[nodeIdx].out = val;
      continue;
    }

    let bestGain = -Infinity;
    let bestFeature = -1;
    let bestThreshold = 0;
    let bestLeftIndices = new Int32Array(0);
    let bestRightIndices = new Int32Array(0);

    for (let f = 0; f < columns.length; f++) {
      const col = columns[f];
      
      // We sort the indices by the feature values
      const sorted = Array.from(nodeIndices).sort((a, b) => col[a] - col[b]);
      
      let gL = 0, hL = 0;
      for (let i = 0; i < sorted.length - 1; i++) {
        const idx = sorted[i];
        gL += grads[idx];
        hL += hesss[idx];

        const gR = sumG - gL;
        const hR = sumH - hL;

        const leftCount = i + 1;
        const rightCount = sorted.length - leftCount;

        // Ensure distinct threshold
        if (col[sorted[i]] !== col[sorted[i + 1]] && leftCount >= config.minLeaf && rightCount >= config.minLeaf) {
          const gain = (gL * gL) / (hL + config.lambda) + 
                       (gR * gR) / (hR + config.lambda) - 
                       (sumG * sumG) / (sumH + config.lambda);

          if (gain > bestGain) {
            bestGain = gain;
            bestFeature = f;
            bestThreshold = (col[sorted[i]] + col[sorted[i + 1]]) / 2;
            bestLeftIndices = new Int32Array(sorted.slice(0, leftCount));
            bestRightIndices = new Int32Array(sorted.slice(leftCount));
          }
        }
      }
    }

    if (bestFeature !== -1 && bestGain > 0) {
      nodes[nodeIdx].feature = bestFeature;
      nodes[nodeIdx].threshold = bestThreshold;
      nodes[nodeIdx].gain = bestGain;

      const leftIdx = nodes.length;
      nodes.push({ feature: -1, threshold: 0, left: -1, right: -1, out: 0 });
      nodes[nodeIdx].left = leftIdx;

      const rightIdx = nodes.length;
      nodes.push({ feature: -1, threshold: 0, left: -1, right: -1, out: 0 });
      nodes[nodeIdx].right = rightIdx;

      q.push({ indices: bestLeftIndices, depth: depth + 1, nodeIdx: leftIdx });
      q.push({ indices: bestRightIndices, depth: depth + 1, nodeIdx: rightIdx });
    } else {
      nodes[nodeIdx].out = val;
    }
  }

  return { nodes };
}

function predictTree(tree: Tree, columns: Float64Array[], row: number): number {
  let nodeIdx = 0;
  while (tree.nodes[nodeIdx].feature !== -1) {
    const node = tree.nodes[nodeIdx];
    const val = columns[node.feature][row];
    if (val <= node.threshold) {
      nodeIdx = node.left;
    } else {
      nodeIdx = node.right;
    }
  }
  return tree.nodes[nodeIdx].out;
}

/* ========================================================================= */
/* SECTION: Inference & Explanations                                         */
/* ========================================================================= */

export function explain(gbdt: GBDT, vector: Float64Array): { base: number, contribs: number[], probability: number } {
  const contribs = new Array(vector.length).fill(0);
  let logOdds = gbdt.base;

  for (const tree of gbdt.trees) {
    let nodeIdx = 0;
    let currentVal = tree.nodes[0].val || 0;

    while (tree.nodes[nodeIdx].feature !== -1) {
      const node = tree.nodes[nodeIdx];
      const featureVal = vector[node.feature];
      
      const nextIdx = featureVal <= node.threshold ? node.left : node.right;
      const nextNode = tree.nodes[nextIdx];
      const nextVal = nextNode.val || nextNode.out; // Out is fallback for leaf

      // Telescoping sum
      contribs[node.feature] += (nextVal - currentVal) * gbdt.config.learningRate;
      currentVal = nextVal;
      
      nodeIdx = nextIdx;
    }
    logOdds += (tree.nodes[nodeIdx].out * gbdt.config.learningRate);
  }

  return {
    base: gbdt.base,
    contribs,
    probability: sigmoid(logOdds)
  };
}

export function normalizedImportance(gbdt: GBDT): number[] {
  let maxFeature = -1;
  for (const tree of gbdt.trees) {
    for (const node of tree.nodes) {
      if (node.feature > maxFeature) {
        maxFeature = node.feature;
      }
    }
  }
  
  if (maxFeature === -1) return [];

  const gains = new Array(maxFeature + 1).fill(0);
  let totalGain = 0;

  for (const tree of gbdt.trees) {
    for (const node of tree.nodes) {
      if (node.feature !== -1 && node.gain) {
        gains[node.feature] += node.gain;
        totalGain += node.gain;
      }
    }
  }

  if (totalGain === 0) return gains;

  for (let i = 0; i < gains.length; i++) {
    gains[i] /= totalGain;
  }

  return gains;
}
