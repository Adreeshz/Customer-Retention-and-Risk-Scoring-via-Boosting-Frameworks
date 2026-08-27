import type { Contract, Customer, InternetService, PaymentMethod } from "./dataset";
import { featureVector, featureVectorFromCustomer } from "./dataset";
import { explain } from "./boosting";
import type { Band, Point, RankedScores, TenureBucket } from "./analytics";
import { bandOf } from "./analytics";
import type { TrainedModel } from "./browserTraining";
import { trainBrowserModel } from "./browserTraining";

/* ============================================================================ */
/* SECTION: SYSTEM CONTRACT — the single interface both execution modes expose  */
/* to the UI: "api" mode talks to the Python backend over REST, "browser" mode  */
/* runs the TypeScript fallback engine locally. Components never know which     */
/* mode they are rendering, which keeps the two tiers disparity-free            */
/* ============================================================================ */
export type SortKey = "risk-desc" | "risk-asc" | "tenure-asc" | "charges-desc" | "tickets-desc";

export interface RowQuery {
  search: string;
  band: "all" | Band;
  contract: string;
  internet: string;
  sort: SortKey;
  page: number;
  size: number;
}

export interface ScoredRow {
  key: string;
  customer: Customer;
  prob: number;
  base: number;
  contribs: number[];
}

export interface PageResult {
  rows: ScoredRow[];
  total: number;
  pages: number;
}

export interface SimProfile {
  tenure: number;
  monthlyCharges: number;
  contract: string;
  internet: string;
  payment: string;
  techSupport: boolean;
  onlineSecurity: boolean;
  streamingTV: boolean;
  paperless: boolean;
  risingBill: boolean;
  tickets: number;
  senior: boolean;
  dependents: boolean;
}

export interface SimResult {
  probability: number;
  base: number;
  contribs: number[];
}

export interface FrameworkStat {
  name: string;
  rocAuc: number;
  prAuc: number;
  f1: number;
  active: boolean;
}

export interface SystemInfo {
  mode: "api" | "browser";
  engine: string;
  datasetSource: string;
  datasetRows: number;
  seed: number;
  frameworks: FrameworkStat[];
  trees: number;
}

export interface ModelSummary {
  curves: RankedScores;
  importance: number[];
  tenure: TenureBucket[];
  churnRate: number;
  trainedMs: number;
  seed: number;
  trainRows: number;
  testRows: number;
  trees: number;
}

export interface Adapter {
  info: SystemInfo;
  featureNames: readonly string[];
  summary: ModelSummary;
  testScores: { labels: number[]; scores: number[] };
  cohortAvg: number;
  totalCount: number;
  fetchRows(query: RowQuery): Promise<PageResult>;
  simulate(profile: SimProfile): Promise<SimResult>;
  retrain(seed: number): Promise<Adapter>;
}

/* ============================================================================ */
/* SECTION: BROWSER ADAPTER — wraps a locally trained model behind the same     */
/* contract; filtering and sorting mirror the backend query endpoint exactly    */
/* ============================================================================ */
export function createBrowserAdapter(model: TrainedModel): Adapter {
  const info: SystemInfo = {
    mode: "browser",
    engine: "In-browser GBDT (TypeScript)",
    datasetSource: "Synthetic Telco mirror (in-code)",
    datasetRows: model.customers.length,
    seed: model.seed,
    frameworks: [
      {
        name: "In-browser GBDT",
        rocAuc: model.curves.rocAuc,
        prAuc: model.curves.prAuc,
        f1: model.confusion.f1,
        active: true,
      },
    ],
    trees: model.gbdt.trees.length,
  };

  let cohortSum = 0;
  for (let i = 0; i < model.probabilities.length; i++) cohortSum += model.probabilities[i];
  const cohortAvg = cohortSum / Math.max(1, model.probabilities.length);

  return {
    info,
    featureNames: model.matrix.names,
    summary: {
      curves: model.curves,
      importance: model.importance,
      tenure: model.tenure,
      churnRate: model.churnRate,
      trainedMs: model.trainedMs,
      seed: model.seed,
      trainRows: model.trainIndex.length,
      testRows: model.testIndex.length,
      trees: model.gbdt.trees.length,
    },
    testScores: {
      labels: model.testIndex.map((i) => (model.customers[i].churned ? 1 : 0)),
      scores: model.testIndex.map((i) => model.probabilities[i]),
    },
    cohortAvg,
    totalCount: model.customers.length,

    async fetchRows(query: RowQuery): Promise<PageResult> {
      const matched = filterAndSort(model, query);
      const pages = Math.max(1, Math.ceil(matched.length / query.size));
      const safePage = Math.min(query.page, pages - 1);
      const slice = matched.slice(safePage * query.size, safePage * query.size + query.size);
      const rows: ScoredRow[] = slice.map((entry) => {
        const ex = explain(model.gbdt, featureVector(model.matrix, entry.index));
        return {
          key: entry.customer.id,
          customer: entry.customer,
          prob: entry.prob,
          base: ex.base,
          contribs: ex.contribs,
        };
      });
      return { rows, total: matched.length, pages };
    },

    async simulate(profile: SimProfile): Promise<SimResult> {
      const customer = customerFromProfile(profile);
      const ex = explain(model.gbdt, featureVectorFromCustomer(customer));
      return { probability: ex.probability, base: ex.base, contribs: ex.contribs };
    },

    async retrain(seed: number): Promise<Adapter> {
      const next = await trainBrowserModel(seed, () => undefined);
      return createBrowserAdapter(next);
    },
  };
}

/* ============================================================================ */
/* SECTION: QUERY ENGINE — identical predicate and ordering semantics to        */
/* backend/model.py query(), so pagination behaves the same in both modes       */
/* ============================================================================ */
function filterAndSort(model: TrainedModel, query: RowQuery): { customer: Customer; index: number; prob: number }[] {
  const needle = query.search.trim().toLowerCase();
  const rows: { customer: Customer; index: number; prob: number }[] = [];

  for (let i = 0; i < model.customers.length; i++) {
    const c = model.customers[i];
    const prob = model.probabilities[i];
    if (needle && !c.id.toLowerCase().includes(needle) && !c.payment.toLowerCase().includes(needle)) continue;
    if (query.band !== "all" && bandOf(prob).band !== query.band) continue;
    if (query.contract !== "all" && c.contract !== query.contract) continue;
    if (query.internet !== "all" && c.internet !== query.internet) continue;
    rows.push({ customer: c, index: i, prob });
  }

  rows.sort((a, b) => {
    let delta = 0;
    switch (query.sort) {
      case "risk-desc":
        delta = b.prob - a.prob;
        break;
      case "risk-asc":
        delta = a.prob - b.prob;
        break;
      case "tenure-asc":
        delta = a.customer.tenure - b.customer.tenure;
        break;
      case "charges-desc":
        delta = b.customer.monthlyCharges - a.customer.monthlyCharges;
        break;
      case "tickets-desc":
        delta = b.customer.tickets - a.customer.tickets;
        break;
    }
    return delta !== 0 ? delta : a.customer.id.localeCompare(b.customer.id);
  });
  return rows;
}

/* ============================================================================ */
/* SECTION: PROFILE SYNTHESIS — builds the simulated customer the in-browser    */
/* engine scores; billing derivation matches backend/features.py exactly        */
/* ============================================================================ */
function customerFromProfile(p: SimProfile): Customer {
  const drift = p.risingBill ? 0.9 : 1;
  return {
    id: "SIM-PROFILE",
    gender: "Female",
    senior: p.senior ? 1 : 0,
    dependents: p.dependents,
    partner: false,
    tenure: p.tenure,
    contract: p.contract as Contract,
    paperless: p.paperless,
    payment: p.payment as PaymentMethod,
    internet: p.internet as InternetService,
    techSupport: p.techSupport,
    onlineSecurity: p.onlineSecurity,
    streamingTV: p.streamingTV,
    tickets: p.tickets,
    monthlyCharges: p.monthlyCharges,
    totalCharges: p.tenure > 0 ? p.monthlyCharges * p.tenure * drift : 0,
    avgMonthly: p.tenure > 0 ? p.monthlyCharges * drift : p.monthlyCharges,
    risingBill: p.risingBill,
    churned: false,
  };
}

/* ============================================================================ */
/* SECTION: SHARED POINT TYPE RE-EXPORT — keeps chart imports stable regardless */
/* of which layer a component already depends on                                */
/* ============================================================================ */
export type { Point };
