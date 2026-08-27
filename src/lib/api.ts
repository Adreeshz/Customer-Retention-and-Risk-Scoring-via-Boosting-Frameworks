import type { Customer, PaymentMethod } from "./dataset";
import type { Point } from "./analytics";
import type { Adapter, FrameworkStat, ModelSummary, PageResult, RowQuery, ScoredRow, SimProfile, SimResult, SystemInfo } from "./adapter";

/* ============================================================================ */
/* SECTION: API BASE RESOLUTION — defaults to the local FastAPI service and can */
/* be overridden per-browser via localStorage for remote deployments            */
/* ============================================================================ */
const DEFAULT_API_BASE = "http://127.0.0.1:8000";

function apiBase(): string {
  try {
    const stored = localStorage.getItem("churnlens:apiBase");
    if (stored) return stored.replace(/\/+$/, "");
  } catch {
    /* storage unavailable — fall through to the default origin */
  }
  return DEFAULT_API_BASE;
}

/* ============================================================================ */
/* SECTION: TRANSPORT — fetch wrapper with per-call timeouts and error mapping  */
/* ============================================================================ */
async function fetchJson<T>(path: string, init?: RequestInit & { timeoutMs?: number }): Promise<T> {
  const { timeoutMs = 30000, ...rest } = init ?? {};
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${apiBase()}${path}`, {
      ...rest,
      signal: controller.signal,
      headers: { "Content-Type": "application/json", ...(rest.headers ?? {}) },
    });
    if (!response.ok) {
      throw new Error(`Backend responded with ${response.status} ${response.statusText}`);
    }
    return (await response.json()) as T;
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error("Backend request timed out");
    }
    throw err;
  } finally {
    window.clearTimeout(timer);
  }
}

/* ============================================================================ */
/* SECTION: RESPONSE SCHEMAS — mirror the FastAPI payload contracts declared in */
/* backend/main.py; every field is validated while mapping into domain types    */
/* ============================================================================ */
interface RawCustomer {
  id: string;
  gender: string;
  senior: number;
  dependents: boolean;
  partner: boolean;
  tenure: number;
  contract: string;
  paperless: boolean;
  payment: string;
  internet: string;
  techSupport: boolean;
  onlineSecurity: boolean;
  streamingTV: boolean;
  tickets: number;
  monthlyCharges: number;
  totalCharges: number;
  avgMonthly: number;
  risingBill: boolean;
  churned: boolean;
}

interface ApiStatusPayload {
  engine: string;
  datasetSource: string;
  datasetRows: number;
  featureCount: number;
  featureNames: string[];
  seed: number;
  trainedAt: string;
  trees: number;
  frameworks: FrameworkStat[];
}

interface ApiMetricsPayload {
  rocAuc: number;
  prAuc: number;
  curves: { roc: Point[]; pr: Point[] };
  importance: number[];
  tenureCurve: { label: string; stayed: number; churned: number; rate: number }[];
  classBalance: { retained: number; churned: number };
  avgProb: number;
  churnRate: number;
  testScores: { p: number[]; y: number[] };
  trainRows: number;
  testRows: number;
  trainedMs: number;
  seed: number;
  trees: number;
}

interface ApiCustomersPayload {
  total: number;
  page: number;
  pages: number;
  rows: { id: string; probability: number; base: number; contribs: number[]; customer: RawCustomer }[];
}

/* ============================================================================ */
/* SECTION: ENDPOINT CALLS — one typed function per backend route               */
/* ============================================================================ */
export async function probeBackend(): Promise<boolean> {
  try {
    await fetchJson<{ status: string }>("/api/health", { timeoutMs: 2500 });
    return true;
  } catch {
    return false;
  }
}

export function fetchStatus(): Promise<ApiStatusPayload> {
  return fetchJson<ApiStatusPayload>("/api/status");
}

export function fetchMetrics(): Promise<ApiMetricsPayload> {
  return fetchJson<ApiMetricsPayload>("/api/metrics");
}

export async function fetchCustomers(query: RowQuery): Promise<PageResult> {
  const params = new URLSearchParams({
    search: query.search,
    band: query.band,
    contract: query.contract,
    internet: query.internet,
    sort: query.sort,
    page: String(query.page),
    size: String(query.size),
  });
  const payload = await fetchJson<ApiCustomersPayload>(`/api/customers?${params.toString()}`);
  return {
    total: payload.total,
    pages: payload.pages,
    rows: payload.rows.map(
      (row): ScoredRow => ({
        key: row.id,
        customer: toCustomer(row.customer),
        prob: row.probability,
        base: row.base,
        contribs: row.contribs,
      }),
    ),
  };
}

export function postSimulate(profile: SimProfile): Promise<SimResult> {
  return fetchJson<SimResult>("/api/simulate", {
    method: "POST",
    body: JSON.stringify({
      tenure: profile.tenure,
      monthly_charges: profile.monthlyCharges,
      contract: profile.contract,
      internet_service: profile.internet,
      payment_method: profile.payment,
      tech_support: profile.techSupport,
      online_security: profile.onlineSecurity,
      streaming_tv: profile.streamingTV,
      paperless_billing: profile.paperless,
      rising_bill: profile.risingBill,
      tickets: profile.tickets,
      senior: profile.senior,
      dependents: profile.dependents,
    }),
  });
}

export function postRetrain(seed: number): Promise<{ ok: boolean; seed: number; rocAuc: number }> {
  return fetchJson<{ ok: boolean; seed: number; rocAuc: number }>("/api/retrain", {
    method: "POST",
    body: JSON.stringify({ seed }),
  });
}

/* ============================================================================ */
/* SECTION: ADAPTER FACTORY — binds the REST client to the shared Adapter       */
/* contract so components consume API mode and browser mode identically         */
/* ============================================================================ */
export function createApiAdapter(status: ApiStatusPayload, metrics: ApiMetricsPayload): Adapter {
  const info: SystemInfo = {
    mode: "api",
    engine: status.engine,
    datasetSource: status.datasetSource,
    datasetRows: status.datasetRows,
    seed: status.seed,
    frameworks: status.frameworks,
    trees: status.trees,
  };

  const summary: ModelSummary = {
    curves: {
      roc: metrics.curves.roc,
      pr: metrics.curves.pr,
      rocAuc: metrics.rocAuc,
      prAuc: metrics.prAuc,
    },
    importance: metrics.importance,
    tenure: metrics.tenureCurve,
    churnRate: metrics.churnRate,
    trainedMs: metrics.trainedMs,
    seed: metrics.seed,
    trainRows: metrics.trainRows,
    testRows: metrics.testRows,
    trees: metrics.trees,
  };

  return {
    info,
    featureNames: status.featureNames,
    summary,
    testScores: { labels: metrics.testScores.y, scores: metrics.testScores.p },
    cohortAvg: metrics.avgProb,
    totalCount: status.datasetRows,
    fetchRows: (query) => fetchCustomers(query),
    simulate: (profile) => postSimulate(profile),
    async retrain(seed: number): Promise<Adapter> {
      await postRetrain(seed);
      const [nextStatus, nextMetrics] = await Promise.all([fetchStatus(), fetchMetrics()]);
      return createApiAdapter(nextStatus, nextMetrics);
    },
  };
}

/* ============================================================================ */
/* SECTION: CUSTOMER COERCION — defensive mapping of raw JSON into the domain   */
/* type, clamping ranges so a malformed row can never break the table or drawer */
/* ============================================================================ */
function toCustomer(raw: RawCustomer): Customer {
  const clamp = (value: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, value));
  const contract = raw.contract === "One year" || raw.contract === "Two year" ? raw.contract : "Month-to-month";
  const internet = raw.internet === "DSL" || raw.internet === "Fiber optic" ? raw.internet : "None";
  const payment = (["Mailed check", "Bank transfer", "Credit card"].includes(raw.payment) ? raw.payment : "Electronic check") as PaymentMethod;
  return {
    id: String(raw.id),
    gender: raw.gender === "Male" ? "Male" : "Female",
    senior: raw.senior ? 1 : 0,
    dependents: Boolean(raw.dependents),
    partner: Boolean(raw.partner),
    tenure: clamp(Math.round(Number(raw.tenure) || 0), 0, 72),
    contract,
    paperless: Boolean(raw.paperless),
    payment,
    internet,
    techSupport: Boolean(raw.techSupport),
    onlineSecurity: Boolean(raw.onlineSecurity),
    streamingTV: Boolean(raw.streamingTV),
    tickets: clamp(Math.round(Number(raw.tickets) || 0), 0, 12),
    monthlyCharges: Number(raw.monthlyCharges) || 0,
    totalCharges: Number(raw.totalCharges) || 0,
    avgMonthly: Number(raw.avgMonthly) || 0,
    risingBill: Boolean(raw.risingBill),
    churned: Boolean(raw.churned),
  };
}
