import { useEffect, useMemo, useState } from "react";
import type { TrainedModel } from "../hooks/useModel";
import { featureVector, type Customer } from "../lib/dataset";
import { explain } from "../lib/boosting";
import { bandOf, buildPlaybook, fmtMoney, fmtPct } from "../lib/analytics";
import type { Band } from "../lib/analytics";
import { RiskGauge, Waterfall } from "./charts";
import { Chip, EmptyState, IconArrowDown, IconArrowUp, IconSpark, IconX, SectionHead } from "./ui";

/* ============================================================================ */
/* SECTION: EXPLORER STATE — filter, sort and pagination definitions            */
/* ============================================================================ */
type SortKey = "risk-desc" | "risk-asc" | "tenure-asc" | "charges-desc" | "tickets-desc";
type BandFilter = "all" | Band;

interface Filters {
  q: string;
  band: BandFilter;
  contract: "all" | "Month-to-month" | "One year" | "Two year";
  internet: "all" | "None" | "DSL" | "Fiber optic";
  sort: SortKey;
}

const PAGE_SIZE = 9;

const SORT_OPTIONS: { value: SortKey; label: string }[] = [
  { value: "risk-desc", label: "Risk ↓" },
  { value: "risk-asc", label: "Risk ↑" },
  { value: "tenure-asc", label: "Tenure ↑" },
  { value: "charges-desc", label: "Charges ↓" },
  { value: "tickets-desc", label: "Tickets ↓" },
];

/* ============================================================================ */
/* SECTION: RISK EXPLORER WORKSPACE — searchable scored cohort with a per-row   */
/* detail drawer containing the gauge, SHAP waterfall and retention playbook    */
/* ============================================================================ */
export function Explorer({ model, notify }: { model: TrainedModel; notify: (msg: string, kind?: "success" | "info" | "error") => void }) {
  const [filters, setFilters] = useState<Filters>({ q: "", band: "all", contract: "all", internet: "all", sort: "risk-desc" });
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);

  const scored = useMemo(() => {
    const rows = model.customers.map((c, i) => ({
      customer: c,
      index: i,
      prob: model.probabilities[i],
    }));
    const q = filters.q.trim().toLowerCase();
    const filtered = rows.filter((r) => {
      if (q && !r.customer.id.toLowerCase().includes(q) && !r.customer.payment.toLowerCase().includes(q)) return false;
      if (filters.band !== "all" && bandOf(r.prob).band !== filters.band) return false;
      if (filters.contract !== "all" && r.customer.contract !== filters.contract) return false;
      if (filters.internet !== "all" && r.customer.internet !== filters.internet) return false;
      return true;
    });
    filtered.sort((a, b) => {
      switch (filters.sort) {
        case "risk-desc":
          return b.prob - a.prob;
        case "risk-asc":
          return a.prob - b.prob;
        case "tenure-asc":
          return a.customer.tenure - b.customer.tenure;
        case "charges-desc":
          return b.customer.monthlyCharges - a.customer.monthlyCharges;
        case "tickets-desc":
          return b.customer.tickets - a.customer.tickets;
      }
    });
    return filtered;
  }, [model, filters]);

  const pageCount = Math.max(1, Math.ceil(scored.length / PAGE_SIZE));
  const safePage = Math.min(page, pageCount - 1);
  const pageRows = scored.slice(safePage * PAGE_SIZE, safePage * PAGE_SIZE + PAGE_SIZE);
  const highRiskCount = useMemo(() => scored.filter((r) => r.prob >= 0.5).length, [scored]);

  useEffect(() => setPage(0), [filters.q, filters.band, filters.contract, filters.internet, filters.sort]);

  const set = (patch: Partial<Filters>) => setFilters((f) => ({ ...f, ...patch }));

  const exportFlagged = () => {
    const flagged = scored.filter((r) => r.prob >= 0.5);
    if (flagged.length === 0) {
      notify("No customers above the 50% risk line in the current view.", "error");
      return;
    }
    const header = "customer_id,tenure,contract,internet_service,monthly_charges,support_tickets,churn_probability,risk_band";
    const lines = flagged.map((r) =>
      [r.customer.id, r.customer.tenure, r.customer.contract, r.customer.internet, r.customer.monthlyCharges, r.customer.tickets, r.prob.toFixed(4), bandOf(r.prob).label].join(","),
    );
    const blob = new Blob([[header, ...lines].join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "churnlens_flagged_customers.csv";
    a.click();
    URL.revokeObjectURL(url);
    notify(`Exported ${flagged.length} flagged customers to CSV.`, "success");
  };

  return (
    <div className="anim-fade-up grid gap-3 xl:grid-cols-[300px_1fr]">
      <FilterPanel filters={filters} set={set} total={model.customers.length} matched={scored.length} highRisk={highRiskCount} onExport={exportFlagged} />

      <div className="panel overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-3">
          <SectionHead
            title="Scored Customers"
            sub={`${scored.length.toLocaleString()} matching · ${highRiskCount.toLocaleString()} above 50% risk`}
          />
          <div className="flex items-center gap-2">
            {(["all", "low", "watch", "high"] as BandFilter[]).map((b) => (
              <BandPill key={b} band={b} active={filters.band === b} onClick={() => set({ band: b })} />
            ))}
          </div>
        </div>

        {pageRows.length === 0 ? (
          <div className="p-6">
            <EmptyState
              title="No customers match these filters"
              hint="Loosen the search text or reset the band and service filters to see the full scored cohort again."
              action={
                <button
                  onClick={() => setFilters({ q: "", band: "all", contract: "all", internet: "all", sort: "risk-desc" })}
                  className="mt-1 rounded-md border border-teal/40 bg-teal/10 px-3 py-1.5 font-mono text-[12px] text-teal transition-colors hover:bg-teal/20"
                >
                  Reset all filters
                </button>
              }
            />
          </div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[760px] text-left text-[13px]">
                <thead>
                  <tr className="border-b border-linesoft font-mono text-[10px] uppercase tracking-[0.12em] text-faint">
                    <th className="px-4 py-2.5 font-medium">Customer</th>
                    <th className="px-3 py-2.5 font-medium">Tenure</th>
                    <th className="px-3 py-2.5 font-medium">Monthly</th>
                    <th className="px-3 py-2.5 font-medium">Tickets</th>
                    <th className="px-3 py-2.5 font-medium">Top driver</th>
                    <th className="px-4 py-2.5 text-right font-medium">Churn risk</th>
                  </tr>
                </thead>
                <tbody>
                  {pageRows.map((row) => (
                    <Row key={row.customer.id} row={row} model={model} onOpen={() => setSelected(row.index)} />
                  ))}
                </tbody>
              </table>
            </div>
            <div className="flex items-center justify-between border-t border-line px-4 py-2.5">
              <p className="font-mono text-[11px] text-faint">
                {safePage * PAGE_SIZE + 1}–{Math.min(scored.length, (safePage + 1) * PAGE_SIZE)} of {scored.length.toLocaleString()}
              </p>
              <div className="flex items-center gap-1.5">
                <PageBtn label="Prev" disabled={safePage === 0} onClick={() => setPage(safePage - 1)} icon={<IconArrowUp size={13} className="-rotate-90" />} />
                {Array.from({ length: pageCount }, (_, i) => i)
                  .filter((i) => i === 0 || i === pageCount - 1 || Math.abs(i - safePage) <= 1)
                  .reduce<(number | "gap")[]>((acc, i) => {
                    if (acc.length > 0 && i - (acc[acc.length - 1] as number) > 1) acc.push("gap");
                    acc.push(i);
                    return acc;
                  }, [])
                  .map((i, k) =>
                    i === "gap" ? (
                      <span key={`gap-${k}`} className="px-1 text-faint">
                        …
                      </span>
                    ) : (
                      <button
                        key={i}
                        onClick={() => setPage(i)}
                        className={`h-7 w-7 rounded-md font-mono text-[11px] transition-colors ${
                          i === safePage ? "bg-raised text-teal" : "text-muted hover:bg-hover hover:text-ink"
                        }`}
                      >
                        {i + 1}
                      </button>
                    ),
                  )}
                <PageBtn label="Next" disabled={safePage >= pageCount - 1} onClick={() => setPage(safePage + 1)} icon={<IconArrowDown size={13} className="-rotate-90" />} />
              </div>
            </div>
          </>
        )}
      </div>

      {selected !== null && <DetailDrawer model={model} index={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}

/* ============================================================================ */
/* SECTION: FILTER PANEL — search, categorical filters, sorting and CSV export  */
/* ============================================================================ */
function FilterPanel({
  filters,
  set,
  total,
  matched,
  highRisk,
  onExport,
}: {
  filters: Filters;
  set: (p: Partial<Filters>) => void;
  total: number;
  matched: number;
  highRisk: number;
  onExport: () => void;
}) {
  const select = (value: string, onChange: (v: never) => void, options: string[], allLabel: string) => (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value as never)}
      className="w-full cursor-pointer rounded-md border border-line bg-canvas px-2.5 py-2 font-mono text-[12px] text-ink transition-colors hover:border-teal/40"
    >
      <option value="all">{allLabel}</option>
      {options.map((o) => (
        <option key={o} value={o}>
          {o}
        </option>
      ))}
    </select>
  );

  return (
    <div className="panel h-fit p-4">
      <SectionHead title="Filters" sub={`${matched.toLocaleString()} of ${total.toLocaleString()} records`} />
      <div className="space-y-3">
        <div className="relative">
          <input
            value={filters.q}
            onChange={(e) => set({ q: e.target.value })}
            placeholder="Search ID or payment method…"
            className="w-full rounded-md border border-line bg-canvas py-2 pl-8 pr-3 text-[13px] text-ink placeholder:text-faint transition-colors focus:border-teal/50 focus:outline-none"
            aria-label="Search customers"
          />
          <span className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-faint">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <circle cx="11" cy="11" r="6.5" />
              <path d="M20 20l-4.4-4.4" />
            </svg>
          </span>
        </div>
        {select(filters.contract, (v) => set({ contract: v as Filters["contract"] }), ["Month-to-month", "One year", "Two year"], "All contracts")}
        {select(filters.internet, (v) => set({ internet: v as Filters["internet"] }), ["None", "DSL", "Fiber optic"], "All internet services")}
        <div>
          <p className="mb-1.5 font-mono text-[10px] uppercase tracking-[0.12em] text-faint">Sort order</p>
          <div className="flex flex-wrap gap-1.5">
            {SORT_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                onClick={() => set({ sort: opt.value })}
                aria-pressed={filters.sort === opt.value}
                className={`rounded-md border px-2.5 py-1.5 font-mono text-[11px] transition-colors ${
                  filters.sort === opt.value
                    ? "border-teal/50 bg-teal/12 text-teal"
                    : "border-line text-muted hover:border-faint hover:text-ink"
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
        <button
          onClick={onExport}
          className="flex w-full items-center justify-center gap-2 rounded-md border border-amber/40 bg-amber/10 px-3 py-2 font-mono text-[12px] font-medium text-amber transition-all hover:bg-amber/20 active:scale-[0.98]"
        >
          <IconSpark size={14} />
          Export {highRisk.toLocaleString()} flagged → CSV
        </button>
        <p className="text-[11px] leading-relaxed text-faint">
          Flagged = probability ≥ 50%. Click any row for the full SHAP-style decomposition and a tailored retention playbook.
        </p>
      </div>
    </div>
  );
}

/* ============================================================================ */
/* SECTION: BAND PILLS + PAGINATION CONTROLS                                    */
/* ============================================================================ */
function BandPill({ band, active, onClick }: { band: BandFilter; active: boolean; onClick: () => void }) {
  const labels: Record<BandFilter, string> = { all: "All", low: "Low", watch: "Watch", high: "High" };
  return (
    <button
      onClick={onClick}
      className={`rounded-full border px-2.5 py-1 font-mono text-[11px] transition-all ${
        active
          ? band === "high"
            ? "border-rose/60 bg-rose/15 text-rose"
            : band === "watch"
              ? "border-amber/60 bg-amber/15 text-amber"
              : band === "low"
                ? "border-teal/60 bg-teal/15 text-teal"
                : "border-sky/60 bg-sky/15 text-sky"
          : "border-line text-muted hover:border-faint hover:text-ink"
      }`}
    >
      {labels[band]}
    </button>
  );
}

function PageBtn({ label, disabled, onClick, icon }: { label: string; disabled: boolean; onClick: () => void; icon: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      className="grid h-7 w-7 place-items-center rounded-md border border-line text-muted transition-colors hover:border-teal/40 hover:text-teal disabled:opacity-30 disabled:hover:border-line disabled:hover:text-muted"
    >
      {icon}
    </button>
  );
}

/* ============================================================================ */
/* SECTION: TABLE ROW — risk bar, band chip and resolved top risk driver        */
/* ============================================================================ */
function Row({ row, model, onOpen }: { row: { customer: Customer; index: number; prob: number }; model: TrainedModel; onOpen: () => void }) {
  const info = bandOf(row.prob);
  const driver = useMemo(() => {
    const ex = explain(model.gbdt, featureVector(model.matrix, row.index));
    let best = 0;
    let bestVal = 0.02;
    ex.contribs.forEach((v, i) => {
      if (v > bestVal) {
        bestVal = v;
        best = i;
      }
    });
    return bestVal > 0.02 ? model.matrix.names[best].replace(/_/g, " ") : "—";
  }, [model, row.index]);

  return (
    <tr
      onClick={onOpen}
      className="group cursor-pointer border-b border-linesoft/70 transition-colors last:border-0 hover:bg-hover/60"
    >
      <td className="px-4 py-3">
        <p className="font-mono text-[12px] font-semibold text-ink group-hover:text-teal">{row.customer.id}</p>
        <p className="text-[11px] text-faint">
          {row.customer.contract} · {row.customer.internet}
        </p>
      </td>
      <td className="px-3 py-3 font-mono text-[12px] text-muted tabular-nums">{row.customer.tenure} mo</td>
      <td className="px-3 py-3 font-mono text-[12px] text-muted tabular-nums">{fmtMoney(row.customer.monthlyCharges)}</td>
      <td className="px-3 py-3">
        <span className={`font-mono text-[12px] tabular-nums ${row.customer.tickets >= 3 ? "text-rose" : "text-muted"}`}>{row.customer.tickets}</span>
      </td>
      <td className="px-3 py-3 text-[11px] text-muted">{driver}</td>
      <td className="px-4 py-3">
        <div className="flex items-center justify-end gap-2.5">
          <div className="h-[7px] w-24 overflow-hidden rounded-full bg-canvas">
            <div className="anim-grow-x h-full rounded-full" style={{ width: `${row.prob * 100}%`, background: info.color }} />
          </div>
          <span className="w-12 text-right font-mono text-[12px] font-semibold tabular-nums" style={{ color: info.color }}>
            {fmtPct(row.prob, 0)}
          </span>
        </div>
      </td>
    </tr>
  );
}

/* ============================================================================ */
/* SECTION: DETAIL DRAWER — gauge, SHAP-style waterfall, profile and playbook   */
/* for one customer; dismissible via backdrop, close button or Escape key       */
/* ============================================================================ */
function DetailDrawer({ model, index, onClose }: { model: TrainedModel; index: number; onClose: () => void }) {
  const customer = model.customers[index];
  const ex = useMemo(() => explain(model.gbdt, featureVector(model.matrix, index)), [model, index]);
  const info = bandOf(ex.probability);

  const items = useMemo(
    () =>
      model.matrix.names
        .map((name, i) => ({ name: name.replace(/_/g, " "), value: ex.contribs[i] }))
        .filter((it) => Math.abs(it.value) > 0.005)
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value)),
    [model, ex],
  );
  const playbook = useMemo(() => buildPlaybook(customer, ex.contribs, model.matrix.names), [customer, ex, model]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const facts: [string, string][] = [
    ["Contract", customer.contract],
    ["Internet", customer.internet],
    ["Payment", customer.payment],
    ["Tenure", `${customer.tenure} months`],
    ["Monthly", fmtMoney(customer.monthlyCharges)],
    ["Total billed", customer.totalCharges > 0 ? fmtMoney(customer.totalCharges) : "no history"],
    ["Avg / month", fmtMoney(customer.avgMonthly)],
    ["Tickets (qtr)", `${customer.tickets}`],
    ["Tech support", customer.techSupport ? "Yes" : "No"],
    ["Online security", customer.onlineSecurity ? "Yes" : "No"],
    ["Paperless", customer.paperless ? "Yes" : "No"],
    ["Outcome", customer.churned ? "Churned" : "Retained"],
  ];

  return (
    <div className="fixed inset-0 z-[60]">
      <button className="absolute inset-0 bg-canvas/70 backdrop-blur-[2px]" onClick={onClose} aria-label="Close detail drawer" />
      <aside className="drawer-in absolute inset-y-0 right-0 flex w-full max-w-[560px] flex-col overflow-y-auto border-l border-line bg-panel shadow-2xl">
        <header className="sticky top-0 z-10 flex items-start justify-between gap-3 border-b border-line bg-panel/95 px-5 py-4 backdrop-blur">
          <div>
            <div className="flex items-center gap-2.5">
              <h3 className="font-display text-lg font-bold text-ink">{customer.id}</h3>
              <Chip color={info.color} soft={info.soft}>
                {info.label}
              </Chip>
            </div>
            <p className="mt-0.5 font-mono text-[11px] text-faint">
              {customer.contract} · {customer.internet} · {customer.payment}
            </p>
          </div>
          <button onClick={onClose} className="rounded-md border border-line p-1.5 text-muted transition-colors hover:border-rose/50 hover:text-rose" aria-label="Close">
            <IconX size={15} />
          </button>
        </header>

        <div className="space-y-5 px-5 py-5">
          <div className="flex flex-col items-center rounded-xl border border-line bg-canvas/50 px-4 pb-2 pt-4">
            <RiskGauge probability={ex.probability} />
            <p className="-mt-1 font-display text-3xl font-bold tabular-nums" style={{ color: info.color }}>
              {fmtPct(ex.probability)}
            </p>
            <p className="mb-2 font-mono text-[11px] text-faint">predicted 30-day churn probability</p>
          </div>

          <div>
            <SectionHead title="Why this score" sub="Additive log-odds decomposition (tree-path SHAP)" />
            <div className="rounded-xl border border-line bg-canvas/40 p-4">
              <Waterfall base={ex.base} items={items} probability={ex.probability} />
            </div>
          </div>

          <div>
            <SectionHead title="Customer profile" sub="Raw fields the model reads" />
            <dl className="grid grid-cols-2 gap-2 sm:grid-cols-3">
              {facts.map(([k, v]) => (
                <div key={k} className="rounded-md border border-linesoft bg-canvas/50 px-2.5 py-2">
                  <dt className="font-mono text-[9.5px] uppercase tracking-wider text-faint">{k}</dt>
                  <dd className={`mt-0.5 truncate font-mono text-[12px] ${k === "Outcome" ? (customer.churned ? "font-semibold text-rose" : "font-semibold text-teal") : "text-ink"}`}>{v}</dd>
                </div>
              ))}
            </dl>
          </div>

          <div>
            <SectionHead title="Retention playbook" sub="Ranked interventions for this profile" />
            <ol className="space-y-2">
              {playbook.map((a, i) => (
                <li key={a.title} className="anim-fade-up flex gap-3 rounded-lg border border-line bg-raised/60 px-3.5 py-3" style={{ animationDelay: `${i * 70}ms` }}>
                  <span className={`mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-full font-mono text-[11px] font-bold ${a.impact === "High" ? "bg-rose/15 text-rose" : "bg-amber/15 text-amber"}`}>
                    {i + 1}
                  </span>
                  <div>
                    <p className="flex flex-wrap items-center gap-2 text-[13px] font-semibold text-ink">
                      {a.title}
                      <span className={`rounded-full px-2 py-px font-mono text-[9.5px] uppercase tracking-wider ${a.impact === "High" ? "bg-rose/12 text-rose" : "bg-amber/12 text-amber"}`}>
                        {a.impact} impact
                      </span>
                    </p>
                    <p className="mt-0.5 text-[12px] leading-relaxed text-muted">{a.detail}</p>
                  </div>
                </li>
              ))}
            </ol>
          </div>
        </div>
      </aside>
    </div>
  );
}


+++ src/components/Explorer.tsx (修改后)
import { useEffect, useRef, useState } from "react";
import type { Adapter, PageResult, RowQuery, ScoredRow, SortKey } from "../lib/adapter";
import { bandOf, buildPlaybook, fmtMoney, fmtPct } from "../lib/analytics";
import type { Band } from "../lib/analytics";
import { RiskGauge, Waterfall } from "./charts";
import { Chip, EmptyState, IconArrowDown, IconArrowUp, IconSpark, IconX, SectionHead } from "./ui";

/* ============================================================================ */
/* SECTION: EXPLORER STATE — filter, sort and pagination definitions shared     */
/* with the backend query endpoint so both engines paginate identically         */
/* ============================================================================ */
type BandFilter = "all" | Band;

interface Filters {
  q: string;
  band: BandFilter;
  contract: string;
  internet: string;
  sort: SortKey;
}

const PAGE_SIZE = 9;
const EXPORT_LIMIT = 500;

const SORT_OPTIONS: { value: SortKey; label: string }[] = [
  { value: "risk-desc", label: "Risk ↓" },
  { value: "risk-asc", label: "Risk ↑" },
  { value: "tenure-asc", label: "Tenure ↑" },
  { value: "charges-desc", label: "Charges ↓" },
  { value: "tickets-desc", label: "Tickets ↓" },
];

/* ============================================================================ */
/* SECTION: RISK EXPLORER WORKSPACE — fetches scored pages from the active      */
/* engine (Python backend or browser fallback), with a per-row detail drawer    */
/* containing the gauge, SHAP-style waterfall and retention playbook            */
/* ============================================================================ */
export function Explorer({
  adapter,
  threshold,
  notify,
}: {
  adapter: Adapter;
  threshold: number;
  notify: (msg: string, kind?: "success" | "info" | "error") => void;
}) {
  const [filters, setFilters] = useState<Filters>({ q: "", band: "all", contract: "all", internet: "all", sort: "risk-desc" });
  const [debouncedQ, setDebouncedQ] = useState("");
  const [page, setPage] = useState(0);
  const [result, setResult] = useState<PageResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<ScoredRow | null>(null);
  const requestId = useRef(0);

  /* debounce the search box so each keystroke does not hit the backend */
  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedQ(filters.q), 220);
    return () => window.clearTimeout(timer);
  }, [filters.q]);

  /* return to the first page whenever the query shape changes */
  useEffect(() => {
    setPage(0);
  }, [adapter, debouncedQ, filters.band, filters.contract, filters.internet, filters.sort]);

  /* fetch the visible page; the request guard drops stale responses */
  useEffect(() => {
    const id = ++requestId.current;
    setLoading(true);
    const query: RowQuery = {
      search: debouncedQ,
      band: filters.band,
      contract: filters.contract,
      internet: filters.internet,
      sort: filters.sort,
      page,
      size: PAGE_SIZE,
    };
    adapter
      .fetchRows(query)
      .then((res) => {
        if (requestId.current === id) setResult(res);
      })
      .catch((err) => {
        if (requestId.current === id) notify(err instanceof Error ? err.message : "Failed to load customers", "error");
      })
      .finally(() => {
        if (requestId.current === id) setLoading(false);
      });
  }, [adapter, debouncedQ, filters.band, filters.contract, filters.internet, filters.sort, page, notify]);

  const pages = result?.pages ?? 1;
  const total = result?.total ?? 0;
  const safePage = Math.min(page, pages - 1);
  const rows = result?.rows ?? [];

  const set = (patch: Partial<Filters>) => setFilters((f) => ({ ...f, ...patch }));

  /* pull up to EXPORT_LIMIT matching rows and export those above the cut-off */
  const exportFlagged = () => {
    const query: RowQuery = {
      search: debouncedQ,
      band: filters.band,
      contract: filters.contract,
      internet: filters.internet,
      sort: filters.sort,
      page: 0,
      size: EXPORT_LIMIT,
    };
    adapter
      .fetchRows(query)
      .then((res) => {
        const flagged = res.rows.filter((r) => r.prob >= threshold);
        if (flagged.length === 0) {
          notify(`No customers above the ${fmtPct(threshold, 0)} risk line in the current view.`, "error");
          return;
        }
        const header = "customer_id,tenure,contract,internet_service,monthly_charges,support_tickets,churn_probability,risk_band";
        const lines = flagged.map((r) =>
          [r.customer.id, r.customer.tenure, r.customer.contract, r.customer.internet, r.customer.monthlyCharges, r.customer.tickets, r.prob.toFixed(4), bandOf(r.prob).label].join(","),
        );
        const blob = new Blob([[header, ...lines].join("\n")], { type: "text/csv" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "churnlens_flagged_customers.csv";
        a.click();
        URL.revokeObjectURL(url);
        const capped = res.total > EXPORT_LIMIT ? ` (first ${EXPORT_LIMIT} matching rows)` : "";
        notify(`Exported ${flagged.length} flagged customers to CSV${capped}.`, "success");
      })
      .catch((err) => notify(err instanceof Error ? err.message : "Export failed", "error"));
  };

  return (
    <div className="anim-fade-up grid gap-3 xl:grid-cols-[300px_1fr]">
      <FilterPanel
        filters={filters}
        set={set}
        total={adapter.totalCount}
        matched={total}
        threshold={threshold}
        onExport={exportFlagged}
      />

      <div className="panel overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-3">
          <SectionHead title="Scored Customers" sub={`${total.toLocaleString()} matching · page ${safePage + 1} of ${pages}`} />
          <div className="flex items-center gap-2">
            {(["all", "low", "watch", "high"] as BandFilter[]).map((b) => (
              <BandPill key={b} band={b} active={filters.band === b} onClick={() => set({ band: b })} />
            ))}
          </div>
        </div>

        {!loading && rows.length === 0 ? (
          <div className="p-6">
            <EmptyState
              title="No customers match these filters"
              hint="Loosen the search text or reset the band and service filters to see the full scored cohort again."
              action={
                <button
                  onClick={() => setFilters({ q: "", band: "all", contract: "all", internet: "all", sort: "risk-desc" })}
                  className="mt-1 rounded-md border border-teal/40 bg-teal/10 px-3 py-1.5 font-mono text-[12px] text-teal transition-colors hover:bg-teal/20"
                >
                  Reset all filters
                </button>
              }
            />
          </div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[760px] text-left text-[13px]">
                <thead>
                  <tr className="border-b border-linesoft font-mono text-[10px] uppercase tracking-[0.12em] text-faint">
                    <th className="px-4 py-2.5 font-medium">Customer</th>
                    <th className="px-3 py-2.5 font-medium">Tenure</th>
                    <th className="px-3 py-2.5 font-medium">Monthly</th>
                    <th className="px-3 py-2.5 font-medium">Tickets</th>
                    <th className="px-3 py-2.5 font-medium">Top driver</th>
                    <th className="px-4 py-2.5 text-right font-medium">Churn risk</th>
                  </tr>
                </thead>
                <tbody className={`transition-opacity duration-200 ${loading ? "opacity-50" : "opacity-100"}`}>
                  {rows.map((row) => (
                    <Row key={row.key} row={row} featureNames={adapter.featureNames} onOpen={() => setSelected(row)} />
                  ))}
                </tbody>
              </table>
            </div>
            <div className="flex items-center justify-between border-t border-line px-4 py-2.5">
              <p className="font-mono text-[11px] text-faint">
                {total === 0 ? "0 results" : `${safePage * PAGE_SIZE + 1}–${Math.min(total, (safePage + 1) * PAGE_SIZE)} of ${total.toLocaleString()}`}
              </p>
              <div className="flex items-center gap-1.5">
                <PageBtn label="Prev" disabled={safePage === 0} onClick={() => setPage(safePage - 1)} icon={<IconArrowUp size={13} className="-rotate-90" />} />
                {Array.from({ length: pages }, (_, i) => i)
                  .filter((i) => i === 0 || i === pages - 1 || Math.abs(i - safePage) <= 1)
                  .reduce<(number | "gap")[]>((acc, i) => {
                    if (acc.length > 0 && i - (acc[acc.length - 1] as number) > 1) acc.push("gap");
                    acc.push(i);
                    return acc;
                  }, [])
                  .map((i, k) =>
                    i === "gap" ? (
                      <span key={`gap-${k}`} className="px-1 text-faint">
                        …
                      </span>
                    ) : (
                      <button
                        key={i}
                        onClick={() => setPage(i)}
                        className={`h-7 w-7 rounded-md font-mono text-[11px] transition-colors ${
                          i === safePage ? "bg-raised text-teal" : "text-muted hover:bg-hover hover:text-ink"
                        }`}
                      >
                        {i + 1}
                      </button>
                    ),
                  )}
                <PageBtn label="Next" disabled={safePage >= pages - 1} onClick={() => setPage(safePage + 1)} icon={<IconArrowDown size={13} className="-rotate-90" />} />
              </div>
            </div>
          </>
        )}
      </div>

      {selected && <DetailDrawer row={selected} featureNames={adapter.featureNames} onClose={() => setSelected(null)} />}
    </div>
  );
}

/* ============================================================================ */
/* SECTION: FILTER PANEL — search, categorical filters, sorting and CSV export  */
/* ============================================================================ */
function FilterPanel({
  filters,
  set,
  total,
  matched,
  threshold,
  onExport,
}: {
  filters: Filters;
  set: (p: Partial<Filters>) => void;
  total: number;
  matched: number;
  threshold: number;
  onExport: () => void;
}) {
  const select = (value: string, onChange: (v: string) => void, options: string[], allLabel: string) => (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full cursor-pointer rounded-md border border-line bg-canvas px-2.5 py-2 font-mono text-[12px] text-ink transition-colors hover:border-teal/40"
    >
      <option value="all">{allLabel}</option>
      {options.map((o) => (
        <option key={o} value={o}>
          {o}
        </option>
      ))}
    </select>
  );

  return (
    <div className="panel h-fit p-4">
      <SectionHead title="Filters" sub={`${matched.toLocaleString()} of ${total.toLocaleString()} records`} />
      <div className="space-y-3">
        <div className="relative">
          <input
            value={filters.q}
            onChange={(e) => set({ q: e.target.value })}
            placeholder="Search ID or payment method…"
            className="w-full rounded-md border border-line bg-canvas py-2 pl-8 pr-3 text-[13px] text-ink placeholder:text-faint transition-colors focus:border-teal/50 focus:outline-none"
            aria-label="Search customers"
          />
          <span className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-faint">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <circle cx="11" cy="11" r="6.5" />
              <path d="M20 20l-4.4-4.4" />
            </svg>
          </span>
        </div>
        {select(filters.contract, (v) => set({ contract: v }), ["Month-to-month", "One year", "Two year"], "All contracts")}
        {select(filters.internet, (v) => set({ internet: v }), ["None", "DSL", "Fiber optic"], "All internet services")}
        <div>
          <p className="mb-1.5 font-mono text-[10px] uppercase tracking-[0.12em] text-faint">Sort order</p>
          <div className="flex flex-wrap gap-1.5">
            {SORT_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                onClick={() => set({ sort: opt.value })}
                aria-pressed={filters.sort === opt.value}
                className={`rounded-md border px-2.5 py-1.5 font-mono text-[11px] transition-colors ${
                  filters.sort === opt.value
                    ? "border-teal/50 bg-teal/12 text-teal"
                    : "border-line text-muted hover:border-faint hover:text-ink"
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
        <button
          onClick={onExport}
          className="flex w-full items-center justify-center gap-2 rounded-md border border-amber/40 bg-amber/10 px-3 py-2 font-mono text-[12px] font-medium text-amber transition-all hover:bg-amber/20 active:scale-[0.98]"
        >
          <IconSpark size={14} />
          Export flagged → CSV
        </button>
        <p className="text-[11px] leading-relaxed text-faint">
          Flagged = probability ≥ {fmtPct(threshold, 0)} (shared with the Overview threshold). Click any row for the full
          decomposition and a tailored retention playbook.
        </p>
      </div>
    </div>
  );
}

/* ============================================================================ */
/* SECTION: BAND PILLS + PAGINATION CONTROLS                                    */
/* ============================================================================ */
function BandPill({ band, active, onClick }: { band: BandFilter; active: boolean; onClick: () => void }) {
  const labels: Record<BandFilter, string> = { all: "All", low: "Low", watch: "Watch", high: "High" };
  return (
    <button
      onClick={onClick}
      className={`rounded-full border px-2.5 py-1 font-mono text-[11px] transition-all ${
        active
          ? band === "high"
            ? "border-rose/60 bg-rose/15 text-rose"
            : band === "watch"
              ? "border-amber/60 bg-amber/15 text-amber"
              : band === "low"
                ? "border-teal/60 bg-teal/15 text-teal"
                : "border-sky/60 bg-sky/15 text-sky"
          : "border-line text-muted hover:border-faint hover:text-ink"
      }`}
    >
      {labels[band]}
    </button>
  );
}

function PageBtn({ label, disabled, onClick, icon }: { label: string; disabled: boolean; onClick: () => void; icon: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      className="grid h-7 w-7 place-items-center rounded-md border border-line text-muted transition-colors hover:border-teal/40 hover:text-teal disabled:opacity-30 disabled:hover:border-line disabled:hover:text-muted"
    >
      {icon}
    </button>
  );
}

/* ============================================================================ */
/* SECTION: TABLE ROW — risk bar, band chip and resolved top risk driver taken  */
/* from the row's precomputed additive contributions                            */
/* ============================================================================ */
function Row({ row, featureNames, onOpen }: { row: ScoredRow; featureNames: readonly string[]; onOpen: () => void }) {
  const info = bandOf(row.prob);
  let driver = "—";
  let bestVal = 0.02;
  row.contribs.forEach((v, i) => {
    if (v > bestVal) {
      bestVal = v;
      driver = (featureNames[i] ?? "feature").replace(/_/g, " ");
    }
  });

  return (
    <tr onClick={onOpen} className="group cursor-pointer border-b border-linesoft/70 transition-colors last:border-0 hover:bg-hover/60">
      <td className="px-4 py-3">
        <p className="font-mono text-[12px] font-semibold text-ink group-hover:text-teal">{row.customer.id}</p>
        <p className="text-[11px] text-faint">
          {row.customer.contract} · {row.customer.internet}
        </p>
      </td>
      <td className="px-3 py-3 font-mono text-[12px] text-muted tabular-nums">{row.customer.tenure} mo</td>
      <td className="px-3 py-3 font-mono text-[12px] text-muted tabular-nums">{fmtMoney(row.customer.monthlyCharges)}</td>
      <td className="px-3 py-3">
        <span className={`font-mono text-[12px] tabular-nums ${row.customer.tickets >= 3 ? "text-rose" : "text-muted"}`}>{row.customer.tickets}</span>
      </td>
      <td className="px-3 py-3 text-[11px] text-muted">{driver}</td>
      <td className="px-4 py-3">
        <div className="flex items-center justify-end gap-2.5">
          <div className="h-[7px] w-24 overflow-hidden rounded-full bg-canvas">
            <div className="anim-grow-x h-full rounded-full" style={{ width: `${row.prob * 100}%`, background: info.color }} />
          </div>
          <span className="w-12 text-right font-mono text-[12px] font-semibold tabular-nums" style={{ color: info.color }}>
            {fmtPct(row.prob, 0)}
          </span>
        </div>
      </td>
    </tr>
  );
}

/* ============================================================================ */
/* SECTION: DETAIL DRAWER — gauge, SHAP-style waterfall, profile and playbook   */
/* for one customer; dismissible via backdrop, close button or Escape key       */
/* ============================================================================ */
function DetailDrawer({ row, featureNames, onClose }: { row: ScoredRow; featureNames: readonly string[]; onClose: () => void }) {
  const customer = row.customer;
  const info = bandOf(row.prob);

  const items = featureNames
    .map((name, i) => ({ name: name.replace(/_/g, " "), value: row.contribs[i] ?? 0 }))
    .filter((it) => Math.abs(it.value) > 0.005)
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  const playbook = buildPlaybook(customer, row.contribs, featureNames);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const facts: [string, string][] = [
    ["Contract", customer.contract],
    ["Internet", customer.internet],
    ["Payment", customer.payment],
    ["Tenure", `${customer.tenure} months`],
    ["Monthly", fmtMoney(customer.monthlyCharges)],
    ["Total billed", customer.totalCharges > 0 ? fmtMoney(customer.totalCharges) : "no history"],
    ["Avg / month", fmtMoney(customer.avgMonthly)],
    ["Tickets (qtr)", `${customer.tickets}`],
    ["Tech support", customer.techSupport ? "Yes" : "No"],
    ["Online security", customer.onlineSecurity ? "Yes" : "No"],
    ["Paperless", customer.paperless ? "Yes" : "No"],
    ["Outcome", customer.churned ? "Churned" : "Retained"],
  ];

  return (
    <div className="fixed inset-0 z-[60]">
      <button className="absolute inset-0 bg-canvas/70 backdrop-blur-[2px]" onClick={onClose} aria-label="Close detail drawer" />
      <aside className="drawer-in absolute inset-y-0 right-0 flex w-full max-w-[560px] flex-col overflow-y-auto border-l border-line bg-panel shadow-2xl">
        <header className="sticky top-0 z-10 flex items-start justify-between gap-3 border-b border-line bg-panel/95 px-5 py-4 backdrop-blur">
          <div>
            <div className="flex items-center gap-2.5">
              <h3 className="font-display text-lg font-bold text-ink">{customer.id}</h3>
              <Chip color={info.color} soft={info.soft}>
                {info.label}
              </Chip>
            </div>
            <p className="mt-0.5 font-mono text-[11px] text-faint">
              {customer.contract} · {customer.internet} · {customer.payment}
            </p>
          </div>
          <button onClick={onClose} className="rounded-md border border-line p-1.5 text-muted transition-colors hover:border-rose/50 hover:text-rose" aria-label="Close">
            <IconX size={15} />
          </button>
        </header>

        <div className="space-y-5 px-5 py-5">
          <div className="flex flex-col items-center rounded-xl border border-line bg-canvas/50 px-4 pb-2 pt-4">
            <RiskGauge probability={row.prob} />
            <p className="-mt-1 font-display text-3xl font-bold tabular-nums" style={{ color: info.color }}>
              {fmtPct(row.prob)}
            </p>
            <p className="mb-2 font-mono text-[11px] text-faint">predicted 30-day churn probability</p>
          </div>

          <div>
            <SectionHead title="Why this score" sub="Additive log-odds decomposition (tree-path SHAP)" />
            <div className="rounded-xl border border-line bg-canvas/40 p-4">
              <Waterfall base={row.base} items={items} probability={row.prob} />
            </div>
          </div>

          <div>
            <SectionHead title="Customer profile" sub="Raw fields the model reads" />
            <dl className="grid grid-cols-2 gap-2 sm:grid-cols-3">
              {facts.map(([k, v]) => (
                <div key={k} className="rounded-md border border-linesoft bg-canvas/50 px-2.5 py-2">
                  <dt className="font-mono text-[9.5px] uppercase tracking-wider text-faint">{k}</dt>
                  <dd className={`mt-0.5 truncate font-mono text-[12px] ${k === "Outcome" ? (customer.churned ? "font-semibold text-rose" : "font-semibold text-teal") : "text-ink"}`}>{v}</dd>
                </div>
              ))}
            </dl>
          </div>

          <div>
            <SectionHead title="Retention playbook" sub="Ranked interventions for this profile" />
            <ol className="space-y-2">
              {playbook.map((a, i) => (
                <li key={a.title} className="anim-fade-up flex gap-3 rounded-lg border border-line bg-raised/60 px-3.5 py-3" style={{ animationDelay: `${i * 70}ms` }}>
                  <span className={`mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-full font-mono text-[11px] font-bold ${a.impact === "High" ? "bg-rose/15 text-rose" : "bg-amber/15 text-amber"}`}>
                    {i + 1}
                  </span>
                  <div>
                    <p className="flex flex-wrap items-center gap-2 text-[13px] font-semibold text-ink">
                      {a.title}
                      <span className={`rounded-full px-2 py-px font-mono text-[9.5px] uppercase tracking-wider ${a.impact === "High" ? "bg-rose/12 text-rose" : "bg-amber/12 text-amber"}`}>
                        {a.impact} impact
                      </span>
                    </p>
                    <p className="mt-0.5 text-[12px] leading-relaxed text-muted">{a.detail}</p>
                  </div>
                </li>
              ))}
            </ol>
          </div>
        </div>
      </aside>
    </div>
  );
}
