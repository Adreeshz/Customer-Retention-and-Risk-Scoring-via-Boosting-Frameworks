import { useEffect, useMemo, useState } from "react";
import type { TrainedModel } from "../hooks/useModel";
import type { Contract, Customer, InternetService, PaymentMethod } from "../lib/dataset";
import { featureVectorFromCustomer } from "../lib/dataset";
import { explain } from "../lib/boosting";
import { bandOf, buildPlaybook, fmtMoney, fmtPct } from "../lib/analytics";
import { RiskGauge, Waterfall } from "./charts";
import { Chip, IconRefresh, IconSpark, SectionHead, Segmented, SliderRow, Stepper, Toggle } from "./ui";
import { clamp, createRng, randInt } from "../lib/random";

/* ============================================================================ */
/* SECTION: SIMULATOR STATE — the editable customer profile plus persistence of */
/* saved scenarios so what-if experiments survive a page reload                 */
/* ============================================================================ */
export interface SimState {
  tenure: number;
  monthlyCharges: number;
  contract: Contract;
  internet: InternetService;
  payment: PaymentMethod;
  techSupport: boolean;
  onlineSecurity: boolean;
  streamingTV: boolean;
  paperless: boolean;
  risingBill: boolean;
  tickets: number;
  senior: boolean;
  dependents: boolean;
}

interface Scenario {
  id: number;
  label: string;
  state: SimState;
  probability: number;
}

const DEFAULT_SIM: SimState = {
  tenure: 8,
  monthlyCharges: 78,
  contract: "Month-to-month",
  internet: "Fiber optic",
  payment: "Electronic check",
  techSupport: false,
  onlineSecurity: false,
  streamingTV: true,
  paperless: true,
  risingBill: true,
  tickets: 3,
  senior: false,
  dependents: false,
};

const SCENARIOS_KEY = "churnlens:scenarios:v1";

function loadScenarios(): Scenario[] {
  try {
    const raw = localStorage.getItem(SCENARIOS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as Scenario[];
    return Array.isArray(parsed) ? parsed.filter((s) => s && typeof s.probability === "number" && s.state) : [];
  } catch {
    return [];
  }
}

/* ============================================================================ */
/* SECTION: WHAT-IF SIMULATOR WORKSPACE — live re-scoring of a synthetic        */
/* customer as any field changes, with the decomposition and playbook updated   */
/* ============================================================================ */
export function Simulator({ model, notify }: { model: TrainedModel; notify: (msg: string, kind?: "success" | "info" | "error") => void }) {
  const [sim, setSim] = useState<SimState>(DEFAULT_SIM);
  const [scenarios, setScenarios] = useState<Scenario[]>(loadScenarios);

  useEffect(() => {
    try {
      localStorage.setItem(SCENARIOS_KEY, JSON.stringify(scenarios.slice(0, 6)));
    } catch {
      /* storage unavailable — scenarios simply stay in memory */
    }
  }, [scenarios]);

  const synthetic: Customer = useMemo(
    () => ({
      id: "SIM-PROFILE",
      gender: "Female",
      senior: sim.senior ? 1 : 0,
      dependents: sim.dependents,
      partner: false,
      tenure: sim.tenure,
      contract: sim.contract,
      paperless: sim.paperless,
      payment: sim.payment,
      internet: sim.internet,
      techSupport: sim.techSupport,
      onlineSecurity: sim.onlineSecurity,
      streamingTV: sim.streamingTV,
      tickets: sim.tickets,
      monthlyCharges: sim.monthlyCharges,
      totalCharges: sim.tenure > 0 ? sim.monthlyCharges * sim.tenure * (sim.risingBill ? 0.9 : 1) : 0,
      avgMonthly: sim.tenure > 0 ? sim.monthlyCharges * (sim.risingBill ? 0.9 : 1) : sim.monthlyCharges,
      risingBill: sim.risingBill,
      churned: false,
    }),
    [sim],
  );

  const ex = useMemo(() => explain(model.gbdt, featureVectorFromCustomer(synthetic)), [model, synthetic]);
  const info = bandOf(ex.probability);
  const cohortAvg = useMemo(() => {
    let sum = 0;
    for (let i = 0; i < model.probabilities.length; i++) sum += model.probabilities[i];
    return sum / Math.max(1, model.probabilities.length);
  }, [model]);

  const items = useMemo(
    () =>
      model.matrix.names
        .map((name, i) => ({ name: name.replace(/_/g, " "), value: ex.contribs[i] }))
        .filter((it) => Math.abs(it.value) > 0.005)
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value)),
    [model, ex],
  );
  const playbook = useMemo(() => buildPlaybook(synthetic, ex.contribs, model.matrix.names), [synthetic, ex, model]);

  const set = (patch: Partial<SimState>) => setSim((s) => ({ ...s, ...patch }));

  const randomize = () => {
    const rng = createRng(Date.now() % 100000);
    setSim({
      tenure: randInt(rng, 0, 72),
      monthlyCharges: Math.round(clamp(20 + rng.next() * 100, 18, 122)),
      contract: (["Month-to-month", "One year", "Two year"] as Contract[])[randInt(rng, 0, 2)],
      internet: (["None", "DSL", "Fiber optic"] as InternetService[])[randInt(rng, 0, 2)],
      payment: (["Electronic check", "Mailed check", "Bank transfer", "Credit card"] as PaymentMethod[])[randInt(rng, 0, 3)],
      techSupport: rng.next() < 0.4,
      onlineSecurity: rng.next() < 0.4,
      streamingTV: rng.next() < 0.5,
      paperless: rng.next() < 0.6,
      risingBill: rng.next() < 0.3,
      tickets: randInt(rng, 0, 7),
      senior: rng.next() < 0.16,
      dependents: rng.next() < 0.3,
    });
  };

  const saveScenario = () => {
    const next: Scenario = {
      id: Date.now(),
      label: `S${scenarios.length + 1} · ${sim.contract.split(" ")[0]} · ${sim.tenure}mo`,
      state: sim,
      probability: ex.probability,
    };
    setScenarios((s) => [next, ...s].slice(0, 6));
    notify(`Scenario saved at ${fmtPct(ex.probability)} risk.`, "success");
  };

  const delta = ex.probability - cohortAvg;

  return (
    <div className="anim-fade-up grid gap-3 xl:grid-cols-[380px_1fr]">
      <div className="panel h-fit p-4">
        <SectionHead
          title="Profile Builder"
          sub="Every change re-scores instantly"
          right={
            <div className="flex gap-1.5">
              <button
                onClick={randomize}
                className="flex items-center gap-1.5 rounded-md border border-line px-2.5 py-1.5 font-mono text-[11px] text-muted transition-colors hover:border-sky/50 hover:text-sky"
              >
                <IconRefresh size={13} /> Random
              </button>
              <button
                onClick={() => setSim(DEFAULT_SIM)}
                className="rounded-md border border-line px-2.5 py-1.5 font-mono text-[11px] text-muted transition-colors hover:border-rose/50 hover:text-rose"
              >
                Reset
              </button>
            </div>
          }
        />
        <div className="space-y-4">
          <SliderRow label="Tenure" value={sim.tenure} min={0} max={72} step={1} display={`${sim.tenure} mo`} onChange={(v) => set({ tenure: v })} />
          <SliderRow
            label="Monthly charges"
            value={sim.monthlyCharges}
            min={18}
            max={122}
            step={1}
            display={fmtMoney(sim.monthlyCharges)}
            onChange={(v) => set({ monthlyCharges: v })}
          />
          <div>
            <p className="mb-1.5 font-mono text-[10px] uppercase tracking-[0.12em] text-faint">Contract</p>
            <Segmented
              ariaLabel="Contract type"
              options={[
                { value: "Month-to-month", label: "Monthly" },
                { value: "One year", label: "1 year" },
                { value: "Two year", label: "2 year" },
              ]}
              value={sim.contract}
              onChange={(v) => set({ contract: v })}
            />
          </div>
          <div>
            <p className="mb-1.5 font-mono text-[10px] uppercase tracking-[0.12em] text-faint">Internet service</p>
            <Segmented
              ariaLabel="Internet service"
              options={[
                { value: "None", label: "None" },
                { value: "DSL", label: "DSL" },
                { value: "Fiber optic", label: "Fiber" },
              ]}
              value={sim.internet}
              onChange={(v) => set({ internet: v })}
            />
          </div>
          <div>
            <p className="mb-1.5 font-mono text-[10px] uppercase tracking-[0.12em] text-faint">Payment method</p>
            <Segmented
              ariaLabel="Payment method"
              options={[
                { value: "Electronic check", label: "E-check" },
                { value: "Mailed check", label: "Mail" },
                { value: "Bank transfer", label: "Bank" },
                { value: "Credit card", label: "Card" },
              ]}
              value={sim.payment}
              onChange={(v) => set({ payment: v })}
            />
          </div>
          <Stepper label="Support tickets (last quarter)" value={sim.tickets} min={0} max={8} onChange={(v) => set({ tickets: v })} />
          <div className="rounded-lg border border-linesoft bg-canvas/50 px-3 py-2">
            <Toggle checked={sim.techSupport} onChange={(v) => set({ techSupport: v })} label="Tech support add-on" />
            <Toggle checked={sim.onlineSecurity} onChange={(v) => set({ onlineSecurity: v })} label="Online security add-on" />
            <Toggle checked={sim.streamingTV} onChange={(v) => set({ streamingTV: v })} label="Streaming TV add-on" />
            <Toggle checked={sim.paperless} onChange={(v) => set({ paperless: v })} label="Paperless billing" />
            <Toggle checked={sim.risingBill} onChange={(v) => set({ risingBill: v })} label="Bill trending above average" />
            <Toggle checked={sim.senior} onChange={(v) => set({ senior: v })} label="Senior citizen" />
            <Toggle checked={sim.dependents} onChange={(v) => set({ dependents: v })} label="Has dependents" />
          </div>
          <button
            onClick={saveScenario}
            className="flex w-full items-center justify-center gap-2 rounded-md border border-teal/40 bg-teal/10 px-3 py-2 font-mono text-[12px] font-medium text-teal transition-all hover:bg-teal/20 active:scale-[0.98]"
          >
            <IconSpark size={14} /> Save scenario
          </button>
          {scenarios.length > 0 && (
            <div>
              <p className="mb-1.5 font-mono text-[10px] uppercase tracking-[0.12em] text-faint">Saved scenarios</p>
              <div className="flex flex-wrap gap-1.5">
                {scenarios.map((s) => {
                  const b = bandOf(s.probability);
                  return (
                    <button
                      key={s.id}
                      onClick={() => {
                        setSim(s.state);
                        notify(`Loaded scenario at ${fmtPct(s.probability)} risk.`, "info");
                      }}
                      className="rounded-full border border-line px-2.5 py-1 font-mono text-[10.5px] text-muted transition-colors hover:border-teal/50 hover:text-ink"
                    >
                      {s.label} · <span style={{ color: b.color }}>{fmtPct(s.probability, 0)}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="space-y-3">
        <div className="panel flex flex-col items-center px-4 pb-3 pt-5 sm:flex-row sm:items-center sm:gap-8 sm:px-8">
          <RiskGauge probability={ex.probability} size={250} />
          <div className="mt-2 text-center sm:mt-0 sm:text-left">
            <p className="font-display text-4xl font-bold tabular-nums" style={{ color: info.color }}>
              {fmtPct(ex.probability)}
            </p>
            <p className="mt-1 flex items-center justify-center gap-2 sm:justify-start">
              <Chip color={info.color} soft={info.soft}>
                {info.label}
              </Chip>
            </p>
            <p className="mt-2 font-mono text-[11px] text-muted">
              cohort average {fmtPct(cohortAvg)} ·{" "}
              <span className={delta >= 0 ? "text-rose" : "text-teal"}>
                {delta >= 0 ? "+" : "−"}
                {fmtPct(Math.abs(delta))} vs average
              </span>
            </p>
          </div>
        </div>

        <div className="grid gap-3 lg:grid-cols-2">
          <div className="panel p-4">
            <SectionHead title="Live decomposition" sub="What pushes this profile up or down" />
            <Waterfall base={ex.base} items={items} probability={ex.probability} />
          </div>
          <div className="panel p-4">
            <SectionHead title="Recommended plays" sub="Auto-generated from the top drivers" />
            <ol className="space-y-2">
              {playbook.map((a, i) => (
                <li key={a.title} className="anim-fade-up flex gap-3 rounded-lg border border-line bg-raised/60 px-3.5 py-3" style={{ animationDelay: `${i * 60}ms` }}>
                  <span className={`mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-full font-mono text-[11px] font-bold ${a.impact === "High" ? "bg-rose/15 text-rose" : "bg-amber/15 text-amber"}`}>
                    {i + 1}
                  </span>
                  <div>
                    <p className="text-[13px] font-semibold text-ink">{a.title}</p>
                    <p className="mt-0.5 text-[12px] leading-relaxed text-muted">{a.detail}</p>
                  </div>
                </li>
              ))}
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
}


+++ src/components/Simulator.tsx (修改后)
import { useEffect, useMemo, useRef, useState } from "react";
import type { Adapter, SimProfile, SimResult } from "../lib/adapter";
import type { Contract, Customer, InternetService, PaymentMethod } from "../lib/dataset";
import { bandOf, buildPlaybook, fmtMoney, fmtPct } from "../lib/analytics";
import { RiskGauge, Waterfall } from "./charts";
import { Chip, IconRefresh, IconSpark, SectionHead, Segmented, SliderRow, Stepper, Toggle } from "./ui";
import { clamp, createRng, randInt } from "../lib/random";

/* ============================================================================ */
/* SECTION: SIMULATOR STATE — the editable customer profile plus persistence of */
/* saved scenarios so what-if experiments survive a page reload                 */
/* ============================================================================ */
export interface SimState {
  tenure: number;
  monthlyCharges: number;
  contract: Contract;
  internet: InternetService;
  payment: PaymentMethod;
  techSupport: boolean;
  onlineSecurity: boolean;
  streamingTV: boolean;
  paperless: boolean;
  risingBill: boolean;
  tickets: number;
  senior: boolean;
  dependents: boolean;
}

interface Scenario {
  id: number;
  label: string;
  state: SimState;
  probability: number;
}

const DEFAULT_SIM: SimState = {
  tenure: 8,
  monthlyCharges: 78,
  contract: "Month-to-month",
  internet: "Fiber optic",
  payment: "Electronic check",
  techSupport: false,
  onlineSecurity: false,
  streamingTV: true,
  paperless: true,
  risingBill: true,
  tickets: 3,
  senior: false,
  dependents: false,
};

const SCENARIOS_KEY = "churnlens:scenarios:v1";

function loadScenarios(): Scenario[] {
  try {
    const raw = localStorage.getItem(SCENARIOS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as Scenario[];
    return Array.isArray(parsed) ? parsed.filter((s) => s && typeof s.probability === "number" && s.state) : [];
  } catch {
    return [];
  }
}

/* ============================================================================ */
/* SECTION: WHAT-IF SIMULATOR WORKSPACE — re-scores a synthetic customer through */
/* the active engine (Python backend or browser) as any field changes, with the */
/* decomposition and playbook updated from the returned contributions           */
/* ============================================================================ */
export function Simulator({ adapter, notify }: { adapter: Adapter; notify: (msg: string, kind?: "success" | "info" | "error") => void }) {
  const [sim, setSim] = useState<SimState>(DEFAULT_SIM);
  const [scenarios, setScenarios] = useState<Scenario[]>(loadScenarios);
  const [result, setResult] = useState<SimResult | null>(null);
  const [pending, setPending] = useState(true);
  const requestId = useRef(0);

  useEffect(() => {
    try {
      localStorage.setItem(SCENARIOS_KEY, JSON.stringify(scenarios.slice(0, 6)));
    } catch {
      /* storage unavailable — scenarios simply stay in memory */
    }
  }, [scenarios]);

  const profile: SimProfile = useMemo(
    () => ({
      tenure: sim.tenure,
      monthlyCharges: sim.monthlyCharges,
      contract: sim.contract,
      internet: sim.internet,
      payment: sim.payment,
      techSupport: sim.techSupport,
      onlineSecurity: sim.onlineSecurity,
      streamingTV: sim.streamingTV,
      paperless: sim.paperless,
      risingBill: sim.risingBill,
      tickets: sim.tickets,
      senior: sim.senior,
      dependents: sim.dependents,
    }),
    [sim],
  );

  /* debounce backend calls so dragging a slider does not flood the API */
  useEffect(() => {
    const id = ++requestId.current;
    setPending(true);
    const timer = window.setTimeout(
      () => {
        adapter
          .simulate(profile)
          .then((res) => {
            if (requestId.current === id) setResult(res);
          })
          .catch((err) => {
            if (requestId.current === id) notify(err instanceof Error ? err.message : "Simulation failed", "error");
          })
          .finally(() => {
            if (requestId.current === id) setPending(false);
          });
      },
      adapter.info.mode === "api" ? 160 : 0,
    );
    return () => window.clearTimeout(timer);
  }, [adapter, profile, notify]);

  /* profile card used by the playbook and the displayed facts */
  const synthetic: Customer = useMemo(
    () => ({
      id: "SIM-PROFILE",
      gender: "Female",
      senior: sim.senior ? 1 : 0,
      dependents: sim.dependents,
      partner: false,
      tenure: sim.tenure,
      contract: sim.contract,
      paperless: sim.paperless,
      payment: sim.payment,
      internet: sim.internet,
      techSupport: sim.techSupport,
      onlineSecurity: sim.onlineSecurity,
      streamingTV: sim.streamingTV,
      tickets: sim.tickets,
      monthlyCharges: sim.monthlyCharges,
      totalCharges: sim.tenure > 0 ? sim.monthlyCharges * sim.tenure * (sim.risingBill ? 0.9 : 1) : 0,
      avgMonthly: sim.tenure > 0 ? sim.monthlyCharges * (sim.risingBill ? 0.9 : 1) : sim.monthlyCharges,
      risingBill: sim.risingBill,
      churned: false,
    }),
    [sim],
  );

  const probability = result?.probability ?? 0;
  const info = bandOf(probability);
  const cohortAvg = adapter.cohortAvg;
  const delta = probability - cohortAvg;

  const items = useMemo(
    () =>
      result
        ? adapter.featureNames
            .map((name, i) => ({ name: name.replace(/_/g, " "), value: result.contribs[i] ?? 0 }))
            .filter((it) => Math.abs(it.value) > 0.005)
            .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
        : [],
    [adapter, result],
  );
  const playbook = useMemo(
    () => (result ? buildPlaybook(synthetic, result.contribs, adapter.featureNames) : []),
    [synthetic, result, adapter],
  );

  const set = (patch: Partial<SimState>) => setSim((s) => ({ ...s, ...patch }));

  const randomize = () => {
    const rng = createRng(Date.now() % 100000);
    setSim({
      tenure: randInt(rng, 0, 72),
      monthlyCharges: Math.round(clamp(20 + rng.next() * 100, 18, 122)),
      contract: (["Month-to-month", "One year", "Two year"] as Contract[])[randInt(rng, 0, 2)],
      internet: (["None", "DSL", "Fiber optic"] as InternetService[])[randInt(rng, 0, 2)],
      payment: (["Electronic check", "Mailed check", "Bank transfer", "Credit card"] as PaymentMethod[])[randInt(rng, 0, 3)],
      techSupport: rng.next() < 0.4,
      onlineSecurity: rng.next() < 0.4,
      streamingTV: rng.next() < 0.5,
      paperless: rng.next() < 0.6,
      risingBill: rng.next() < 0.3,
      tickets: randInt(rng, 0, 7),
      senior: rng.next() < 0.16,
      dependents: rng.next() < 0.3,
    });
  };

  const saveScenario = () => {
    if (!result) return;
    const next: Scenario = {
      id: Date.now(),
      label: `S${scenarios.length + 1} · ${sim.contract.split(" ")[0]} · ${sim.tenure}mo`,
      state: sim,
      probability: result.probability,
    };
    setScenarios((s) => [next, ...s].slice(0, 6));
    notify(`Scenario saved at ${fmtPct(result.probability)} risk.`, "success");
  };

  return (
    <div className="anim-fade-up grid gap-3 xl:grid-cols-[380px_1fr]">
      <div className="panel h-fit p-4">
        <SectionHead
          title="Profile Builder"
          sub="Every change re-scores instantly"
          right={
            <div className="flex gap-1.5">
              <button
                onClick={randomize}
                className="flex items-center gap-1.5 rounded-md border border-line px-2.5 py-1.5 font-mono text-[11px] text-muted transition-colors hover:border-sky/50 hover:text-sky"
              >
                <IconRefresh size={13} /> Random
              </button>
              <button
                onClick={() => setSim(DEFAULT_SIM)}
                className="rounded-md border border-line px-2.5 py-1.5 font-mono text-[11px] text-muted transition-colors hover:border-rose/50 hover:text-rose"
              >
                Reset
              </button>
            </div>
          }
        />
        <div className="space-y-4">
          <SliderRow label="Tenure" value={sim.tenure} min={0} max={72} step={1} display={`${sim.tenure} mo`} onChange={(v) => set({ tenure: v })} />
          <SliderRow
            label="Monthly charges"
            value={sim.monthlyCharges}
            min={18}
            max={122}
            step={1}
            display={fmtMoney(sim.monthlyCharges)}
            onChange={(v) => set({ monthlyCharges: v })}
          />
          <div>
            <p className="mb-1.5 font-mono text-[10px] uppercase tracking-[0.12em] text-faint">Contract</p>
            <Segmented
              ariaLabel="Contract type"
              options={[
                { value: "Month-to-month", label: "Monthly" },
                { value: "One year", label: "1 year" },
                { value: "Two year", label: "2 year" },
              ]}
              value={sim.contract}
              onChange={(v) => set({ contract: v })}
            />
          </div>
          <div>
            <p className="mb-1.5 font-mono text-[10px] uppercase tracking-[0.12em] text-faint">Internet service</p>
            <Segmented
              ariaLabel="Internet service"
              options={[
                { value: "None", label: "None" },
                { value: "DSL", label: "DSL" },
                { value: "Fiber optic", label: "Fiber" },
              ]}
              value={sim.internet}
              onChange={(v) => set({ internet: v })}
            />
          </div>
          <div>
            <p className="mb-1.5 font-mono text-[10px] uppercase tracking-[0.12em] text-faint">Payment method</p>
            <Segmented
              ariaLabel="Payment method"
              options={[
                { value: "Electronic check", label: "E-check" },
                { value: "Mailed check", label: "Mail" },
                { value: "Bank transfer", label: "Bank" },
                { value: "Credit card", label: "Card" },
              ]}
              value={sim.payment}
              onChange={(v) => set({ payment: v })}
            />
          </div>
          <Stepper label="Support tickets (last quarter)" value={sim.tickets} min={0} max={8} onChange={(v) => set({ tickets: v })} />
          <div className="rounded-lg border border-linesoft bg-canvas/50 px-3 py-2">
            <Toggle checked={sim.techSupport} onChange={(v) => set({ techSupport: v })} label="Tech support add-on" />
            <Toggle checked={sim.onlineSecurity} onChange={(v) => set({ onlineSecurity: v })} label="Online security add-on" />
            <Toggle checked={sim.streamingTV} onChange={(v) => set({ streamingTV: v })} label="Streaming TV add-on" />
            <Toggle checked={sim.paperless} onChange={(v) => set({ paperless: v })} label="Paperless billing" />
            <Toggle checked={sim.risingBill} onChange={(v) => set({ risingBill: v })} label="Bill trending above average" />
            <Toggle checked={sim.senior} onChange={(v) => set({ senior: v })} label="Senior citizen" />
            <Toggle checked={sim.dependents} onChange={(v) => set({ dependents: v })} label="Has dependents" />
          </div>
          <button
            onClick={saveScenario}
            disabled={!result}
            className="flex w-full items-center justify-center gap-2 rounded-md border border-teal/40 bg-teal/10 px-3 py-2 font-mono text-[12px] font-medium text-teal transition-all hover:bg-teal/20 active:scale-[0.98] disabled:opacity-50"
          >
            <IconSpark size={14} /> Save scenario
          </button>
          {scenarios.length > 0 && (
            <div>
              <p className="mb-1.5 font-mono text-[10px] uppercase tracking-[0.12em] text-faint">Saved scenarios</p>
              <div className="flex flex-wrap gap-1.5">
                {scenarios.map((s) => {
                  const b = bandOf(s.probability);
                  return (
                    <button
                      key={s.id}
                      onClick={() => {
                        setSim(s.state);
                        notify(`Loaded scenario at ${fmtPct(s.probability)} risk.`, "info");
                      }}
                      className="rounded-full border border-line px-2.5 py-1 font-mono text-[10.5px] text-muted transition-colors hover:border-teal/50 hover:text-ink"
                    >
                      {s.label} · <span style={{ color: b.color }}>{fmtPct(s.probability, 0)}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="space-y-3">
        <div className="panel relative flex flex-col items-center px-4 pb-3 pt-5 sm:flex-row sm:items-center sm:gap-8 sm:px-8">
          {pending && (
            <span className="absolute right-4 top-4 flex items-center gap-1.5 rounded-full border border-line bg-canvas/70 px-2.5 py-1 font-mono text-[10px] text-faint">
              <IconRefresh size={11} className="anim-spin" />
              scoring via {adapter.info.mode === "api" ? "Python" : "browser"}…
            </span>
          )}
          <RiskGauge probability={probability} size={250} />
          <div className="mt-2 text-center sm:mt-0 sm:text-left">
            <p className="font-display text-4xl font-bold tabular-nums transition-colors duration-300" style={{ color: info.color }}>
              {fmtPct(probability)}
            </p>
            <p className="mt-1 flex items-center justify-center gap-2 sm:justify-start">
              <Chip color={info.color} soft={info.soft}>
                {info.label}
              </Chip>
            </p>
            <p className="mt-2 font-mono text-[11px] text-muted">
              cohort average {fmtPct(cohortAvg)} ·{" "}
              <span className={delta >= 0 ? "text-rose" : "text-teal"}>
                {delta >= 0 ? "+" : "−"}
                {fmtPct(Math.abs(delta))} vs average
              </span>
            </p>
          </div>
        </div>

        <div className="grid gap-3 lg:grid-cols-2">
          <div className="panel p-4">
            <SectionHead title="Live decomposition" sub="What pushes this profile up or down" />
            {result ? (
              <Waterfall base={result.base} items={items} probability={result.probability} />
            ) : (
              <p className="py-10 text-center font-mono text-[12px] text-faint">Scoring profile…</p>
            )}
          </div>
          <div className="panel p-4">
            <SectionHead title="Recommended plays" sub="Auto-generated from the top drivers" />
            <ol className="space-y-2">
              {playbook.map((a, i) => (
                <li key={a.title} className="anim-fade-up flex gap-3 rounded-lg border border-line bg-raised/60 px-3.5 py-3" style={{ animationDelay: `${i * 60}ms` }}>
                  <span className={`mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-full font-mono text-[11px] font-bold ${a.impact === "High" ? "bg-rose/15 text-rose" : "bg-amber/15 text-amber"}`}>
                    {i + 1}
                  </span>
                  <div>
                    <p className="text-[13px] font-semibold text-ink">{a.title}</p>
                    <p className="mt-0.5 text-[12px] leading-relaxed text-muted">{a.detail}</p>
                  </div>
                </li>
              ))}
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
}
