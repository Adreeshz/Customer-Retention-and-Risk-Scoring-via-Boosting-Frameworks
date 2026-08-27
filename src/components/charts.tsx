import { useEffect, useState } from "react";
import { curvePath } from "../hooks/useModel";
import type { Point } from "../lib/analytics";
import { fmtPct, fmtSigned } from "../lib/analytics";

/* ============================================================================ */
/* SECTION: CURVE CHART — generic ROC / Precision-Recall plot with axes, area   */
/* fill, optional diagonal reference and stroke-draw animation on mount         */
/* ============================================================================ */
export function CurveChart({
  points,
  color,
  areaFrom,
  diagonal,
  xLabel,
  yLabel,
  metric,
  metricValue,
}: {
  points: Point[];
  color: string;
  areaFrom?: "zero" | "curve";
  diagonal?: boolean;
  xLabel: string;
  yLabel: string;
  metric: string;
  metricValue: number;
}) {
  const W = 300;
  const H = 210;
  const [dash, setDash] = useState(1400);
  const d = curvePath(points, W, H, 8);
  const area = `${d} L${W - 8},${H - 8} L8,${H - 8} Z`;

  useEffect(() => {
    const el = document.getElementById(`path-${metric}`) as SVGPathElement | null;
    if (el) setDash(Math.ceil(el.getTotalLength()) + 10);
  }, [metric, points]);

  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <span className="font-mono text-[11px] text-faint">
          {xLabel} × {yLabel}
        </span>
        <span className="font-mono text-[12px] font-semibold" style={{ color }}>
          {metric} {fmtPct(metricValue)}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`${metric} curve`}>
        <defs>
          <linearGradient id={`grad-${metric}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.28" />
            <stop offset="100%" stopColor={color} stopOpacity="0.02" />
          </linearGradient>
        </defs>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={8} y1={H - 8 - t * (H - 16)} x2={W - 8} y2={H - 8 - t * (H - 16)} stroke="#1a2740" strokeWidth="1" />
            <line x1={8 + t * (W - 16)} y1={8} x2={8 + t * (W - 16)} y2={H - 8} stroke="#1a2740" strokeWidth="1" />
            <text x={8 + t * (W - 16)} y={H - 8 + 12} textAnchor="middle" className="tick-label">
              {t}
            </text>
          </g>
        ))}
        {diagonal && <line x1={8} y1={H - 8} x2={W - 8} y2={8} stroke="#5c6c8c" strokeWidth="1" strokeDasharray="4 5" opacity="0.6" />}
        {areaFrom === "curve" && <path d={area} fill={`url(#grad-${metric})`} />}
        <path id={`path-${metric}`} d={d} fill="none" stroke={color} strokeWidth="2.2" className="anim-draw" style={{ ["--dash" as string]: dash }} />
      </svg>
    </div>
  );
}

/* ============================================================================ */
/* SECTION: IMPORTANCE BARS — gain-based feature importance with rank coloring  */
/* ============================================================================ */
export function ImportanceBars({ rows }: { rows: { name: string; value: number }[] }) {
  const max = Math.max(1e-9, ...rows.map((r) => r.value));
  return (
    <ul className="space-y-2">
      {rows.map((r, i) => (
        <li key={r.name} className="grid grid-cols-[130px_1fr_44px] items-center gap-2.5">
          <span className="truncate font-mono text-[11px] text-muted" title={r.name}>
            {r.name}
          </span>
          <span className="relative h-[9px] overflow-hidden rounded-full bg-canvas">
            <span
              className="anim-grow-x absolute inset-y-0 left-0 rounded-full"
              style={{
                width: `${(r.value / max) * 100}%`,
                background: i === 0 ? "#f2637a" : i === 1 ? "#f2b33d" : "#33d6ae",
                opacity: 0.9 - i * 0.03,
                animationDelay: `${i * 45}ms`,
              }}
            />
          </span>
          <span className="text-right font-mono text-[11px] text-ink tabular-nums">{fmtPct(r.value)}</span>
        </li>
      ))}
    </ul>
  );
}

/* ============================================================================ */
/* SECTION: TENURE CHART — stacked stayed-vs-churned bars per tenure cohort     */
/* with the churn-rate trend line overlaid                                      */
/* ============================================================================ */
export function TenureChart({ buckets }: { buckets: { label: string; stayed: number; churned: number; rate: number }[] }) {
  const W = 340;
  const H = 200;
  const padL = 10;
  const padB = 22;
  const maxCount = Math.max(1, ...buckets.map((b) => b.stayed + b.churned));
  const bw = (W - padL * 2) / buckets.length;

  const ratePoints = buckets
    .map((b, i) => `${padL + i * bw + bw / 2},${H - padB - b.rate * (H - padB - 14)}`)
    .join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Churn rate by tenure cohort">
      {buckets.map((b, i) => {
        const total = b.stayed + b.churned;
        const hChurn = (b.churned / maxCount) * (H - padB - 14);
        const hStay = (b.stayed / maxCount) * (H - padB - 14);
        const x = padL + i * bw + 3;
        return (
          <g key={b.label}>
            <rect x={x} y={H - padB - hStay - hChurn} width={bw - 6} height={hChurn} rx="3" fill="#f2637a" opacity="0.85" />
            <rect x={x} y={H - padB - hStay} width={bw - 6} height={hStay} rx="3" fill="#33d6ae" opacity="0.28" />
            {i % 2 === 0 && (
              <text x={x + (bw - 6) / 2} y={H - 8} textAnchor="middle" className="tick-label">
                {b.label}
              </text>
            )}
            <text x={x + (bw - 6) / 2} y={H - padB - hStay - hChurn - 5} textAnchor="middle" className="tick-label" fill="#f2b33d">
              {total > 0 ? `${Math.round(b.rate * 100)}` : ""}
            </text>
          </g>
        );
      })}
      <polyline points={ratePoints} fill="none" stroke="#f2b33d" strokeWidth="1.8" strokeDasharray="3 4" opacity="0.8" />
    </svg>
  );
}

/* ============================================================================ */
/* SECTION: CONFUSION GRID — threshold-driven 2×2 outcome tiles with derived    */
/* rates; FP highlighted amber and FN highlighted rose for quick triage         */
/* ============================================================================ */
export function ConfusionGrid({ tp, fp, tn, fn }: { tp: number; fp: number; tn: number; fn: number }) {
  const cell = (label: string, value: number, color: string, soft: string, note: string) => (
    <div className="rounded-lg border border-line px-3 py-2.5 text-center" style={{ background: soft }}>
      <p className="font-mono text-[10px] uppercase tracking-wider" style={{ color }}>
        {label}
      </p>
      <p className="font-display text-xl font-bold text-ink tabular-nums">{value}</p>
      <p className="text-[10px] text-faint">{note}</p>
    </div>
  );
  return (
    <div className="grid grid-cols-2 gap-2">
      {cell("TP", tp, "#33d6ae", "rgba(51,214,174,0.08)", "caught churners")}
      {cell("FP", fp, "#f2b33d", "rgba(242,179,61,0.08)", "false alarms")}
      {cell("FN", fn, "#f2637a", "rgba(242,99,122,0.1)", "missed churners")}
      {cell("TN", tn, "#5fa8f2", "rgba(95,168,242,0.07)", "correctly kept")}
    </div>
  );
}

/* ============================================================================ */
/* SECTION: RISK GAUGE — semicircular dial with a gradient arc and an eased     */
/* needle that animates whenever the scored probability changes                 */
/* ============================================================================ */
export function RiskGauge({ probability, size = 240 }: { probability: number; size?: number }) {
  const H = size * 0.62;
  const cx = size / 2;
  const cy = H - 18;
  const r = size / 2 - 16;
  const polar = (angleDeg: number, radius: number) => {
    const rad = ((angleDeg - 180) * Math.PI) / 180;
    return { x: cx + radius * Math.cos(rad), y: cy + radius * Math.sin(rad) };
  };
  const arc = (from: number, to: number, radius: number) => {
    const a = polar(from, radius);
    const b = polar(to, radius);
    return `M${a.x},${a.y} A${radius},${radius} 0 ${to - from > 180 ? 1 : 0} 1 ${b.x},${b.y}`;
  };
  const clamped = Math.min(1, Math.max(0, probability));
  const angle = clamped * 180;

  return (
    <svg viewBox={`0 0 ${size} ${H}`} style={{ width: size, maxWidth: "100%" }} role="img" aria-label={`Risk gauge at ${Math.round(clamped * 100)} percent`}>
      <defs>
        <linearGradient id="gauge-grad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#33d6ae" />
          <stop offset="55%" stopColor="#f2b33d" />
          <stop offset="100%" stopColor="#f2637a" />
        </linearGradient>
      </defs>
      <path d={arc(0, 180, r)} fill="none" stroke="#1a2740" strokeWidth="13" strokeLinecap="round" />
      <path d={arc(0, 180, r)} fill="none" stroke="url(#gauge-grad)" strokeWidth="13" strokeLinecap="round" opacity="0.92" />
      {[0, 25, 50, 75, 100].map((t) => {
        const p1 = polar((t / 100) * 180, r - 12);
        const p2 = polar((t / 100) * 180, r - 19);
        const pt = polar((t / 100) * 180, r - 30);
        return (
          <g key={t}>
            <line x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y} stroke="#5c6c8c" strokeWidth="1.5" />
            <text x={pt.x} y={pt.y + 3} textAnchor="middle" className="tick-label">
              {t}
            </text>
          </g>
        );
      })}
      <g className="needle-rotate" style={{ transform: `rotate(${angle}deg)`, transformOrigin: `${cx}px ${cy}px` }}>
        <line x1={cx} y1={cy} x2={cx} y2={cy - (r - 26)} stroke="#e8eefa" strokeWidth="2.6" strokeLinecap="round" />
      </g>
      <circle cx={cx} cy={cy} r="7" fill="#141f36" stroke="#e8eefa" strokeWidth="2.4" />
    </svg>
  );
}

/* ============================================================================ */
/* SECTION: CONTRIBUTION WATERFALL — base log-odds plus each feature effect     */
/* rendered as cumulative floating bars that always reconcile to the final risk */
/* ============================================================================ */
export interface WaterfallItem {
  name: string;
  value: number;
}

export function Waterfall({ base, items, probability }: { base: number; items: WaterfallItem[]; probability: number }) {
  const shown = items.slice(0, 9);
  const rest = items.slice(9).reduce((sum, it) => sum + it.value, 0);
  const rows: WaterfallItem[] = rest !== 0 ? [...shown, { name: "all other features", value: rest }] : shown;

  const bounds: number[] = [0, base];
  let cum = base;
  for (const r of rows) {
    cum += r.value;
    bounds.push(cum);
  }
  const lo = Math.min(...bounds) - 0.25;
  const hi = Math.max(...bounds) + 0.25;
  const scale = (v: number) => ((v - lo) / (hi - lo)) * 100;

  const bar = (from: number, to: number, color: string, delay: number) => {
    const left = scale(Math.min(from, to));
    const width = Math.max(0.6, Math.abs(scale(to) - scale(from)));
    return (
      <span
        className="anim-grow-x absolute top-1/2 h-[13px] -translate-y-1/2 rounded-[3px]"
        style={{ left: `${left}%`, width: `${width}%`, background: color, animationDelay: `${delay}ms` }}
      />
    );
  };

  let running = base;
  return (
    <div>
      <div className="relative mb-1 h-5">
        <span className="absolute top-0 h-full w-px bg-faint/50" style={{ left: `${scale(0)}%` }} />
        <span className="absolute -translate-x-1/2 font-mono text-[10px] text-faint" style={{ left: `${scale(0)}%` }}>
          neutral
        </span>
      </div>
      <ul className="space-y-[7px]">
        <li className="grid grid-cols-[150px_1fr_52px] items-center gap-2">
          <span className="truncate text-[12px] text-muted">base rate (intercept)</span>
          <span className="relative h-[18px] rounded bg-canvas/70">{bar(0, base, "#5fa8f2", 0)}</span>
          <span className="text-right font-mono text-[11px] text-ink tabular-nums">{fmtSigned(base)}</span>
        </li>
        {rows.map((r, i) => {
          const from = running;
          running += r.value;
          const color = r.value >= 0 ? "#f2637a" : "#33d6ae";
          return (
            <li key={r.name} className="grid grid-cols-[150px_1fr_52px] items-center gap-2">
              <span className="truncate text-[12px] text-muted" title={r.name}>
                {r.name}
              </span>
              <span className="relative h-[18px] rounded bg-canvas/70">
                {bar(from, running, r.value >= 0 ? "#f2637a" : "#33d6ae", 60 + i * 55)}
              </span>
              <span className="text-right font-mono text-[11px] tabular-nums" style={{ color }}>
                {fmtSigned(r.value)}
              </span>
            </li>
          );
        })}
        <li className="grid grid-cols-[150px_1fr_52px] items-center gap-2 border-t border-line pt-2">
          <span className="text-[12px] font-semibold text-ink">churn probability</span>
          <span className="relative h-[18px] rounded bg-canvas/70">
            {bar(0, running, probability >= 0.5 ? "#f2637a" : probability >= 0.25 ? "#f2b33d" : "#33d6ae", 120 + rows.length * 55)}
          </span>
          <span className="text-right font-mono text-[12px] font-semibold text-ink tabular-nums">{fmtPct(probability, 1)}</span>
        </li>
      </ul>
      <p className="mt-2 text-[11px] leading-relaxed text-faint">
        Effects are log-odds contributions from the exact tree-path decomposition — they always sum to the final score.
      </p>
    </div>
  );
}


+++ src/components/charts.tsx (修改后)
import { useEffect, useState } from "react";
import { curvePath } from "../lib/browserTraining";
import type { Point } from "../lib/analytics";
import { fmtPct, fmtSigned } from "../lib/analytics";

/* ============================================================================ */
/* SECTION: CURVE CHART — generic ROC / Precision-Recall plot with axes, area   */
/* fill, optional diagonal reference and stroke-draw animation on mount         */
/* ============================================================================ */
export function CurveChart({
  points,
  color,
  areaFrom,
  diagonal,
  xLabel,
  yLabel,
  metric,
  metricValue,
}: {
  points: Point[];
  color: string;
  areaFrom?: "zero" | "curve";
  diagonal?: boolean;
  xLabel: string;
  yLabel: string;
  metric: string;
  metricValue: number;
}) {
  const W = 300;
  const H = 210;
  const [dash, setDash] = useState(1400);
  const d = curvePath(points, W, H, 8);
  const area = `${d} L${W - 8},${H - 8} L8,${H - 8} Z`;

  useEffect(() => {
    const el = document.getElementById(`path-${metric}`) as SVGPathElement | null;
    if (el) setDash(Math.ceil(el.getTotalLength()) + 10);
  }, [metric, points]);

  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <span className="font-mono text-[11px] text-faint">
          {xLabel} × {yLabel}
        </span>
        <span className="font-mono text-[12px] font-semibold" style={{ color }}>
          {metric} {fmtPct(metricValue)}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`${metric} curve`}>
        <defs>
          <linearGradient id={`grad-${metric}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.28" />
            <stop offset="100%" stopColor={color} stopOpacity="0.02" />
          </linearGradient>
        </defs>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={8} y1={H - 8 - t * (H - 16)} x2={W - 8} y2={H - 8 - t * (H - 16)} stroke="#1a2740" strokeWidth="1" />
            <line x1={8 + t * (W - 16)} y1={8} x2={8 + t * (W - 16)} y2={H - 8} stroke="#1a2740" strokeWidth="1" />
            <text x={8 + t * (W - 16)} y={H - 8 + 12} textAnchor="middle" className="tick-label">
              {t}
            </text>
          </g>
        ))}
        {diagonal && <line x1={8} y1={H - 8} x2={W - 8} y2={8} stroke="#5c6c8c" strokeWidth="1" strokeDasharray="4 5" opacity="0.6" />}
        {areaFrom === "curve" && <path d={area} fill={`url(#grad-${metric})`} />}
        <path id={`path-${metric}`} d={d} fill="none" stroke={color} strokeWidth="2.2" className="anim-draw" style={{ ["--dash" as string]: dash }} />
      </svg>
    </div>
  );
}

/* ============================================================================ */
/* SECTION: IMPORTANCE BARS — gain-based feature importance with rank coloring  */
/* ============================================================================ */
export function ImportanceBars({ rows }: { rows: { name: string; value: number }[] }) {
  const max = Math.max(1e-9, ...rows.map((r) => r.value));
  return (
    <ul className="space-y-2">
      {rows.map((r, i) => (
        <li key={r.name} className="grid grid-cols-[130px_1fr_44px] items-center gap-2.5">
          <span className="truncate font-mono text-[11px] text-muted" title={r.name}>
            {r.name}
          </span>
          <span className="relative h-[9px] overflow-hidden rounded-full bg-canvas">
            <span
              className="anim-grow-x absolute inset-y-0 left-0 rounded-full"
              style={{
                width: `${(r.value / max) * 100}%`,
                background: i === 0 ? "#f2637a" : i === 1 ? "#f2b33d" : "#33d6ae",
                opacity: 0.9 - i * 0.03,
                animationDelay: `${i * 45}ms`,
              }}
            />
          </span>
          <span className="text-right font-mono text-[11px] text-ink tabular-nums">{fmtPct(r.value)}</span>
        </li>
      ))}
    </ul>
  );
}

/* ============================================================================ */
/* SECTION: TENURE CHART — stacked stayed-vs-churned bars per tenure cohort     */
/* with the churn-rate trend line overlaid                                      */
/* ============================================================================ */
export function TenureChart({ buckets }: { buckets: { label: string; stayed: number; churned: number; rate: number }[] }) {
  const W = 340;
  const H = 200;
  const padL = 10;
  const padB = 22;
  const maxCount = Math.max(1, ...buckets.map((b) => b.stayed + b.churned));
  const bw = (W - padL * 2) / buckets.length;

  const ratePoints = buckets
    .map((b, i) => `${padL + i * bw + bw / 2},${H - padB - b.rate * (H - padB - 14)}`)
    .join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Churn rate by tenure cohort">
      {buckets.map((b, i) => {
        const total = b.stayed + b.churned;
        const hChurn = (b.churned / maxCount) * (H - padB - 14);
        const hStay = (b.stayed / maxCount) * (H - padB - 14);
        const x = padL + i * bw + 3;
        return (
          <g key={b.label}>
            <rect x={x} y={H - padB - hStay - hChurn} width={bw - 6} height={hChurn} rx="3" fill="#f2637a" opacity="0.85" />
            <rect x={x} y={H - padB - hStay} width={bw - 6} height={hStay} rx="3" fill="#33d6ae" opacity="0.28" />
            {i % 2 === 0 && (
              <text x={x + (bw - 6) / 2} y={H - 8} textAnchor="middle" className="tick-label">
                {b.label}
              </text>
            )}
            <text x={x + (bw - 6) / 2} y={H - padB - hStay - hChurn - 5} textAnchor="middle" className="tick-label" fill="#f2b33d">
              {total > 0 ? `${Math.round(b.rate * 100)}` : ""}
            </text>
          </g>
        );
      })}
      <polyline points={ratePoints} fill="none" stroke="#f2b33d" strokeWidth="1.8" strokeDasharray="3 4" opacity="0.8" />
    </svg>
  );
}

/* ============================================================================ */
/* SECTION: CONFUSION GRID — threshold-driven 2×2 outcome tiles with derived    */
/* rates; FP highlighted amber and FN highlighted rose for quick triage         */
/* ============================================================================ */
export function ConfusionGrid({ tp, fp, tn, fn }: { tp: number; fp: number; tn: number; fn: number }) {
  const cell = (label: string, value: number, color: string, soft: string, note: string) => (
    <div className="rounded-lg border border-line px-3 py-2.5 text-center" style={{ background: soft }}>
      <p className="font-mono text-[10px] uppercase tracking-wider" style={{ color }}>
        {label}
      </p>
      <p className="font-display text-xl font-bold text-ink tabular-nums">{value}</p>
      <p className="text-[10px] text-faint">{note}</p>
    </div>
  );
  return (
    <div className="grid grid-cols-2 gap-2">
      {cell("TP", tp, "#33d6ae", "rgba(51,214,174,0.08)", "caught churners")}
      {cell("FP", fp, "#f2b33d", "rgba(242,179,61,0.08)", "false alarms")}
      {cell("FN", fn, "#f2637a", "rgba(242,99,122,0.1)", "missed churners")}
      {cell("TN", tn, "#5fa8f2", "rgba(95,168,242,0.07)", "correctly kept")}
    </div>
  );
}

/* ============================================================================ */
/* SECTION: RISK GAUGE — semicircular dial with a gradient arc and an eased     */
/* needle that animates whenever the scored probability changes                 */
/* ============================================================================ */
export function RiskGauge({ probability, size = 240 }: { probability: number; size?: number }) {
  const H = size * 0.62;
  const cx = size / 2;
  const cy = H - 18;
  const r = size / 2 - 16;
  const polar = (angleDeg: number, radius: number) => {
    const rad = ((angleDeg - 180) * Math.PI) / 180;
    return { x: cx + radius * Math.cos(rad), y: cy + radius * Math.sin(rad) };
  };
  const arc = (from: number, to: number, radius: number) => {
    const a = polar(from, radius);
    const b = polar(to, radius);
    return `M${a.x},${a.y} A${radius},${radius} 0 ${to - from > 180 ? 1 : 0} 1 ${b.x},${b.y}`;
  };
  const clamped = Math.min(1, Math.max(0, probability));
  const angle = clamped * 180;

  return (
    <svg viewBox={`0 0 ${size} ${H}`} style={{ width: size, maxWidth: "100%" }} role="img" aria-label={`Risk gauge at ${Math.round(clamped * 100)} percent`}>
      <defs>
        <linearGradient id="gauge-grad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#33d6ae" />
          <stop offset="55%" stopColor="#f2b33d" />
          <stop offset="100%" stopColor="#f2637a" />
        </linearGradient>
      </defs>
      <path d={arc(0, 180, r)} fill="none" stroke="#1a2740" strokeWidth="13" strokeLinecap="round" />
      <path d={arc(0, 180, r)} fill="none" stroke="url(#gauge-grad)" strokeWidth="13" strokeLinecap="round" opacity="0.92" />
      {[0, 25, 50, 75, 100].map((t) => {
        const p1 = polar((t / 100) * 180, r - 12);
        const p2 = polar((t / 100) * 180, r - 19);
        const pt = polar((t / 100) * 180, r - 30);
        return (
          <g key={t}>
            <line x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y} stroke="#5c6c8c" strokeWidth="1.5" />
            <text x={pt.x} y={pt.y + 3} textAnchor="middle" className="tick-label">
              {t}
            </text>
          </g>
        );
      })}
      <g className="needle-rotate" style={{ transform: `rotate(${angle}deg)`, transformOrigin: `${cx}px ${cy}px` }}>
        <line x1={cx} y1={cy} x2={cx} y2={cy - (r - 26)} stroke="#e8eefa" strokeWidth="2.6" strokeLinecap="round" />
      </g>
      <circle cx={cx} cy={cy} r="7" fill="#141f36" stroke="#e8eefa" strokeWidth="2.4" />
    </svg>
  );
}

/* ============================================================================ */
/* SECTION: CONTRIBUTION WATERFALL — base log-odds plus each feature effect     */
/* rendered as cumulative floating bars that always reconcile to the final risk */
/* ============================================================================ */
export interface WaterfallItem {
  name: string;
  value: number;
}

export function Waterfall({ base, items, probability }: { base: number; items: WaterfallItem[]; probability: number }) {
  const shown = items.slice(0, 9);
  const rest = items.slice(9).reduce((sum, it) => sum + it.value, 0);
  const rows: WaterfallItem[] = rest !== 0 ? [...shown, { name: "all other features", value: rest }] : shown;

  const bounds: number[] = [0, base];
  let cum = base;
  for (const r of rows) {
    cum += r.value;
    bounds.push(cum);
  }
  const lo = Math.min(...bounds) - 0.25;
  const hi = Math.max(...bounds) + 0.25;
  const scale = (v: number) => ((v - lo) / (hi - lo)) * 100;

  const bar = (from: number, to: number, color: string, delay: number) => {
    const left = scale(Math.min(from, to));
    const width = Math.max(0.6, Math.abs(scale(to) - scale(from)));
    return (
      <span
        className="anim-grow-x absolute top-1/2 h-[13px] -translate-y-1/2 rounded-[3px]"
        style={{ left: `${left}%`, width: `${width}%`, background: color, animationDelay: `${delay}ms` }}
      />
    );
  };

  let running = base;
  return (
    <div>
      <div className="relative mb-1 h-5">
        <span className="absolute top-0 h-full w-px bg-faint/50" style={{ left: `${scale(0)}%` }} />
        <span className="absolute -translate-x-1/2 font-mono text-[10px] text-faint" style={{ left: `${scale(0)}%` }}>
          neutral
        </span>
      </div>
      <ul className="space-y-[7px]">
        <li className="grid grid-cols-[150px_1fr_52px] items-center gap-2">
          <span className="truncate text-[12px] text-muted">base rate (intercept)</span>
          <span className="relative h-[18px] rounded bg-canvas/70">{bar(0, base, "#5fa8f2", 0)}</span>
          <span className="text-right font-mono text-[11px] text-ink tabular-nums">{fmtSigned(base)}</span>
        </li>
        {rows.map((r, i) => {
          const from = running;
          running += r.value;
          const color = r.value >= 0 ? "#f2637a" : "#33d6ae";
          return (
            <li key={r.name} className="grid grid-cols-[150px_1fr_52px] items-center gap-2">
              <span className="truncate text-[12px] text-muted" title={r.name}>
                {r.name}
              </span>
              <span className="relative h-[18px] rounded bg-canvas/70">
                {bar(from, running, r.value >= 0 ? "#f2637a" : "#33d6ae", 60 + i * 55)}
              </span>
              <span className="text-right font-mono text-[11px] tabular-nums" style={{ color }}>
                {fmtSigned(r.value)}
              </span>
            </li>
          );
        })}
        <li className="grid grid-cols-[150px_1fr_52px] items-center gap-2 border-t border-line pt-2">
          <span className="text-[12px] font-semibold text-ink">churn probability</span>
          <span className="relative h-[18px] rounded bg-canvas/70">
            {bar(0, running, probability >= 0.5 ? "#f2637a" : probability >= 0.25 ? "#f2b33d" : "#33d6ae", 120 + rows.length * 55)}
          </span>
          <span className="text-right font-mono text-[12px] font-semibold text-ink tabular-nums">{fmtPct(probability, 1)}</span>
        </li>
      </ul>
      <p className="mt-2 text-[11px] leading-relaxed text-faint">
        Effects are log-odds contributions from the exact tree-path decomposition — they always sum to the final score.
      </p>
    </div>
  );
}
