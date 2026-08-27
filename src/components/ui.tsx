/* ============= */
/* SECTION: Imports */
/* ============= */
import React, { useEffect, useState } from "react";

/* ============= */
/* SECTION: Icons */
/* ============= */
export interface IconProps { size?: number; className?: string; }

export function IconRadar({ size = 24, className = "" }: IconProps) {
  return (
    <svg width={size} height={size} className={className} viewBox="0 0 24 24" stroke="currentColor" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <circle cx="12" cy="12" r="6" />
      <circle cx="12" cy="12" r="2" />
      <line x1="12" y1="2" x2="12" y2="22" />
      <line x1="2" y1="12" x2="22" y2="12" />
    </svg>
  );
}

export function IconGrid({ size = 24, className = "" }: IconProps) {
  return (
    <svg width={size} height={size} className={className} viewBox="0 0 24 24" stroke="currentColor" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="7" height="7" />
      <rect x="14" y="3" width="7" height="7" />
      <rect x="14" y="14" width="7" height="7" />
      <rect x="3" y="14" width="7" height="7" />
    </svg>
  );
}

export function IconUsers({ size = 24, className = "" }: IconProps) {
  return (
    <svg width={size} height={size} className={className} viewBox="0 0 24 24" stroke="currentColor" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
      <circle cx="9" cy="7" r="4" />
      <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
      <path d="M16 3.13a4 4 0 0 1 0 7.75" />
    </svg>
  );
}

export function IconFlask({ size = 24, className = "" }: IconProps) {
  return (
    <svg width={size} height={size} className={className} viewBox="0 0 24 24" stroke="currentColor" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 3h6" />
      <path d="M10 9l-6 8a2 2 0 0 0 1.5 3.5h13a2 2 0 0 0 1.5-3.5l-6-8V3z" />
    </svg>
  );
}

export function IconRefresh({ size = 24, className = "" }: IconProps) {
  return (
    <svg width={size} height={size} className={className} viewBox="0 0 24 24" stroke="currentColor" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
      <path d="M3 3v5h5" />
    </svg>
  );
}

export function IconArrowUp({ size = 24, className = "" }: IconProps) {
  return (
    <svg width={size} height={size} className={className} viewBox="0 0 24 24" stroke="currentColor" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="19" x2="12" y2="5" />
      <polyline points="5 12 12 5 19 12" />
    </svg>
  );
}

export function IconArrowDown({ size = 24, className = "" }: IconProps) {
  return (
    <svg width={size} height={size} className={className} viewBox="0 0 24 24" stroke="currentColor" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="5" x2="12" y2="19" />
      <polyline points="19 12 12 19 5 12" />
    </svg>
  );
}

export function IconSpark({ size = 24, className = "" }: IconProps) {
  return (
    <svg width={size} height={size} className={className} viewBox="0 0 24 24" stroke="currentColor" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 3c0 4.97-4.03 9-9 9 4.97 0 9 4.03 9 9 0-4.97 4.03-9 9-9-4.97 0-9-4.03-9-9z" />
    </svg>
  );
}

export function IconX({ size = 24, className = "" }: IconProps) {
  return (
    <svg width={size} height={size} className={className} viewBox="0 0 24 24" stroke="currentColor" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}

export function IconPulse({ size = 24, className = "" }: IconProps) {
  return (
    <svg width={size} height={size} className={className} viewBox="0 0 24 24" stroke="currentColor" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
    </svg>
  );
}

/* ============= */
/* SECTION: Components */
/* ============= */

export type ToastKind = "success" | "info" | "error";
export interface ToastItem { id: number; kind: ToastKind; message: string; }

export function ToastHost({ toasts, onDismiss }: { toasts: ToastItem[]; onDismiss: (id: number) => void }) {
  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2">
      {toasts.map(t => {
        const borderClass = t.kind === "success" ? "border-teal-500" : t.kind === "info" ? "border-sky-500" : "border-rose-500";
        return (
          <div key={t.id} className={`toast-in bg-canvas border-l-4 ${borderClass} border-y border-r border-line p-4 rounded shadow-lg flex items-center justify-between min-w-[250px]`}>
            <span className="text-sm font-medium">{t.message}</span>
            <button onClick={() => onDismiss(t.id)} className="text-muted hover:text-white transition-colors">
              <IconX size={16} />
            </button>
          </div>
        );
      })}
    </div>
  );
}

export function Chip({ color, soft, children }: { color: string; soft: string; children: React.ReactNode }) {
  return (
    <span 
      className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium" 
      style={{ backgroundColor: soft, color: color }}
    >
      {children}
    </span>
  );
}

export function EmptyState({ title, hint, action }: { title: string; hint: string; action?: React.ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center p-8 text-center border border-dashed border-line rounded-lg">
      <h3 className="text-lg font-medium text-white mb-2">{title}</h3>
      <p className="text-sm text-muted mb-4">{hint}</p>
      {action && <div>{action}</div>}
    </div>
  );
}

export function SectionHead({ title, sub, right }: { title: string; sub: string; right?: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between mb-4 panel-title">
      <div>
        <h2 className="text-xl font-semibold text-white">{title}</h2>
        <p className="text-sm text-muted">{sub}</p>
      </div>
      {right && <div>{right}</div>}
    </div>
  );
}

export function CountUp({ value, format }: { value: number; format: (v: number) => string }) {
  const [display, setDisplay] = useState(0);
  
  useEffect(() => {
    let start = display;
    const end = value;
    const duration = 800;
    const startTime = performance.now();
    
    let frame: number;
    const animate = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const ease = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
      setDisplay(start + (end - start) * ease);
      
      if (progress < 1) {
        frame = requestAnimationFrame(animate);
      }
    };
    
    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, [value]);
  
  return <span>{format(display)}</span>;
}

export function SliderRow({ label, value, min, max, step, display, onChange }: { label: string; value: number; min: number; max: number; step: number; display: string; onChange: (v: number) => void }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className="flex flex-col gap-2 w-full">
      <div className="flex justify-between items-center text-sm">
        <span className="font-medium text-white">{label}</span>
        <span className="text-muted font-mono">{display}</span>
      </div>
      <input 
        type="range" 
        min={min} max={max} step={step} 
        value={value} 
        onChange={e => onChange(parseFloat(e.target.value))}
        className="w-full accent-teal-500"
        style={{ "--fill": `${pct}%` } as React.CSSProperties}
      />
    </div>
  );
}

export function Segmented({ ariaLabel, options, value, onChange }: { ariaLabel: string; options: { value: string; label: string }[]; value: string; onChange: (v: string) => void }) {
  return (
    <div role="group" aria-label={ariaLabel} className="inline-flex rounded bg-canvas border border-line p-1">
      {options.map(opt => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            onClick={() => onChange(opt.value)}
            className={`px-3 py-1 text-sm rounded transition-colors ${active ? "bg-teal-500 text-white font-medium shadow" : "text-muted hover:text-white"}`}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

export function Stepper({ label, value, min, max, onChange }: { label: string; value: number; min: number; max: number; onChange: (v: number) => void }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-line last:border-0">
      <span className="text-sm font-medium text-white">{label}</span>
      <div className="flex items-center gap-3">
        <button 
          onClick={() => onChange(Math.max(min, value - 1))}
          disabled={value <= min}
          className="w-6 h-6 flex items-center justify-center rounded border border-line bg-canvas text-muted hover:text-white disabled:opacity-50"
        >
          −
        </button>
        <span className="text-sm font-mono w-6 text-center">{value}</span>
        <button 
          onClick={() => onChange(Math.min(max, value + 1))}
          disabled={value >= max}
          className="w-6 h-6 flex items-center justify-center rounded border border-line bg-canvas text-muted hover:text-white disabled:opacity-50"
        >
          +
        </button>
      </div>
    </div>
  );
}

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <label className="flex items-center justify-between cursor-pointer py-2">
      <span className="text-sm font-medium text-white">{label}</span>
      <div className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${checked ? "bg-teal-500" : "bg-slate-700"}`}>
        <span className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${checked ? "translate-x-5" : "translate-x-1"}`} />
      </div>
      <input type="checkbox" className="sr-only" checked={checked} onChange={e => onChange(e.target.checked)} />
    </label>
  );
}
