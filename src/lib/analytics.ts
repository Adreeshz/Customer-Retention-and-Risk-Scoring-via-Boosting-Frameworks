/* ============= */
/* SECTION: Types */
/* ============= */
import { Customer } from "./dataset";

export type Band = "low" | "watch" | "high";
export interface Point { x: number; y: number; }
export interface RankedScores {
  roc: Point[];
  pr: Point[];
  rocAuc: number;
  prAuc: number;
}
export interface Confusion {
  tp: number; fp: number; tn: number; fn: number;
  precision: number; recall: number; f1: number; accuracy: number;
}
export interface TenureBucket {
  label: string;
  stayed: number;
  churned: number;
  rate: number;
}

/* ============= */
/* SECTION: Formatters */
/* ============= */
export function bandOf(prob: number): { band: Band; label: string; color: string; soft: string } {
  if (prob < 0.25) {
    return { band: "low", label: "Low risk", color: "#33d6ae", soft: "rgba(51,214,174,0.12)" };
  } else if (prob < 0.50) {
    return { band: "watch", label: "Watchlist", color: "#f2b33d", soft: "rgba(242,179,61,0.12)" };
  } else {
    return { band: "high", label: "High risk", color: "#f2637a", soft: "rgba(242,99,122,0.12)" };
  }
}

export function fmtPct(n: number, decimals: number = 1): string {
  return (n * 100).toFixed(decimals) + "%";
}

export function fmtMoney(n: number): string {
  return "$" + n.toFixed(2);
}

export function fmtSigned(n: number): string {
  if (n > 0) return "+" + n.toFixed(2);
  if (n < 0) return "−" + Math.abs(n).toFixed(2);
  return "0.00";
}

/* ============= */
/* SECTION: Analytics */
/* ============= */
export function confusionAt(labels: number[], scores: number[], threshold: number): Confusion {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (let i = 0; i < labels.length; i++) {
    const p = scores[i] >= threshold ? 1 : 0;
    const l = labels[i];
    if (p === 1 && l === 1) tp++;
    else if (p === 1 && l === 0) fp++;
    else if (p === 0 && l === 1) fn++;
    else if (p === 0 && l === 0) tn++;
  }
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  const accuracy = labels.length > 0 ? (tp + tn) / labels.length : 0;
  
  return { tp, fp, tn, fn, precision, recall, f1, accuracy };
}

export function rankScores(labels: number[], scores: number[]): RankedScores {
  const data = labels.map((l, i) => ({ label: l, score: scores[i] })).sort((a, b) => b.score - a.score);
  let totalPos = 0, totalNeg = 0;
  for (const d of data) {
    if (d.label === 1) totalPos++;
    else totalNeg++;
  }
  
  const roc: Point[] = [];
  const pr: Point[] = [];
  
  let tp = 0, fp = 0;
  let rocAuc = 0;
  let prevFpr = 0;
  let prAuc = 0;
  let prevRecall = 0;
  let prevPrecision = 1;
  
  for (let i = 0; i < data.length; i++) {
    if (data[i].label === 1) tp++;
    else fp++;
    
    const tpr = totalPos > 0 ? tp / totalPos : 0;
    const fpr = totalNeg > 0 ? fp / totalNeg : 0;
    const precision = tp + fp > 0 ? tp / (tp + fp) : 1;
    const recall = tpr;
    
    roc.push({ x: fpr, y: tpr });
    pr.push({ x: recall, y: precision });
    
    rocAuc += (fpr - prevFpr) * tpr;
    prAuc += (recall - prevRecall) * ((precision + prevPrecision) / 2);
    
    prevFpr = fpr;
    prevRecall = recall;
    prevPrecision = precision;
  }
  
  const downsample = (pts: Point[], maxPts: number) => {
    if (pts.length <= maxPts) return pts;
    const res: Point[] = [];
    const step = pts.length / maxPts;
    for (let i = 0; i < maxPts; i++) {
      res.push(pts[Math.floor(i * step)]);
    }
    return res;
  };
  
  return {
    roc: downsample(roc, 200),
    pr: downsample(pr, 200),
    rocAuc,
    prAuc
  };
}

export function tenureBuckets(customers: Customer[]): TenureBucket[] {
  const edges = [0, 9, 17, 25, 33, 41, 49, 57, 65, 73];
  const buckets: TenureBucket[] = [];
  
  for (let i = 0; i < edges.length - 1; i++) {
    const min = edges[i];
    const max = edges[i + 1];
    let stayed = 0;
    let churned = 0;
    
    for (const c of customers) {
      if (c.tenure >= min && (i === edges.length - 2 ? c.tenure <= max : c.tenure < max)) {
        if (c.churned) churned++;
        else stayed++;
      }
    }
    
    const total = stayed + churned;
    buckets.push({
      label: `${min}-${max}`,
      stayed,
      churned,
      rate: total > 0 ? churned / total : 0
    });
  }
  
  return buckets;
}

export function buildPlaybook(_customer: Customer, contribs: number[], featureNames: readonly string[]): { title: string; detail: string; impact: "High" | "Medium" }[] {
  const data = featureNames.map((name, i) => ({ name, val: contribs[i] })).sort((a, b) => b.val - a.val);
  const positiveDrivers = data.filter(d => d.val > 0).slice(0, 5);
  
  const items: { title: string; detail: string; impact: "High" | "Medium" }[] = [];
  
  for (let i = 0; i < positiveDrivers.length; i++) {
    const { name } = positiveDrivers[i];
    const impact: "High" | "Medium" = i < 2 ? "High" : "Medium";
    
    if (name.includes("contract_month_to_month")) {
      items.push({ title: "Upgrade Contract", detail: "Recommend switching from month-to-month to a 1 or 2-year contract with a discount.", impact });
    } else if (name.includes("tech_support")) {
      items.push({ title: "Offer Tech Support", detail: "Suggest adding tech support to improve service reliability and satisfaction.", impact });
    } else if (name.includes("internet_fiber")) {
      items.push({ title: "Review Fiber Pricing", detail: "Evaluate fiber service pricing or bundle options to improve perceived value.", impact });
    } else if (name.includes("payment_echeck")) {
      items.push({ title: "Switch Payment Method", detail: "Encourage setting up automatic payments or credit card billing.", impact });
    } else if (name.includes("tenure")) {
      items.push({ title: "Loyalty Incentive", detail: "Provide a loyalty discount or perk to establish a longer-term relationship.", impact });
    } else if (name.includes("rising_bill")) {
      items.push({ title: "Billing Review", detail: "Proactively discuss recent bill increases and optimize their plan.", impact });
    } else if (name.includes("support_tickets")) {
      items.push({ title: "Proactive Support", detail: "Reach out to resolve recurring issues and ensure service satisfaction.", impact });
    } else {
      items.push({ title: `Review ${name}`, detail: `Investigate and address issues related to ${name.replace(/_/g, " ")}.`, impact });
    }
  }
  
  return items.slice(0, 5);
}
