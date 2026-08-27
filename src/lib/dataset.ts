import { createRng, randInt, clamp } from "./random";

/* ========================================================================= */
/* SECTION: Types & Interfaces                                               */
/* ========================================================================= */

export type Contract = "Month-to-month" | "One year" | "Two year";
export type InternetService = "None" | "DSL" | "Fiber optic";
export type PaymentMethod = "Electronic check" | "Mailed check" | "Bank transfer" | "Credit card";

export interface Customer {
  id: string;
  gender: "Male" | "Female";
  senior: 0 | 1;
  dependents: boolean;
  partner: boolean;
  tenure: number;
  contract: Contract;
  paperless: boolean;
  payment: PaymentMethod;
  internet: InternetService;
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

export interface FeatureMatrix {
  columns: Float64Array[];  // 17 columns, each of length N
  names: string[];          // 17 feature names
}

/* ========================================================================= */
/* SECTION: Dataset Generators                                               */
/* ========================================================================= */

export function generateCustomers(seed: number, count: number): Customer[] {
  const rng = createRng(seed);
  const customers: Customer[] = [];

  for (let i = 0; i < count; i++) {
    const id = `CL-${(i + 1).toString().padStart(4, "0")}`;
    const gender = rng.next() > 0.5 ? "Male" : "Female";
    const senior = rng.next() < 0.16 ? 1 : 0;
    const partner = rng.next() > 0.5;
    const dependents = rng.next() < 0.3;

    // U-shaped tenure distribution (many short, many long)
    const isShort = rng.next() < 0.45;
    const tenure = isShort ? randInt(rng, 0, 18) : randInt(rng, 48, 72);

    // Contract usually tracks tenure
    let contract: Contract = "Month-to-month";
    if (tenure > 48) {
      contract = rng.next() > 0.4 ? "Two year" : "One year";
    } else if (tenure > 12) {
      contract = rng.next() > 0.6 ? "One year" : "Month-to-month";
    }

    const internet: InternetService = rng.next() < 0.22 ? "None" : (rng.next() < 0.55 ? "Fiber optic" : "DSL");
    const paperless = rng.next() < 0.6;
    
    const paymentMethods: PaymentMethod[] = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"];
    const payment = paymentMethods[randInt(rng, 0, 3)];

    const hasInternet = internet !== "None";
    const techSupport = hasInternet && rng.next() < (contract === "Month-to-month" ? 0.2 : 0.6);
    const onlineSecurity = hasInternet && rng.next() < 0.4;
    const streamingTV = hasInternet && rng.next() < 0.4;

    // Base pricing model
    let monthlyCharges = 20; // Base line
    if (internet === "DSL") monthlyCharges += 30;
    else if (internet === "Fiber optic") monthlyCharges += 50;

    if (techSupport) monthlyCharges += 5;
    if (onlineSecurity) monthlyCharges += 5;
    if (streamingTV) monthlyCharges += 15;
    
    // Add some noise to charges
    monthlyCharges += (rng.next() * 10 - 5);
    monthlyCharges = clamp(monthlyCharges, 18, 120);

    const totalCharges = monthlyCharges * tenure + (rng.next() * 50);
    const avgMonthly = tenure === 0 ? monthlyCharges : totalCharges / (tenure + 1);
    const risingBill = monthlyCharges > avgMonthly * 1.08;

    // Tickets distribution, heavily correlated with churn & issues
    const baseTickets = rng.next() < 0.7 ? randInt(rng, 0, 2) : randInt(rng, 3, 5);
    const tickets = clamp(baseTickets + (techSupport ? -1 : 1) + (risingBill ? 2 : 0), 0, 8);

    // Calculate churn probability based on features (logistic-like formulation)
    let logOdds = -1.5; 
    
    if (contract === "Month-to-month") logOdds += 1.8;
    if (contract === "Two year") logOdds -= 1.5;
    if (internet === "Fiber optic") logOdds += 0.8;
    if (techSupport) logOdds -= 0.6;
    if (onlineSecurity) logOdds -= 0.5;
    if (tenure < 12) logOdds += 1.0;
    if (tenure > 60) logOdds -= 1.2;
    if (tickets >= 4) logOdds += 1.5;
    if (paperless) logOdds += 0.3;
    
    const churnProb = 1 / (1 + Math.exp(-logOdds));
    const churned = rng.next() < churnProb;

    customers.push({
      id, gender, senior, dependents, partner, tenure, contract,
      paperless, payment, internet, techSupport, onlineSecurity,
      streamingTV, tickets, monthlyCharges, totalCharges, avgMonthly,
      risingBill, churned
    });
  }

  return customers;
}

/* ========================================================================= */
/* SECTION: Feature Extraction                                               */
/* ========================================================================= */

export function featureVectorFromCustomer(customer: Customer): Float64Array {
  const v = new Float64Array(17);
  v[0] = customer.tenure;
  v[1] = customer.monthlyCharges;
  v[2] = customer.avgMonthly;
  v[3] = customer.risingBill ? 1 : 0;
  v[4] = customer.contract === "Month-to-month" ? 1 : 0;
  v[5] = customer.contract === "One year" ? 1 : 0;
  v[6] = customer.internet === "Fiber optic" ? 1 : 0;
  v[7] = customer.internet === "DSL" ? 1 : 0;
  v[8] = customer.techSupport ? 1 : 0;
  v[9] = customer.onlineSecurity ? 1 : 0;
  v[10] = customer.streamingTV ? 1 : 0;
  v[11] = customer.paperless ? 1 : 0;
  v[12] = customer.payment === "Electronic check" ? 1 : 0;
  v[13] = customer.senior;
  v[14] = customer.dependents ? 1 : 0;
  v[15] = customer.tickets;
  v[16] = customer.tickets >= 4 ? 1 : 0;
  return v;
}

export function buildFeatureMatrix(customers: Customer[]): FeatureMatrix {
  const n = customers.length;
  const columns = Array.from({ length: 17 }, () => new Float64Array(n));
  
  for (let i = 0; i < n; i++) {
    const v = featureVectorFromCustomer(customers[i]);
    for (let j = 0; j < 17; j++) {
      columns[j][i] = v[j];
    }
  }

  const names = [
    "tenure",
    "monthly_charges",
    "avg_monthly_charge",
    "rising_bill",
    "contract_month_to_month",
    "contract_one_year",
    "internet_fiber",
    "internet_dsl",
    "tech_support",
    "online_security",
    "streaming_tv",
    "paperless_billing",
    "payment_echeck",
    "senior_citizen",
    "has_dependents",
    "support_tickets",
    "tickets_overloaded"
  ];

  return { columns, names };
}

export function featureVector(matrix: FeatureMatrix, index: number): Float64Array {
  const v = new Float64Array(17);
  for (let j = 0; j < 17; j++) {
    v[j] = matrix.columns[j][index];
  }
  return v;
}
