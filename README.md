--- README.md (原始)
# Customer Retention and Risk Scoring via Boosting Frameworks

> **ChurnLens** — an end-to-end churn prediction system delivered as an interactive web
> application. A real gradient-boosting ensemble (log-loss GBDT) is trained **in the browser**,
> scores every customer, explains each score with an exact SHAP-style decomposition, and turns
> the top risk drivers into a retention playbook.

---

## 1. Problem Statement

Acquiring new customers costs significantly more than retaining existing ones. This project
builds a churn prediction system using **customer usage logs, contract terms, support ticket
counts, and billing histories** to identify churn risk *before* cancellation occurs, so the
business can intervene with targeted retention strategies.

## 2. Dataset Selection

| Dataset | Verdict | Why |
| --- | --- | --- |
| **IBM Telco Customer Churn** (7,043 rows) | **Selected** | Mirrors every required dimension: `Contract` (terms), `MonthlyCharges`/`TotalCharges` (billing), `TechSupport`/`InternetService`/`StreamingTV` (usage), plus an engineered support-ticket dimension. Industry-standard, clean, ideal for an ML course. |
| Bank Customer Churn | Rejected | Lacks an explicit support-ticket dimension. |
| Kaggle SaaS/B2B churn | Rejected | Heavy missingness and time-series alignment overhead. |

Because the deployment target here is a web application, the app ships with a **deterministic
synthetic cohort of 2,400 customers** whose field distributions and churn correlations follow
the published Telco dataset shapes (U-shaped tenure, ~26 % churn rate, month-to-month / fiber /
e-check risk factors). Any seed reproduces the exact same cohort and model, which keeps results
reproducible for grading and demonstration. The same pipeline can be pointed at the real CSV by
swapping the generator in `src/lib/dataset.ts`.

### Business-dimension mapping

| Brief requirement | Fields used |
| --- | --- |
| Usage logs | `InternetService`, `StreamingTV`, `OnlineSecurity`, tenure |
| Contract terms | `Contract`, `PaperlessBilling`, `PaymentMethod` |
| Support tickets | `support_tickets` (quarterly count), `tickets_overloaded` |
| Billing histories | `MonthlyCharges`, `TotalCharges`, `avg_monthly_charge`, `rising_bill` |

## 3. Methodology

1. **Preprocessing** — categorical expansion into 17 numeric model features; `TotalCharges`
   parsed defensively; new customers (tenure 0) get `totalCharges = 0` and fall back to
   `avgMonthly = MonthlyCharges` (guarded `tenure + 1` division).
2. **Feature engineering** — `avg_monthly_charge`, `rising_bill` (current spend > 1.08 × historical
   average), `tickets_overloaded` (≥ 3 tickets/quarter), contract and service one-hots.
3. **Model** — stochastic gradient boosting on the logistic loss: 150 regression trees, depth 4,
   learning rate 0.085, 85 % row subsampling per tree, L2 leaf regularization (λ = 1.2), Newton-step
   leaf values. Implemented from scratch in TypeScript (`src/lib/boosting.ts`) with presorted
   split search.
4. **Evaluation** — stratified 80/20 split; **ROC-AUC and PR-AUC** (imbalance-aware) as primary
   metrics; F1 / precision / recall with a tunable decision threshold; confusion matrix;
   gain-based feature importance; churn-rate-by-tenure EDA.
5. **Interpretability** — exact additive decomposition per customer via **tree-path
   contributions**: at every split on the decision path, the child-value minus node-value change is
   attributed to the split feature. Contributions always sum to the model logit, giving a
   trustworthy "why this score" waterfall.
6. **Action layer** — a rule-based retention playbook keyed to each customer's strongest drivers
   (contract conversion, autopay migration, support escalation, rate-plan review, onboarding
   nurture), plus CSV export of every flagged account.

## 4. Project Structure

```text
├── index.html                  # Shell, fonts (Space Grotesk + IBM Plex), favicon
├── README.md                   # Single source of truth for documentation
└── src/
    ├── main.tsx                # React entry point
    ├── App.tsx                 # Shell: header, tabs, training/error states, toasts
    ├── index.css               # Tailwind v4 theme tokens, ambient background, motion
    ├── hooks/useModel.ts       # Pipeline orchestration: generate → fit → score → evaluate
    ├── lib/
    │   ├── random.ts           # Seeded PRNG (mulberry32) and distributions
    │   ├── dataset.ts          # Synthetic Telco-mirror cohort + feature matrix builder
    │   ├── boosting.ts         # GBDT trainer, tree-path SHAP decomposition, importance
    │   └── analytics.ts        # ROC/PR curves, AUC, confusion metrics, risk bands, playbook
    └── components/
        ├── ui.tsx              # Icons, toasts, segmented controls, sliders, CountUp
        ├── charts.tsx          # SVG curves, importance, tenure, confusion, gauge, waterfall
        ├── Overview.tsx        # Model performance workspace
        ├── Explorer.tsx        # Risk explorer: filters, table, drawer, CSV export
        └── Simulator.tsx       # What-if lab with live re-scoring and saved scenarios
```

## 5. Application Surfaces

- **Model Performance** — ROC/PR curves with AUC readouts, threshold-tunable confusion matrix,
  feature importance, tenure-vs-churn EDA, and full training facts.
- **Risk Explorer** — searchable, filterable, sortable table of all 2,400 scored customers with
  pagination; row drawer shows the risk gauge, SHAP-style waterfall, profile and playbook;
  one-click CSV export of flagged accounts.
- **What-If Lab** — build any customer profile and watch the score, decomposition and playbook
  recompute instantly; save up to six scenarios (persisted across reloads).

## 6. How to Run

```bash
npm install        # install dependencies
npm run dev        # local development server
npm run build      # production build (dist/)
```

## 7. Edge Cases Handled

| Case | Handling |
| --- | --- |
| New customers with no billing history | `totalCharges = 0`, average-charge fallback, `tenure + 1` guard |
| Class imbalance (~1 : 3) | PR-AUC as primary metric, tunable threshold, stratified split |
| Constant features | Split search skips them; trees degrade safely to stumps |
| Empty filter results | Dedicated empty state with one-click filter reset |
| Training failure | Error panel with retry; per-run cancellation guard in the hook |
| Corrupt/absent localStorage | Safe fallback to defaults for prefs and scenarios |
| Score ties in AUC | Points emitted only at distinct-score boundaries |
| Probability extremes | Numerically stable sigmoid; log-odds intercept clamped |

## 8. Future Work

- Fit on the real IBM CSV via a drag-and-drop loader and compare AUC deltas.
- Calibrate probabilities (Platt scaling / isotonic regression).
- Replace rules with uplift modeling to prioritize interventions by incremental value.
- Time-aware features (ticket velocity, charge trend slopes) with a survival-analysis baseline.


+++ README.md (修改后)
# Customer Retention and Risk Scoring via Boosting Frameworks

**ChurnLens** — a fully functioning churn prediction system. A Python backend
(`backend/*.py`) acquires and cleans the data, trains gradient-boosting models and
serves exact per-customer risk explanations over REST; a React/CSS frontend turns
those predictions into an interactive retention console. When the backend is not
running, the frontend degrades gracefully to an in-browser fallback engine that
implements the same pipeline in TypeScript, so the dashboard always works.

---

## 1. Problem Statement

Acquiring new customers costs significantly more than retaining existing ones.
This project builds a churn prediction system using **customer usage logs,
contract terms, support ticket counts and billing histories** to identify churn
risks *before* cancellation occurs, then explains every risk score so retention
teams can act on it.

## 2. Dataset

**Primary: Telco Customer Churn (IBM)** — 7,043 customers, 21 fields. The backend
downloads it automatically from public mirrors into `data/raw/` on first launch.

| Business dimension (brief) | Dataset columns |
| --- | --- |
| Contract terms | `Contract`, `PaperlessBilling`, `PaymentMethod` |
| Billing histories | `MonthlyCharges`, `TotalCharges` (+ engineered `AvgMonthlyCharge`, `RisingBill`) |
| Usage logs | `InternetService`, `StreamingTV`, `OnlineSecurity`, `TechSupport`, `tenure` |
| Support tickets | `TicketsLastQuarter` (synthetic mirror) — see proxy note below |
| Target | `Churn` (≈26.5% positive — realistic class imbalance) |

**Support-ticket note.** The real Telco file has no ticket column, so on real data
`SupportTickets` is a documented *support-gap proxy*:
`2 × (missing TechSupport + missing OnlineSecurity) + early-tenure flag` (0–5,
only for internet subscribers). The synthetic mirror contains real simulated
quarterly ticket counts (0–8). Either way the model consumes one unified
`support_tickets` feature — there is no schema disparity between sources.

**Fallback: synthetic mirror (`.xlsx`).** If every download mirror is unreachable,
`backend/synthetic_data_generator.py` writes
`data/raw/telco_churn_synthetic.xlsx` (2,000 rows, Telco-shaped distributions,
seeded and reproducible) and the pipeline continues from that workbook. Per
project policy, synthetic data artifacts are always stored as **`.xlsx`**; no
`.tar` files are used anywhere.

## 3. Architecture

```
┌────────────────────────────┐        REST (JSON)        ┌────────────────────────────┐
│  React + Tailwind frontend │ ◄───────────────────────► │  Python backend (FastAPI)  │
│  src/                      │   /api/status|metrics|    │  backend/                  │
│  · Model Performance tab   │   customers|simulate|     │  · data_loader.py          │
│  · Risk Explorer tab       │   retrain                 │  · features.py             │
│  · What-If Lab tab         │                           │  · model.py (sklearn GBC,  │
│  · in-browser fallback GBDT│                           │    XGBoost, LightGBM)      │
└────────────────────────────┘                           │  · shap_explain.py         │
                                                         └────────────────────────────┘
```

**Dual-mode contract.** Both engines expose one adapter interface
(`src/lib/adapter.ts`): scored-page queries, what-if simulation with additive
contributions, metrics and retraining. Components never know which engine serves
them, and both agree on the 17-feature vocabulary, risk-band cut-offs
(<0.25 low / <0.50 watch / else high) and tenure-bucket edges.

## 4. Methodology

1. **Acquisition** — cached CSV → mirrored downloads (validated ≥ 7,000 rows and
   a `Churn` column) → synthetic `.xlsx` fallback.
2. **Preprocessing** — coerce `TotalCharges` (blank for 11 brand-new customers →
   filled per the tenure-0 edge case), map `"No internet service"` sentinels,
   clamp ranges, encode categoricals, unify the schema.
3. **Feature engineering** — 17 features: billing ratios
   (`avg_monthly_charge = TotalCharges / (tenure+1)`, `rising_bill`), contract and
   service indicators, `support_tickets` and a `tickets_overloaded` flag.
4. **Modeling** — scikit-learn `GradientBoostingClassifier`
   (150 trees, depth 4, lr 0.085, 85% subsampling) is the scoring engine; XGBoost
   and LightGBM are trained on the identical split and compared when installed.
5. **Evaluation** — stratified 80/20 split; ROC-AUC and PR-AUC (imbalance-aware),
   threshold-tunable confusion matrix, F1/precision/recall, gain-based feature
   importance, churn-by-tenure EDA.
6. **Interpretability** — every score is decomposed by walking each tree's
   decision path (`backend/shap_explain.py`): per-feature log-odds effects that
   **sum exactly to the model logit** (SHAP-style additive explanation), rendered
   as waterfalls in the UI plus a rule-based retention playbook.

## 5. Project Structure

```text
├── backend/                      # Python machine-learning backend
│   ├── main.py                   # FastAPI REST service (uvicorn)
│   ├── model.py                  # training, evaluation, scoring, querying
│   ├── data_loader.py            # download / validate / unify the dataset
│   ├── synthetic_data_generator.py  # offline .xlsx fallback generator
│   ├── features.py               # shared 17-feature engineering matrix
│   ├── shap_explain.py           # exact tree-path contribution decomposition
│   ├── config.py                 # paths, URLs, hyperparameters, API settings
│   ├── run.py                    # `python run.py` service launcher
│   └── requirements.txt
├── src/                          # React + Tailwind frontend
│   ├── lib/                      # api client, adapter contract, analytics,
│   │                             # boosting engine (fallback), dataset, charts
│   ├── hooks/useSystem.ts        # backend-first boot with graceful fallback
│   └── components/               # Overview, Explorer, Simulator workspaces
├── data/raw/                     # downloaded CSV or synthetic .xlsx (generated)
├── data/processed/               # canonical cleaned frame (telco_churn_processed.xlsx)
└── README.md                     # single source of documentation
```

## 6. How to Run

### Backend (Python 3.10+)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py                                       # serves http://127.0.0.1:8000
```

First launch downloads the real Telco CSV (or generates the `.xlsx` mirror when
offline), trains the ensemble and warms the service.

### Frontend

```bash
npm install
npm run dev        # local development
npm run build      # production bundle in dist/
```

The dashboard probes `http://127.0.0.1:8000` on load. Connected → **Python API
mode** (real Telco dataset). Not connected → **browser engine mode** (in-browser
GBDT on the synthetic mirror) with a visible status pill; click the pill or
"Retrain model" at any time to re-probe or refit. To point the frontend at a
remote backend, set `localStorage.setItem("churnlens:apiBase", "http://host:8000")`.

## 7. REST API

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/api/health` | liveness probe |
| GET | `/api/status` | engine, dataset provenance, framework comparison |
| GET | `/api/metrics` | curves, importance, EDA buckets, held-out test scores |
| GET | `/api/customers?search&band&contract&internet&sort&page&size` | scored page with exact contributions |
| POST | `/api/simulate` | score one what-if profile with decomposition |
| POST | `/api/retrain` | refit all engines with a new seed |

## 8. Edge Cases Handled

- Blank `TotalCharges` for tenure-0 customers (real data: 11 rows) → filled at 0.
- `"No internet service"` / `"No"` sentinels → canonical `None` service flags.
- Missing ticket column on real data → documented support-gap proxy.
- Class imbalance (~1:3) → PR-AUC as headline metric, threshold slider for the
  precision/recall trade-off.
- Corrupt downloads / unreachable mirrors → validation then `.xlsx` fallback.
- Stale network responses in the UI → request guards, debouncing, timeouts.
- Corrupt `localStorage` → safe defaults for tab, threshold and scenarios.

## 9. Academic Notes

- Boosting frameworks are compared on an identical stratified split; the
  scikit-learn engine is the scoring engine because its tree internals allow the
  exact additive (SHAP-style) decomposition used for explanations.
- Tree-path telescoping guarantees `base + Σ contributions = logit(p)` for every
  customer, in both Python and TypeScript engines — verifiable in any waterfall.
- All comments in the codebase are English section dividers.
