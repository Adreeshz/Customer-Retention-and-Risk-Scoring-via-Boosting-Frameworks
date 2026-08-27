# ==============================================================================
# SECTION: SYNTHETIC WORKBOOK GENERATOR — writes data/raw/telco_churn_synthetic
# .xlsx, an offline fallback mirror of the Telco schema (synthetic data is
# always stored as .xlsx per the project data policy). It reproduces the
# published Telco shapes: U-shaped tenure, ~26% churn and service correlations
# ==============================================================================
import numpy as np
import pandas as pd

from config import SYNTHETIC_XLSX_PATH

# ==============================================================================
# SECTION: FIELD SAMPLERS — each business field is drawn from a tenure-aware
# conditional distribution so boosting has realistic structure to recover
# ==============================================================================
def _sample_tenure(rng: np.random.RandomState) -> int:
    segment = rng.choice(["new", "mid", "loyal"], p=[0.34, 0.30, 0.36])
    if segment == "new":
        return int(rng.randint(0, 13))
    if segment == "mid":
        return int(rng.randint(13, 49))
    return int(rng.randint(49, 73))


def _sample_contract(rng: np.random.RandomState, tenure: int) -> str:
    if tenure <= 12:
        weights = [0.76, 0.15, 0.09]
    elif tenure <= 48:
        weights = [0.48, 0.26, 0.26]
    else:
        weights = [0.34, 0.26, 0.40]
    return str(rng.choice(["Month-to-month", "One year", "Two year"], p=weights))


def _sample_internet(rng: np.random.RandomState, contract: str) -> str:
    fiber_bias = 0.55 if contract == "Month-to-month" else 0.44
    return str(rng.choice(["None", "DSL", "Fiber optic"], p=[0.21, 1 - 0.21 - fiber_bias, fiber_bias]))


def _sample_payment(rng: np.random.RandomState, contract: str) -> str:
    echeck = 0.44 if contract == "Month-to-month" else 0.16
    return str(
        rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
            p=[echeck, 0.21, 0.20, max(0.05, 0.59 - echeck)],
        )
    )


def _sample_tickets(rng: np.random.RandomState, tenure: int, internet: str, tech_support: bool, online_security: bool) -> int:
    lam = 0.7
    if internet == "Fiber optic":
        lam += 0.75
    if not tech_support:
        lam += 0.7
    if not online_security:
        lam += 0.35
    if tenure <= 12:
        lam += 0.5
    raw = -np.log(max(1e-9, 1 - rng.random_sample())) * lam
    return int(min(8, max(0, round(raw))))


def _sample_charges(rng: np.random.RandomState, internet: str, tech_support: bool, online_security: bool, streaming_tv: bool) -> float:
    if internet == "None":
        base = rng.uniform(19, 30)
    elif internet == "DSL":
        base = rng.uniform(38, 62)
    else:
        base = rng.uniform(64, 100)
    if tech_support:
        base += 9.5
    if online_security:
        base += 6.5
    if streaming_tv and internet != "None":
        base += 8.5
    return float(min(122.0, max(18.5, base + rng.normal(0, 3.5))))


def _sample_churn(rng: np.random.RandomState, f: dict) -> bool:
    logit = -1.55
    if f["contract"] == "Month-to-month":
        logit += 1.35
    if f["contract"] == "Two year":
        logit -= 0.55
    if f["internet"] == "Fiber optic":
        logit += 0.5
    if f["internet"] == "None":
        logit -= 0.3
    logit += 0.85 * np.exp(-f["tenure"] / 9)
    logit -= 0.012 * f["tenure"]
    logit += 0.3 * min(f["tickets"], 5)
    if not f["tech_support"]:
        logit += 0.42
    if not f["online_security"]:
        logit += 0.3
    logit += 0.011 * (f["monthly"] - 60)
    logit += rng.normal(0, 0.5)
    return bool(rng.random_sample() < 1 / (1 + np.exp(-logit)))


# ==============================================================================
# SECTION: RECORD ASSEMBLY — builds one full Telco-shaped row per iteration,
# including the support-ticket dimension required by the problem statement
# ==============================================================================
def generate_synthetic_frame(rows: int = 2000, seed: int = 20260214) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    records = []

    for i in range(rows):
        tenure = _sample_tenure(rng)
        contract = _sample_contract(rng, tenure)
        internet = _sample_internet(rng, contract)
        tech_support = rng.random_sample() < (0.55 if internet == "DSL" else 0.22 if internet == "Fiber optic" else 0.40)
        online_security = rng.random_sample() < (0.52 if internet == "DSL" else 0.20 if internet == "Fiber optic" else 0.42)
        streaming_tv = rng.random_sample() < (0.30 if internet == "None" else 0.50)
        tickets = _sample_tickets(rng, tenure, internet, tech_support, online_security)
        monthly = _sample_charges(rng, internet, tech_support, online_security, streaming_tv)

        # ---- billing history with the tenure-0 edge case (no charges yet) ----
        if tenure <= 0:
            total = 0.0
        else:
            total = round(monthly * tenure * rng.uniform(0.92, 1.1) * rng.uniform(0.97, 1.03), 2)

        churned = _sample_churn(
            rng,
            {
                "tenure": tenure,
                "contract": contract,
                "internet": internet,
                "tech_support": tech_support,
                "online_security": online_security,
                "tickets": tickets,
                "monthly": monthly,
            },
        )

        def yes_no(flag: bool, offline: bool = False) -> str:
            if offline:
                return "No internet service"
            return "Yes" if flag else "No"

        offline = internet == "None"
        records.append(
            {
                "customerID": f"SYN-{10001 + i}",
                "gender": "Female" if rng.random_sample() < 0.5 else "Male",
                "SeniorCitizen": int(rng.random_sample() < 0.16),
                "Partner": "Yes" if rng.random_sample() < 0.48 else "No",
                "Dependents": "Yes" if rng.random_sample() < 0.30 else "No",
                "tenure": tenure,
                "PhoneService": "Yes" if rng.random_sample() < 0.90 else "No",
                "MultipleLines": "No phone service" if rng.random_sample() < 0.10 else ("Yes" if rng.random_sample() < 0.42 else "No"),
                "InternetService": internet,
                "OnlineSecurity": yes_no(online_security, offline),
                "OnlineBackup": yes_no(rng.random_sample() < 0.45, offline),
                "DeviceProtection": yes_no(rng.random_sample() < 0.45, offline),
                "TechSupport": yes_no(tech_support, offline),
                "StreamingTV": yes_no(streaming_tv, offline),
                "StreamingMovies": yes_no(rng.random_sample() < 0.50, offline),
                "Contract": contract,
                "PaperlessBilling": "Yes" if rng.random_sample() < 0.60 else "No",
                "PaymentMethod": _sample_payment(rng, contract),
                "MonthlyCharges": round(monthly, 2),
                "TotalCharges": total,
                "Churn": "Yes" if churned else "No",
                "TicketsLastQuarter": tickets,
            }
        )

    return pd.DataFrame.from_records(records)


# ==============================================================================
# SECTION: WORKBOOK WRITER — persists the synthetic cohort as .xlsx on demand
# ==============================================================================
def ensure_synthetic_workbook(path=None, rows: int = 2000) -> str:
    target = path or SYNTHETIC_XLSX_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        frame = generate_synthetic_frame(rows=rows)
        frame.to_excel(target, index=False, sheet_name="telco_synthetic")
        print(f"[data] synthetic fallback written to {target}")
    return str(target)


if __name__ == "__main__":
    ensure_synthetic_workbook()
