# ==============================================================================
# SECTION: DATASET LOADER — acquisition strategy for the churn project:
#   1. reuse a previously downloaded Telco CSV when it exists on disk
#   2. download the real IBM Telco Customer Churn CSV from public mirrors
#   3. fall back to the synthetic .xlsx mirror when the network is unavailable
# then unify both sources into one canonical, model-ready frame
# ==============================================================================
import pandas as pd
import requests

from config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    DATASET_URLS,
    DOWNLOAD_TIMEOUT_SECONDS,
    EXPECTED_COLUMN,
    EXPECTED_MIN_ROWS,
    PROCESSED_XLSX_PATH,
    REAL_CSV_PATH,
    SYNTHETIC_XLSX_PATH,
)
from synthetic_data_generator import ensure_synthetic_workbook

# ==============================================================================
# SECTION: REAL DATASET ACQUISITION — validates shape before trusting a file
# ==============================================================================
def _frame_is_valid(frame: pd.DataFrame) -> bool:
    return EXPECTED_COLUMN in frame.columns and len(frame) >= EXPECTED_MIN_ROWS


def _read_real_csv(path) -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(path)
        return frame if _frame_is_valid(frame) else None
    except Exception as exc:  # noqa: BLE001 — a corrupt cache must never crash the service
        print(f"[data] ignoring unreadable cached CSV ({exc})")
        return None


def download_real_dataset() -> str | None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    cached = _read_real_csv(REAL_CSV_PATH)
    if cached is not None:
        return str(REAL_CSV_PATH)

    for url in DATASET_URLS:
        try:
            print(f"[data] downloading Telco dataset from {url}")
            response = requests.get(url, timeout=DOWNLOAD_TIMEOUT_SECONDS)
            response.raise_for_status()
            REAL_CSV_PATH.write_bytes(response.content)
            if _read_real_csv(REAL_CSV_PATH) is not None:
                print(f"[data] saved {len(response.content)} bytes to {REAL_CSV_PATH}")
                return str(REAL_CSV_PATH)
            print("[data] downloaded file failed validation, trying next mirror")
            REAL_CSV_PATH.unlink(missing_ok=True)
        except requests.RequestException as exc:
            print(f"[data] mirror unavailable ({exc})")
    return None


# ==============================================================================
# SECTION: SOURCE RESOLUTION — picks the real dataset, else the .xlsx mirror
# ==============================================================================
def load_raw_frame() -> tuple[pd.DataFrame, str]:
    real_path = download_real_dataset()
    if real_path:
        return pd.read_csv(real_path), "IBM Telco Customer Churn (CSV, real dataset)"

    workbook = ensure_synthetic_workbook(SYNTHETIC_XLSX_PATH)
    return pd.read_excel(workbook), "Synthetic Telco mirror (XLSX fallback)"


# ==============================================================================
# SECTION: SCHEMA UNIFICATION — maps raw Telco values onto canonical columns and
# resolves edge cases: blank TotalCharges for brand-new customers, the
# "No internet service" sentinel, and the missing support-ticket column on the
# real dataset (replaced by a documented support-gap proxy)
# ==============================================================================
def _flag(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().map({"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0}).fillna(0).astype(int)


def _support_ticket_proxy(frame: pd.DataFrame) -> pd.Series:
    exposed = (frame["InternetService"] != "None").astype(int)
    gap = (1 - frame["TechSupport"]) + (1 - frame["OnlineSecurity"])
    early = (frame["tenure"] <= 12).astype(int)
    return (exposed * (2 * gap + early)).clip(upper=5).astype(int)


def prepare_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()

    if "customerID" not in frame.columns:
        frame["customerID"] = [f"ROW-{i}" for i in range(len(frame))]

    frame["gender"] = frame["gender"].astype(str).where(frame["gender"].astype(str).isin(["Female", "Male"]), "Female")
    frame["SeniorCitizen"] = pd.to_numeric(frame["SeniorCitizen"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    frame["Partner"] = _flag(frame["Partner"])
    frame["Dependents"] = _flag(frame["Dependents"])
    frame["tenure"] = pd.to_numeric(frame["tenure"], errors="coerce").fillna(0).astype(int).clip(0, 72)
    frame["Contract"] = frame["Contract"].astype(str).where(
        frame["Contract"].astype(str).isin(["Month-to-month", "One year", "Two year"]), "Month-to-month"
    )
    frame["PaperlessBilling"] = _flag(frame["PaperlessBilling"])
    frame["PaymentMethod"] = frame["PaymentMethod"].astype(str).where(
        frame["PaymentMethod"].astype(str).isin(["Electronic check", "Mailed check", "Bank transfer", "Credit card"]),
        "Electronic check",
    )

    internet = frame["InternetService"].astype(str).str.strip()
    frame["InternetService"] = internet.map({"No": "None", "No internet service": "None", "None": "None", "DSL": "DSL", "Fiber optic": "Fiber optic"}).fillna("None")

    frame["TechSupport"] = _flag(frame["TechSupport"])
    frame["OnlineSecurity"] = _flag(frame["OnlineSecurity"])
    frame["StreamingTV"] = _flag(frame["StreamingTV"])

    frame["MonthlyCharges"] = pd.to_numeric(frame["MonthlyCharges"], errors="coerce").fillna(frame["MonthlyCharges"].median()).clip(18, 130)

    # ---- edge case: 11 real records carry blank TotalCharges at tenure 0 ----
    total = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    total = total.where(~((total.isna()) & (frame["tenure"] == 0)), 0.0)
    frame["TotalCharges"] = total.fillna(total.median()).clip(lower=0)

    frame["Churn"] = _flag(frame["Churn"])

    if "TicketsLastQuarter" in frame.columns:
        frame["TicketsLastQuarter"] = pd.to_numeric(frame["TicketsLastQuarter"], errors="coerce").fillna(0).astype(int).clip(0, 12)
    else:
        frame["TicketsLastQuarter"] = _support_ticket_proxy(frame)

    canonical = frame[
        [
            "customerID",
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "InternetService",
            "TechSupport",
            "OnlineSecurity",
            "StreamingTV",
            "MonthlyCharges",
            "TotalCharges",
            "TicketsLastQuarter",
            "Churn",
        ]
    ].reset_index(drop=True)

    return canonical


# ==============================================================================
# SECTION: PROCESSED ARTIFACT CACHE — persists the canonical frame as .xlsx so
# the cleaning step is auditable and reusable across retraining cycles
# ==============================================================================
def save_processed(frame: pd.DataFrame) -> str:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_excel(PROCESSED_XLSX_PATH, index=False, sheet_name="telco_processed")
    return str(PROCESSED_XLSX_PATH)


def load_dataset() -> tuple[pd.DataFrame, str]:
    raw, source = load_raw_frame()
    prepared = prepare_frame(raw)
    save_processed(prepared)
    return prepared, source


if __name__ == "__main__":
    data, origin = load_dataset()
    print(f"[data] {origin}: {len(data)} rows, churn rate {data['Churn'].mean():.2%}")
