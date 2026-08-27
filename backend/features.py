# ==============================================================================
# SECTION: FEATURE ENGINEERING — builds the numeric model matrix from the
# canonical frame. The 17 column names are the shared vocabulary of the whole
# project: the frontend renders them verbatim in importance bars, waterfalls
# and playbooks, so any change here must be mirrored in src/lib/dataset.ts
# ==============================================================================
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
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
    "tickets_overloaded",
]

# ==============================================================================
# SECTION: DATASET MATRIX — vectorized encoding of every customer record, with
# the ratio metrics (average monthly spend, rising-bill flag) derived safely
# even for tenure-0 customers who have no billing history yet
# ==============================================================================
def build_feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    tenure = frame["tenure"].to_numpy(dtype=np.float64)
    monthly = frame["MonthlyCharges"].to_numpy(dtype=np.float64)
    total = frame["TotalCharges"].to_numpy(dtype=np.float64)
    tickets = frame["TicketsLastQuarter"].to_numpy(dtype=np.float64)
    internet = frame["InternetService"].to_numpy()
    contract = frame["Contract"].to_numpy()
    payment = frame["PaymentMethod"].to_numpy()

    avg_monthly = total / (tenure + 1.0)
    rising_bill = (monthly > avg_monthly * 1.08).astype(np.float64)

    matrix = np.column_stack(
        [
            tenure,
            monthly,
            avg_monthly,
            rising_bill,
            (contract == "Month-to-month").astype(np.float64),
            (contract == "One year").astype(np.float64),
            (internet == "Fiber optic").astype(np.float64),
            (internet == "DSL").astype(np.float64),
            frame["TechSupport"].to_numpy(dtype=np.float64),
            frame["OnlineSecurity"].to_numpy(dtype=np.float64),
            frame["StreamingTV"].to_numpy(dtype=np.float64),
            frame["PaperlessBilling"].to_numpy(dtype=np.float64),
            (payment == "Electronic check").astype(np.float64),
            frame["SeniorCitizen"].to_numpy(dtype=np.float64),
            frame["Dependents"].to_numpy(dtype=np.float64),
            tickets,
            (tickets >= 3).astype(np.float64),
        ]
    )
    return np.ascontiguousarray(matrix, dtype=np.float64)


# ==============================================================================
# SECTION: PROFILE ENCODING — encodes one what-if profile sent by the frontend.
# Billing derivation mirrors the simulator exactly: when a rising bill is
# declared, the historical average is modeled 10% below the current monthly
# ==============================================================================
def profile_to_feature_row(profile: dict) -> np.ndarray:
    tenure = float(min(72, max(0, profile.get("tenure", 0))))
    monthly = float(min(130, max(18, profile.get("monthly_charges", 50))))
    rising = bool(profile.get("rising_bill", False))
    tickets = float(min(12, max(0, profile.get("tickets", 0))))

    avg_monthly = monthly * (0.9 if rising else 1.0) if tenure > 0 else monthly

    contract = profile.get("contract", "Month-to-month")
    internet = profile.get("internet_service", "None")
    payment = profile.get("payment_method", "Electronic check")

    return np.array(
        [
            tenure,
            monthly,
            avg_monthly,
            1.0 if rising else 0.0,
            1.0 if contract == "Month-to-month" else 0.0,
            1.0 if contract == "One year" else 0.0,
            1.0 if internet == "Fiber optic" else 0.0,
            1.0 if internet == "DSL" else 0.0,
            1.0 if profile.get("tech_support", False) else 0.0,
            1.0 if profile.get("online_security", False) else 0.0,
            1.0 if profile.get("streaming_tv", False) else 0.0,
            1.0 if profile.get("paperless_billing", False) else 0.0,
            1.0 if payment == "Electronic check" else 0.0,
            1.0 if profile.get("senior", False) else 0.0,
            1.0 if profile.get("dependents", False) else 0.0,
            tickets,
            1.0 if tickets >= 3 else 0.0,
        ],
        dtype=np.float64,
    )
