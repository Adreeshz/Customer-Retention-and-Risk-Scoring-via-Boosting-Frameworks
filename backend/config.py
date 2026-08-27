# ==============================================================================
# SECTION: PATH CONFIGURATION — every filesystem location used by the pipeline,
# resolved relative to this file so the backend runs from any working directory
# ==============================================================================
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
DATA_RAW_DIR = PROJECT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

# ==============================================================================
# SECTION: DATASET LOCATIONS — the real Telco CSV download target plus the
# synthetic fallback workbook (synthetic artifacts are always stored as .xlsx)
# ==============================================================================
REAL_CSV_PATH = DATA_RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
SYNTHETIC_XLSX_PATH = DATA_RAW_DIR / "telco_churn_synthetic.xlsx"
PROCESSED_XLSX_PATH = DATA_PROCESSED_DIR / "telco_churn_processed.xlsx"

# ==============================================================================
# SECTION: DOWNLOAD SOURCES — public mirrors of the IBM Telco Customer Churn
# file; each candidate is tried in order until one passes validation
# ==============================================================================
DATASET_URLS = [
    "https://raw.githubusercontent.com/dsrscientist/dataset1/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/aniruddhachoudhury/Churn-Models/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
]
EXPECTED_MIN_ROWS = 7000
EXPECTED_COLUMN = "Churn"
DOWNLOAD_TIMEOUT_SECONDS = 25

# ==============================================================================
# SECTION: MODEL HYPERPARAMETERS — one shared vocabulary with the in-browser
# fallback engine so both tiers report identical feature names and band cuts
# ==============================================================================
RANDOM_SEED = 42
TEST_SIZE = 0.2

HYPERPARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.085,
    "max_depth": 4,
    "min_samples_leaf": 10,
    "subsample": 0.85,
    "random_state": RANDOM_SEED,
}

# ==============================================================================
# SECTION: RISK BAND CUT-OFFS — must match the frontend bandOf() thresholds
# ==============================================================================
BAND_CUTS = {"low_upper": 0.25, "watch_upper": 0.50}

# ==============================================================================
# SECTION: API SETTINGS — local FastAPI service coordinates and query limits
# ==============================================================================
API_HOST = "127.0.0.1"
API_PORT = 8000
MAX_PAGE_SIZE = 500
DEFAULT_PAGE_SIZE = 9
