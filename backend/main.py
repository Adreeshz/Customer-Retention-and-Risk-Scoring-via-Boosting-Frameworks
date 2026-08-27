# ==============================================================================
# SECTION: FASTAPI SERVICE — REST surface consumed by the React dashboard:
#   GET  /api/health      liveness probe used by the frontend on boot
#   GET  /api/status      engine, dataset provenance and framework comparison
#   GET  /api/metrics     evaluation artifacts (curves, importance, EDA, scores)
#   GET  /api/customers   filtered, sorted, paginated scored customers
#   POST /api/simulate    score one what-if profile with exact decomposition
#   POST /api/retrain     refit every engine with a new random seed
# ==============================================================================
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
from features import FEATURE_COLUMNS
from model import ChurnModel

# ==============================================================================
# SECTION: APPLICATION STATE — one model instance, warmed during startup so the
# first dashboard request is served instantly
# ==============================================================================
_state: dict = {"model": None}


def get_model() -> ChurnModel:
    if _state["model"] is None:
        _state["model"] = ChurnModel(config.RANDOM_SEED)
    return _state["model"]


@asynccontextmanager
def lifespan(_: FastAPI):
    get_model()
    yield


app = FastAPI(title="ChurnLens API", version="1.0.0", lifespan=lifespan)

# ==============================================================================
# SECTION: CORS — the dashboard is served from a different origin in development
# ==============================================================================
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

# ==============================================================================
# SECTION: REQUEST SCHEMAS — validated payloads for the write endpoints
# ==============================================================================
class ProfilePayload(BaseModel):
    tenure: int = Field(default=8, ge=0, le=72)
    monthly_charges: float = Field(default=78.0, ge=18, le=130)
    contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    internet_service: Literal["None", "DSL", "Fiber optic"] = "Fiber optic"
    payment_method: Literal["Electronic check", "Mailed check", "Bank transfer", "Credit card"] = "Electronic check"
    tech_support: bool = False
    online_security: bool = False
    streaming_tv: bool = True
    paperless_billing: bool = True
    rising_bill: bool = True
    tickets: int = Field(default=3, ge=0, le=12)
    senior: bool = False
    dependents: bool = False


class RetrainPayload(BaseModel):
    seed: int = Field(default=config.RANDOM_SEED, ge=1, le=10**6)


# ==============================================================================
# SECTION: READ ENDPOINTS — health, status, metrics and the customer explorer
# ==============================================================================
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "churnlens-backend", "version": "1.0.0"}


@app.get("/api/status")
def status() -> dict:
    model = get_model()
    return {
        "engine": "scikit-learn GradientBoosting",
        "datasetSource": model.source,
        "datasetRows": int(len(model.frame)),
        "featureCount": len(FEATURE_COLUMNS),
        "featureNames": list(FEATURE_COLUMNS),
        "seed": int(model.seed),
        "trainedAt": model.trained_at,
        "trees": int(config.HYPERPARAMS["n_estimators"]),
        "frameworks": model.frameworks,
        "hyperparams": {
            "learningRate": config.HYPERPARAMS["learning_rate"],
            "maxDepth": config.HYPERPARAMS["max_depth"],
            "subsample": config.HYPERPARAMS["subsample"],
            "minSamplesLeaf": config.HYPERPARAMS["min_samples_leaf"],
        },
    }


@app.get("/api/metrics")
def metrics() -> dict:
    model = get_model()
    churned = int(model.labels.sum())
    return {
        "rocAuc": model.roc_auc,
        "prAuc": model.pr_auc,
        "curves": model.curves,
        "importance": model.importance,
        "tenureCurve": model.tenure_curve,
        "classBalance": {"retained": int(len(model.labels) - churned), "churned": churned},
        "avgProb": model.avg_prob,
        "churnRate": model.churn_rate,
        "testScores": model.test_scores,
        "trainRows": model.train_rows,
        "testRows": model.test_rows,
        "trainedMs": model.trained_ms,
        "seed": int(model.seed),
        "trees": int(config.HYPERPARAMS["n_estimators"]),
    }


@app.get("/api/customers")
def customers(
    search: str = "",
    band: str = "all",
    contract: str = "all",
    internet: str = "all",
    sort: str = "risk-desc",
    page: int = Query(default=0, ge=0),
    size: int = Query(default=config.DEFAULT_PAGE_SIZE, ge=1, le=config.MAX_PAGE_SIZE),
) -> dict:
    model = get_model()
    return model.query(search=search, band=band, contract=contract, internet=internet, sort=sort, page=page, size=size)


# ==============================================================================
# SECTION: WRITE ENDPOINTS — what-if scoring and retraining
# ==============================================================================
@app.post("/api/simulate")
def simulate(payload: ProfilePayload) -> dict:
    model = get_model()
    return model.simulate(payload.model_dump())


@app.post("/api/retrain")
def retrain(payload: RetrainPayload) -> dict:
    model = get_model()
    model.train(payload.seed)
    return {"ok": True, "seed": int(model.seed), "rocAuc": model.roc_auc, "trainedAt": model.trained_at}
