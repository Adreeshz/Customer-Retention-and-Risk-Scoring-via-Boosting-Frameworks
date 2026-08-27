# ==============================================================================
# SECTION: CHURN MODEL SERVICE — loads the dataset once, then fits a scikit-learn
# GradientBoosting classifier (the scoring engine) plus optional XGBoost and
# LightGBM comparisons. Every artifact the dashboard needs — curves, confusion
# inputs, importance, tenure EDA and exact per-row decompositions — is computed
# here so the frontend renders identical numbers in API mode and browser mode
# ==============================================================================
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

import config
from data_loader import load_dataset
from features import FEATURE_COLUMNS, build_feature_matrix, profile_to_feature_row
from shap_explain import explain_row

# tenure bucket edges mirror src/lib/analytics.ts tenureBuckets() exactly
TENURE_EDGES = [0, 9, 17, 25, 33, 41, 49, 57, 65, 73]


def _r5(value) -> float:
    return round(float(value), 5)


def _band(probability: float) -> str:
    if probability < config.BAND_CUTS["low_upper"]:
        return "low"
    if probability < config.BAND_CUTS["watch_upper"]:
        return "watch"
    return "high"


# ==============================================================================
# SECTION: CURVE HELPERS — ROC / PR points downsampled for compact payloads
# ==============================================================================
def _thin(xs, ys, cap: int = 180) -> list[dict]:
    points = list(zip(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)))
    if len(points) > cap:
        stride = max(1, len(points) // cap)
        points = points[::stride] + [points[-1]]
    return [{"x": _r5(x), "y": _r5(y)} for x, y in points]


def _optional_framework(name: str, builder, x_test, y_test, seed: int) -> dict | None:
    # each framework is compared on identical hyperparameters when installed
    try:
        candidate = builder(seed)
        prob = candidate.predict_proba(x_test)[:, 1]
        # both arrays share length n_thresholds + 1, so reversing aligns them for auc()
        precision, recall, _ = precision_recall_curve(y_test, prob)
        return {
            "name": name,
            "rocAuc": _r5(roc_auc_score(y_test, prob)),
            "prAuc": _r5(auc(recall[::-1], precision[::-1])),
            "f1": _r5(f1_score(y_test, (prob >= 0.5).astype(int))),
            "active": False,
        }
    except Exception as exc:  # noqa: BLE001 — optional frameworks must never break the service
        print(f"[model] {name} comparison skipped ({exc})")
        return None


# ==============================================================================
# SECTION: MODEL OBJECT — dataset, matrices, fitted engines and all evaluation
# artifacts exposed through small, frontend-shaped payload builders
# ==============================================================================
class ChurnModel:
    def __init__(self, seed: int):
        self.frame, self.source = load_dataset()
        self.matrix = build_feature_matrix(self.frame)
        self.labels = self.frame["Churn"].to_numpy(dtype=int)
        self._customer_ids = self.frame["customerID"].to_numpy()
        self.train(seed)

    # --------------------------------------------------------------------------
    # SECTION: TRAINING — stratified split, gradient boosting fit, scoring of
    # the full cohort and evaluation artifacts; optional frameworks compared
    # --------------------------------------------------------------------------
    def train(self, seed: int) -> None:
        x_train, x_test, y_train, y_test = train_test_split(
            self.matrix,
            self.labels,
            test_size=config.TEST_SIZE,
            stratify=self.labels,
            random_state=seed + 7919,
        )

        params = dict(config.HYPERPARAMS)
        params["random_state"] = seed

        started = time.perf_counter()
        engine = GradientBoostingClassifier(**params).fit(x_train, y_train)
        trained_ms = int((time.perf_counter() - started) * 1000)

        probabilities = engine.predict_proba(self.matrix)[:, 1]
        test_prob = engine.predict_proba(x_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, test_prob)
        # both arrays share length n_thresholds + 1, so reversing aligns them for auc()
        precision, recall, _ = precision_recall_curve(y_test, test_prob)

        importance_raw = engine.feature_importances_
        importance_total = float(importance_raw.sum()) or 1.0
        importance = [_r5(v / importance_total) for v in importance_raw]

        frameworks = [
            {
                "name": "scikit-learn GradientBoosting",
                "rocAuc": _r5(roc_auc_score(y_test, test_prob)),
                "prAuc": _r5(auc(recall[::-1], precision[::-1])),
                "f1": _r5(f1_score(y_test, (test_prob >= 0.5).astype(int))),
                "active": True,
            }
        ]
        xgb = _optional_framework(
            "XGBoost",
            lambda s: __import__("xgboost").XGBClassifier(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                subsample=params["subsample"],
                reg_lambda=1.2,
                min_child_weight=params["min_samples_leaf"],
                tree_method="hist",
                eval_metric="logloss",
                random_state=s,
                n_jobs=4,
            ).fit(x_train, y_train),
            x_test,
            y_test,
            seed,
        )
        if xgb:
            frameworks.append(xgb)
        lgb = _optional_framework(
            "LightGBM",
            lambda s: __import__("lightgbm").LGBMClassifier(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                subsample=params["subsample"],
                subsample_freq=1,
                reg_lambda=1.2,
                min_child_samples=params["min_samples_leaf"],
                random_state=s,
                verbosity=-1,
            ).fit(x_train, y_train),
            x_test,
            y_test,
            seed,
        )
        if lgb:
            frameworks.append(lgb)

        tenure_array = self.frame["tenure"].to_numpy()
        bucket_index = np.minimum(len(TENURE_EDGES) - 2, tenure_array // 8)
        tenure_curve = []
        for b in range(len(TENURE_EDGES) - 1):
            mask = bucket_index == b
            stayed = int(mask.sum() - self.labels[mask].sum())
            churned = int(self.labels[mask].sum())
            total = stayed + churned
            tenure_curve.append(
                {
                    "label": f"{TENURE_EDGES[b]}–{TENURE_EDGES[b + 1] - 1}",
                    "stayed": stayed,
                    "churned": churned,
                    "rate": _r5(churned / total) if total else 0.0,
                }
            )

        # one atomic swap keeps concurrent readers consistent during retraining
        self.__dict__.update(
            {
                "engine": engine,
                "seed": seed,
                "trained_ms": trained_ms,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "probabilities": probabilities,
                "roc_auc": _r5(roc_auc_score(y_test, test_prob)),
                "pr_auc": _r5(auc(recall[::-1], precision[::-1])),
                "curves": {"roc": _thin(fpr, tpr), "pr": _thin(recall[::-1], precision[::-1])},
                "test_scores": {"p": [_r5(v) for v in test_prob], "y": [int(v) for v in y_test]},
                "importance": importance,
                "tenure_curve": tenure_curve,
                "train_rows": int(len(x_train)),
                "test_rows": int(len(x_test)),
                "avg_prob": _r5(float(probabilities.mean())),
                "churn_rate": _r5(float(self.labels.mean())),
                "frameworks": frameworks,
            }
        )
        print(f"[model] seed #{seed} fitted in {trained_ms} ms — ROC-AUC {self.roc_auc:.3f}, PR-AUC {self.pr_auc:.3f}")

    # --------------------------------------------------------------------------
    # SECTION: EXPLANATION — exact additive decomposition for one dataset row
    # --------------------------------------------------------------------------
    def explain_index(self, index: int) -> dict:
        return explain_row(self.engine, self.matrix[index])

    # --------------------------------------------------------------------------
    # SECTION: WHAT-IF SCORING — encodes a frontend profile and scores it with
    # the same pipeline used for dataset rows
    # --------------------------------------------------------------------------
    def simulate(self, profile: dict) -> dict:
        row = profile_to_feature_row(profile)
        return explain_row(self.engine, row)

    # --------------------------------------------------------------------------
    # SECTION: CUSTOMER PAYLOAD — one canonical row shaped for the React UI
    # --------------------------------------------------------------------------
    def customer_payload(self, index: int) -> dict:
        row = self.frame.iloc[index]
        tenure = int(row["tenure"])
        monthly = float(row["MonthlyCharges"])
        total = float(row["TotalCharges"])
        avg = total / (tenure + 1.0)
        return {
            "id": str(row["customerID"]),
            "gender": str(row["gender"]),
            "senior": int(row["SeniorCitizen"]),
            "dependents": bool(row["Dependents"]),
            "partner": bool(row["Partner"]),
            "tenure": tenure,
            "contract": str(row["Contract"]),
            "paperless": bool(row["PaperlessBilling"]),
            "payment": str(row["PaymentMethod"]),
            "internet": str(row["InternetService"]),
            "techSupport": bool(row["TechSupport"]),
            "onlineSecurity": bool(row["OnlineSecurity"]),
            "streamingTV": bool(row["StreamingTV"]),
            "tickets": int(row["TicketsLastQuarter"]),
            "monthlyCharges": round(monthly, 2),
            "totalCharges": round(total, 2),
            "avgMonthly": round(avg, 2),
            "risingBill": bool(monthly > avg * 1.08),
            "churned": bool(row["Churn"]),
        }

    # --------------------------------------------------------------------------
    # SECTION: FILTERED QUERY — search, band, contract and internet filters with
    # deterministic sorting and pagination; contributions computed per page
    # --------------------------------------------------------------------------
    def query(self, search: str, band: str, contract: str, internet: str, sort: str, page: int, size: int) -> dict:
        frame = self.frame
        probs = self.probabilities
        mask = np.ones(len(frame), dtype=bool)

        needle = search.strip().lower()
        if needle:
            id_match = frame["customerID"].astype(str).str.lower().str.contains(needle, regex=False).to_numpy()
            pay_match = frame["PaymentMethod"].astype(str).str.lower().str.contains(needle, regex=False).to_numpy()
            mask &= id_match | pay_match
        if band in ("low", "watch", "high"):
            mask &= np.array([_band(p) == band for p in probs])
        if contract in ("Month-to-month", "One year", "Two year"):
            mask &= (frame["Contract"].to_numpy() == contract)
        if internet in ("None", "DSL", "Fiber optic"):
            mask &= (frame["InternetService"].to_numpy() == internet)

        matched = np.flatnonzero(mask)
        ids = self._customer_ids[matched]

        sorters = {
            "risk-desc": (-probs[matched], ids),
            "risk-asc": (probs[matched], ids),
            "tenure-asc": (frame["tenure"].to_numpy()[matched], ids),
            "charges-desc": (-frame["MonthlyCharges"].to_numpy()[matched], ids),
            "tickets-desc": (-frame["TicketsLastQuarter"].to_numpy()[matched], ids),
        }
        primary, tiebreak = sorters.get(sort, sorters["risk-desc"])
        order = np.lexsort((tiebreak, primary))
        ordered = matched[order]

        total = int(len(ordered))
        pages = max(1, (total + size - 1) // size)
        safe_page = min(page, pages - 1)
        page_indices = ordered[safe_page * size : (safe_page + 1) * size]

        rows = []
        for index in page_indices:
            explanation = self.explain_index(int(index))
            rows.append(
                {
                    "id": str(self._customer_ids[index]),
                    "probability": _r5(probs[index]),
                    "base": explanation["base"],
                    "contribs": explanation["contribs"],
                    "customer": self.customer_payload(int(index)),
                }
            )

        return {"total": total, "page": int(safe_page), "pages": pages, "rows": rows}