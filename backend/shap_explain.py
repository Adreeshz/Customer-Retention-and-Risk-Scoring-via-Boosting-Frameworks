# ==============================================================================
# SECTION: ADDITIVE TREE-PATH EXPLAINER — decomposes one prediction of a fitted
# scikit-learn GradientBoostingClassifier into per-feature log-odds effects.
# Walking root-to-leaf and telescoping (leaf - parent) at every split yields
# contributions that sum exactly to the model logit, the same guarantee the
# frontend's in-browser engine provides (SHAP-style additive decomposition)
# ==============================================================================
import math

import numpy as np


def _sigmoid(logit: float) -> float:
    # numerically stable logistic function for both sign regimes
    if logit >= 0:
        z = math.exp(-logit)
        return 1.0 / (1.0 + z)
    z = math.exp(logit)
    return z / (1.0 + z)


def _base_logit(model) -> float:
    # the boosting init is the log-odds of the training class prior
    try:
        prior = float(model.init_.class_prior_[1])
    except (AttributeError, IndexError, TypeError):
        prior = 0.5
    prior = min(0.99, max(0.01, prior))
    return math.log(prior / (1.0 - prior))


# ==============================================================================
# SECTION: SINGLE-PREDICTION DECOMPOSITION — returns base, per-feature effects,
# the reconstructed logit and the authoritative probability
# ==============================================================================
def explain_row(model, row: np.ndarray) -> dict:
    base = _base_logit(model)
    learning_rate = float(model.learning_rate)
    feature_count = model.n_features_in_
    contribs = np.zeros(feature_count, dtype=np.float64)

    # every stage of a binary classifier holds exactly one regression tree
    for stage in model.estimators_:
        tree = stage[0].tree_
        feature = tree.feature
        threshold = tree.threshold
        left = tree.children_left
        right = tree.children_right
        value = tree.value[:, 0, 0]

        node = 0
        base += learning_rate * float(value[0])
        while feature[node] >= 0:
            goes_left = row[feature[node]] <= threshold[node]
            child = left[node] if goes_left else right[node]
            contribs[feature[node]] += learning_rate * (float(value[child]) - float(value[node]))
            node = child

    logit = base + float(contribs.sum())
    probability = _sigmoid(logit)

    return {
        "base": round(base, 5),
        "contribs": [round(float(c), 5) for c in contribs],
        "logit": round(logit, 5),
        "probability": round(probability, 5),
    }


def explain_batch(model, matrix: np.ndarray) -> list[dict]:
    # page-sized batches keep REST responses fast while staying exact
    return [explain_row(model, matrix[i]) for i in range(matrix.shape[0])]
