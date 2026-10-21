from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def choose_threshold(y_true, probabilities, false_negative_cost: float, false_positive_cost: float) -> float:
    best = (float("inf"), 0.5)
    for threshold in np.linspace(0.05, 0.95, 91):
        predicted = probabilities >= threshold
        matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
        _, fp, fn, _ = matrix.ravel()
        cost = false_negative_cost * fn + false_positive_cost * fp
        if cost < best[0]:
            best = (float(cost), float(threshold))
    return best[1]


def classification_metrics(y_true, probabilities, threshold: float = 0.5) -> dict[str, Any]:
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
        "threshold": threshold,
    }


def cross_validated_metrics(model, x, y, folds: int, random_state: int) -> dict[str, Any]:
    splitter = StratifiedKFold(folds, shuffle=True, random_state=random_state)
    probabilities = cross_val_predict(model, x, y, cv=splitter, method="predict_proba", n_jobs=1)[:, 1]
    return classification_metrics(y, probabilities)


def permutation_importances(model, x, y, random_state: int) -> list[dict[str, float | str]]:
    result = permutation_importance(model, x, y, scoring="roc_auc", n_repeats=5,
                                    random_state=random_state, n_jobs=1)
    return [{"feature": str(name), "importance_mean": float(value)}
            for name, value in sorted(zip(x.columns, result.importances_mean),
                                      key=lambda item: item[1], reverse=True)[:15]]


def slice_metrics(model, x: pd.DataFrame, y, probability_threshold: float) -> dict[str, Any]:
    output = {}
    for column in ("age_band", "personal_status_sex"):
        if column in x:
            output[column] = {}
            for value, group in x.groupby(column, observed=True):
                indices = group.index
                output[column][str(value)] = {
                    "count": len(indices),
                    "positive_rate": float(y.loc[indices].mean()),
                    "recall": float(recall_score(y.loc[indices], (model.predict_proba(group)[:, 1] >= probability_threshold).astype(int),
                                                  zero_division=0)),
                }
    return output
