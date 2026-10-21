import numpy as np

from credit_risk.evaluate import choose_threshold, classification_metrics


def test_threshold_optimizes_explicit_costs():
    threshold = choose_threshold([0, 0, 1, 1], [0.1, 0.4, 0.45, 0.9], 5, 1)
    assert threshold <= 0.45


def test_metrics_has_required_operational_fields():
    result = classification_metrics(np.array([0, 1, 1]), np.array([0.1, 0.8, 0.6]))
    assert {"roc_auc", "pr_auc", "brier_score", "confusion_matrix"} <= result.keys()
