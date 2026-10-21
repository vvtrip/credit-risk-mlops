from typing import Any

import numpy as np
import pandas as pd


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.nanquantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    reference_counts = np.histogram(reference, bins=edges)[0] / len(reference)
    current_counts = np.histogram(current, bins=edges)[0] / len(current)
    reference_counts = np.clip(reference_counts, 1e-6, None)
    current_counts = np.clip(current_counts, 1e-6, None)
    return float(np.sum((current_counts - reference_counts) * np.log(current_counts / reference_counts)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, psi_threshold: float = 0.2) -> dict[str, Any]:
    report = {}
    for column in reference.columns.intersection(current.columns):
        if pd.api.types.is_numeric_dtype(reference[column]):
            value = population_stability_index(reference[column].dropna(), current[column].dropna())
            report[column] = {"metric": "psi", "value": value, "drifted": value >= psi_threshold}
        else:
            ref = reference[column].value_counts(normalize=True)
            cur = current[column].value_counts(normalize=True)
            categories = ref.index.union(cur.index)
            p = ref.reindex(categories, fill_value=0).to_numpy() + 1e-6
            q = cur.reindex(categories, fill_value=0).to_numpy() + 1e-6
            midpoint = (p + q) / 2
            js = float(0.5 * np.sum(p * np.log(p / midpoint)) + 0.5 * np.sum(q * np.log(q / midpoint)))
            report[column] = {"metric": "js_divergence", "value": js, "drifted": js >= 0.1}
    return report
