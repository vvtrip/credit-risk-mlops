import pandas as pd

from credit_risk.features import add_features


def test_feature_engineering_adds_stable_domain_features():
    source = pd.DataFrame({"credit_amount": [1200], "duration_months": [12],
                           "age": [40], "existing_credits": [2]})
    result = add_features(source)
    assert result.loc[0, "credit_amount_per_month"] == 100
    assert result.loc[0, "age_band"] == "midlife"
    assert result.loc[0, "credit_history_duration_ratio"] == 6
