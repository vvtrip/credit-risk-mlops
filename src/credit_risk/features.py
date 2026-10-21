import pandas as pd


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["credit_amount_per_month"] = result["credit_amount"] / result["duration_months"].clip(lower=1)
    result["age_band"] = pd.cut(result["age"], bins=[0, 25, 35, 50, 100], labels=["young", "adult", "midlife", "senior"])
    result["credit_history_duration_ratio"] = result["duration_months"] / result["existing_credits"].clip(lower=1)
    return result
