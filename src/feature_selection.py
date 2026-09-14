"""Transparent, Week-1 appropriate feature-selection rules."""
import pandas as pd

DROP_FEATURES = {
    "education_num": "Exact numeric duplicate of education level; retain the readable categorical education field.",
    "native_country": "Missing in 1.8% of records and highly imbalanced (United-States dominates); not retained for a compact baseline feature set.",
}

def select_features(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Drop redundant and low-information candidate fields after documented review."""
    return data.drop(columns=list(DROP_FEATURES)), DROP_FEATURES.copy()
