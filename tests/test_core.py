"""Focused regression tests for the core Week 1 preprocessing contract."""
import unittest

import pandas as pd

from src.data_cleaning import clean_data
from src.preprocessing import NUMERICAL_FEATURES, preprocess_features


def sample_data() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25, 45, 45],
        "workclass": ["Private", None, None],
        "fnlwgt": [100, 200, 200],
        "education": ["Bachelors", "HS-grad", "HS-grad"],
        "education_num": [13, 9, 9],
        "marital_status": ["Never-married", "Married-civ-spouse", "Married-civ-spouse"],
        "occupation": ["Tech-support", None, None],
        "relationship": ["Not-in-family", "Husband", "Husband"],
        "race": ["White", "White", "White"],
        "sex": ["Female", "Male", "Male"],
        "capital_gain": [0, 5000, 5000],
        "capital_loss": [0, 0, 0],
        "hours_per_week": [40, 50, 50],
        "income": ["<=50K", ">50K", ">50K"],
    })


class CorePreprocessingTests(unittest.TestCase):
    def test_cleaning_removes_exact_duplicates_and_sampling_weight(self) -> None:
        cleaned, log = clean_data(sample_data())
        self.assertEqual(cleaned.shape[0], 2)
        self.assertNotIn("fnlwgt", cleaned.columns)
        self.assertEqual(log["duplicates_removed"], 1)

    def test_preprocessing_imputes_and_standardizes_features(self) -> None:
        cleaned, _ = clean_data(sample_data())
        features, _ = preprocess_features(cleaned)
        self.assertFalse(features.isna().any().any())
        self.assertEqual(features.shape[0], 2)
        for column in NUMERICAL_FEATURES:
            self.assertAlmostEqual(features[column].mean(), 0.0)
            expected_std = 0.0 if cleaned[column].nunique() == 1 else 1.0
            self.assertAlmostEqual(features[column].std(ddof=0), expected_std)


if __name__ == "__main__":
    unittest.main()
