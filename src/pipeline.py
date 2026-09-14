"""Command-line pipeline that produces all data and analysis artefacts."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from src.data_loading import load_raw_data
from src.data_cleaning import clean_data
from src.feature_selection import select_features
from src.preprocessing import preprocess_features, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.eda import create_eda_outputs

def run_pipeline(root: Path = PROJECT_ROOT) -> dict:
    raw = load_raw_data(root / "data/raw")
    missing_before = raw.isna().sum()
    cleaned, cleaning_log = clean_data(raw)
    selected, selection_log = select_features(cleaned)
    features, transformer = preprocess_features(selected)
    processed = pd.concat([features.reset_index(drop=True), selected[["income"]].reset_index(drop=True)], axis=1)
    out_data = root / "data/processed"; tables = root / "outputs/tables"; figures = root / "outputs/figures"
    out_data.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
    processed.to_csv(out_data / "adult_census_income_processed.csv", index=False)
    cleaned.to_csv(out_data / "adult_census_income_cleaned_before_encoding.csv", index=False)
    create_eda_outputs(raw, figures, tables)
    numeric_before = selected[NUMERICAL_FEATURES].agg(["min", "max", "mean", "std"]).T
    numeric_after = processed[NUMERICAL_FEATURES].agg(["min", "max", "mean", "std"]).T
    scaling = numeric_before.add_suffix("_before").join(numeric_after.add_suffix("_after"))
    scaling.to_csv(tables / "scaling_before_after.csv")
    stages = pd.DataFrame([
        ["Raw combined UCI files", raw.shape[0], raw.shape[1], int(raw.isna().sum().sum()), "Original data; '?' parsed as missing"],
        ["After exact-duplicate removal", cleaned.shape[0], cleaned.shape[1], int(cleaned.isna().sum().sum()), "Removed exact repeated rows; removed fnlwgt"],
        ["After feature selection", selected.shape[0], selected.shape[1], int(selected.isna().sum().sum()), "Removed education_num and native_country"],
        ["Processed output", processed.shape[0], processed.shape[1], int(processed.isna().sum().sum()), "Median/mode imputation, one-hot encoding, standardization"],
    ], columns=["stage", "rows", "columns", "missing_cells", "description"])
    stages.to_csv(tables / "pipeline_before_after.csv", index=False)
    metadata = {"raw_shape": list(raw.shape), "cleaned_shape": list(cleaned.shape), "selected_shape": list(selected.shape), "processed_shape": list(processed.shape), "raw_missing_cells": int(raw.isna().sum().sum()), "processed_missing_cells": int(processed.isna().sum().sum()), "missing_by_column": {k:int(v) for k,v in missing_before[missing_before.gt(0)].items()}, "cleaning": cleaning_log, "feature_selection": selection_log, "numerical_features": NUMERICAL_FEATURES, "categorical_features": CATEGORICAL_FEATURES, "encoded_feature_count": int(features.shape[1])}
    (tables / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata

if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result, indent=2))
