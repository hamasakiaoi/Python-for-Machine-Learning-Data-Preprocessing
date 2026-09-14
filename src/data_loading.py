"""Load the original UCI Adult Census Income files without altering them."""
from pathlib import Path
import pandas as pd

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
           "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
           "hours_per_week", "native_country", "income"]

def load_raw_data(raw_dir: Path) -> pd.DataFrame:
    """Return the official UCI training and test partitions as one labelled DataFrame."""
    train = pd.read_csv(raw_dir / "adult.data", names=COLUMNS, skipinitialspace=True,
                        na_values="?", keep_default_na=True)
    test = pd.read_csv(raw_dir / "adult.test", names=COLUMNS, skiprows=1, skipinitialspace=True,
                       na_values="?", keep_default_na=True)
    data = pd.concat([train, test], ignore_index=True)
    data["income"] = data["income"].str.replace(".", "", regex=False)
    return data
