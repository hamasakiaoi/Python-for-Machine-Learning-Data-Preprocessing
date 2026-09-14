"""Data-quality checks and conservative cleaning decisions."""
import pandas as pd

def clean_data(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove exact duplicate rows and drop the census sampling-weight field.

    ``fnlwgt`` is an observation weight used to represent population estimates, not a
    person-level predictive characteristic; retaining it can make a downstream model
    reflect sampling design rather than individual attributes.
    """
    duplicates = int(data.duplicated().sum())
    cleaned = data.drop_duplicates().copy()
    cleaned = cleaned.drop(columns=["fnlwgt"])
    return cleaned, {"duplicates_removed": duplicates, "columns_removed": ["fnlwgt"]}
