"""Missing-value treatment, nominal encoding, and numerical standardization."""
from typing import Any
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERICAL_FEATURES = ["age", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_FEATURES = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex"]

def build_preprocessor() -> ColumnTransformer:
    """Build deterministic transformations suitable for numerical and nominal inputs."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("numeric", numeric, NUMERICAL_FEATURES),
                              ("categorical", categorical, CATEGORICAL_FEATURES)],
                             verbose_feature_names_out=False)

def preprocess_features(data: pd.DataFrame) -> tuple[pd.DataFrame, Any]:
    """Return encoded/scaled features and the fitted transformer."""
    transformer = build_preprocessor()
    transformed = transformer.fit_transform(data)
    features = pd.DataFrame(transformed, columns=transformer.get_feature_names_out(), index=data.index)
    return features, transformer
