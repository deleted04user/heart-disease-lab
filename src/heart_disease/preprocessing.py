"""Leakage-safe transformations fitted inside each model/CV pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL, NUMERIC


def build_preprocessor(*, scale: bool = True) -> ColumnTransformer:
    numerical = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        numerical.append(("scale", StandardScaler()))
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numerical), NUMERIC),
            ("categorical", categorical, CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
