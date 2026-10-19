"""Custom transformer definitions for the tabular network intrusion detection pipeline.

Provides zero-leakage, Scikit-learn compatible transformers for infinite value clipping,
missing value imputation, categorical encoding with unseen category protection,
and schema subset selection.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from ml.preprocessing.schema import (
    ALTERNATIVE_FEATURE_COLUMNS_NO_STTL,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PRIMARY_FEATURE_COLUMNS,
)


class InfClipper(BaseEstimator, TransformerMixin):
    """Replaces infinite values (inf, -inf) with maximum/minimum finite values observed in training data."""

    def __init__(self, clip_strategy: str = "max_finite") -> None:
        self.clip_strategy = clip_strategy
        self.max_finite_: Optional[np.ndarray] = None
        self.min_finite_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray | pd.DataFrame, y: Optional[np.ndarray] = None) -> InfClipper:
        """Compute the maximum finite values per column strictly on the training partition."""
        X_mat = np.asarray(X, dtype=np.float64)
        is_finite = np.isfinite(X_mat)

        # Replace non-finite with NaN temporarily to compute nanmin/nanmax
        temp = np.where(is_finite, X_mat, np.nan)
        self.max_finite_ = np.nanmax(temp, axis=0)
        self.min_finite_ = np.nanmin(temp, axis=0)

        # Fallback if an entire column was non-finite
        self.max_finite_ = np.nan_to_num(self.max_finite_, nan=0.0)
        self.min_finite_ = np.nan_to_num(self.min_finite_, nan=0.0)
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Clip infinite values to the training partition bounds."""
        X_mat = np.asarray(X, dtype=np.float64).copy()
        if self.max_finite_ is None or self.min_finite_ is None:
            raise ValueError("InfClipper must be fitted before transforming data.")

        for j in range(X_mat.shape[1]):
            X_mat[np.isposinf(X_mat[:, j]), j] = self.max_finite_[j]
            X_mat[np.isneginf(X_mat[:, j]), j] = self.min_finite_[j]
            # Replace NaNs with median/finite bound
            X_mat[np.isnan(X_mat[:, j]), j] = self.max_finite_[j]

        return X_mat

    def get_feature_names_out(self, input_features: Optional[np.ndarray | List[str]] = None) -> np.ndarray:
        """Return feature names unchanged for downstream ColumnTransformer compatibility."""
        if input_features is None:
            n_feats = len(self.max_finite_) if self.max_finite_ is not None else 0
            return np.array([f"x{i}" for i in range(n_feats)], dtype=object)
        return np.asarray(input_features, dtype=object)


class FeatureSchemaSelector(BaseEstimator, TransformerMixin):
    """Filters predictor matrices to conform to primary or alternative schemas."""

    def __init__(self, schema: str = "primary") -> None:
        self.schema = schema
        self.selected_columns_: List[str] = (
            PRIMARY_FEATURE_COLUMNS if schema == "primary" else ALTERNATIVE_FEATURE_COLUMNS_NO_STTL
        )

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> FeatureSchemaSelector:
        """Validate presence of selected columns."""
        if isinstance(X, pd.DataFrame):
            missing = set(self.selected_columns_) - set(X.columns)
            if missing:
                raise ValueError(f"Input DataFrame is missing required schema features: {sorted(list(missing))}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Subset columns according to configured schema."""
        if isinstance(X, pd.DataFrame):
            return X[self.selected_columns_].copy()
        return X

    def get_feature_names_out(self, input_features: Optional[np.ndarray | List[str]] = None) -> np.ndarray:
        """Return selected schema columns."""
        return np.array(self.selected_columns_, dtype=object)


def build_categorical_encoder() -> OneHotEncoder:
    """Build a robust OneHotEncoder configured to safely ignore unseen categories during evaluation."""
    return OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        drop=None,
    )


def build_numerical_scaler(scaler_type: str = "standard") -> BaseEstimator:
    """Construct configured numerical scaler."""
    if scaler_type == "robust":
        return RobustScaler()
    return StandardScaler()
