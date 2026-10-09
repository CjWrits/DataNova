"""Preprocessing pipeline factory module.

Constructs unified, zero-leakage Scikit-learn ColumnTransformer pipelines
that chain feature schema validation, infinite value clipping, numerical scaling,
and categorical one-hot encoding with unseen category protection.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ml.preprocessing.schema import (
    ALTERNATIVE_FEATURE_COLUMNS_NO_STTL,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PRIMARY_FEATURE_COLUMNS,
    validate_feature_matrix_columns,
)
from ml.preprocessing.transformers import (
    FeatureSchemaSelector,
    InfClipper,
    build_categorical_encoder,
    build_numerical_scaler,
)


def create_preprocessing_pipeline(
    schema: str = "primary",
    scaler_type: str = "standard",
) -> Pipeline:
    """Create an un-fitted Scikit-learn Pipeline encapsulating full feature transformation.

    Parameters:
        schema: 'primary' (all 42 flow features) or 'alternative_no_sttl' (41 features excluding sttl).
        scaler_type: 'standard' (StandardScaler) or 'robust' (RobustScaler).

    Returns:
        Pipeline: Unfitted Scikit-learn pipeline ready to be fitted exclusively on X_train.
    """
    if schema not in ["primary", "alternative_no_sttl"]:
        raise ValueError(f"Invalid schema '{schema}'. Must be 'primary' or 'alternative_no_sttl'.")

    # Determine numerical feature set for selected schema
    if schema == "alternative_no_sttl":
        num_cols = [c for c in NUMERICAL_FEATURES if c != "sttl"]
    else:
        num_cols = list(NUMERICAL_FEATURES)

    cat_cols = list(CATEGORICAL_FEATURES)

    # Numerical sub-pipeline: clip infinites, then scale
    num_pipeline = Pipeline([
        ("inf_clipper", InfClipper()),
        ("scaler", build_numerical_scaler(scaler_type)),
    ])

    # Categorical sub-pipeline: one-hot encoding with unknown handling
    cat_pipeline = Pipeline([
        ("one_hot", build_categorical_encoder()),
    ])

    # Combine into ColumnTransformer
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # Complete pipeline with initial schema selection
    full_pipeline = Pipeline([
        ("schema_selector", FeatureSchemaSelector(schema=schema)),
        ("preprocessor", column_transformer),
    ])

    return full_pipeline


def prepare_train_val_test_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    schema: str = "primary",
    scaler_type: str = "standard",
) -> Tuple[Pipeline, Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
    """Fit preprocessing pipeline strictly on train_df and transform val_df and test_df without leakage.

    Parameters:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Testing DataFrame.
        target_col: 'label' (binary) or 'attack_cat' (multiclass).
        schema: 'primary' or 'alternative_no_sttl'.
        scaler_type: 'standard' or 'robust'.

    Returns:
        (fitted_pipeline, (X_train_trans, y_train), (X_val_trans, y_val), (X_test_trans, y_test))
    """
    # Separate predictors and targets
    feature_cols = PRIMARY_FEATURE_COLUMNS if schema == "primary" else ALTERNATIVE_FEATURE_COLUMNS_NO_STTL

    # Validate that no targets or identifiers are present
    valid, violations = validate_feature_matrix_columns(feature_cols, schema=schema)
    if not valid:
        raise ValueError(f"Feature validation failed: {violations}")

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()

    X_val = val_df[feature_cols].copy()
    y_val = val_df[target_col].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    # Instantiate and fit exclusively on training data
    pipeline = create_preprocessing_pipeline(schema=schema, scaler_type=scaler_type)
    X_train_trans = pipeline.fit_transform(X_train)
    X_val_trans = pipeline.transform(X_val)
    X_test_trans = pipeline.transform(X_test)

    # Retrieve transformed feature names
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    X_train_out = pd.DataFrame(X_train_trans, columns=feature_names, index=X_train.index)
    X_val_out = pd.DataFrame(X_val_trans, columns=feature_names, index=X_val.index)
    X_test_out = pd.DataFrame(X_test_trans, columns=feature_names, index=X_test.index)

    return pipeline, (X_train_out, y_train), (X_val_out, y_val), (X_test_out, y_test)
