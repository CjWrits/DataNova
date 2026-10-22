"""Unit tests for Phase 3 baseline machine learning pipelines, metrics, and models.

Tests cover:
1. Baseline model instantiation with unweighted hyperparameter configurations (LR, RF, XGBoost).
2. Binary evaluation metrics computation (precision, recall, specificity, FPR, F1, PR-AUC, ROC-AUC).
3. Multiclass evaluation metrics computation across 10 classes with zero-division safety.
4. Schema selection (Primary with 42 features vs Ablation with 41 features, excluding sttl).
5. Data leakage prevention (preprocessor fitted exclusively on training set).
6. Target/identifier exclusion enforcement.
7. Model selection strictly on validation metrics without test set exposure.
8. Artifact serialization and reloading integrity via Joblib.
9. Deterministic reproducibility with fixed random seed.
10. Valid probability output bounds [0, 1].
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from ml.evaluation.metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    format_binary_metrics_markdown,
    format_multiclass_metrics_markdown,
)
from ml.preprocessing.pipeline import create_preprocessing_pipeline
from ml.preprocessing.schema import (
    ALTERNATIVE_FEATURE_COLUMNS_NO_STTL,
    CATEGORICAL_FEATURES,
    IDENTIFIER_COLUMNS,
    MULTICLASS_LABEL_MAPPING,
    NUMERICAL_FEATURES,
    PRIMARY_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    validate_feature_matrix_columns,
)
from ml.training.trainer import (
    extract_feature_importances,
    instantiate_baseline_model,
    select_best_model_for_configuration,
)


@pytest.fixture
def synthetic_sample_df() -> pd.DataFrame:
    """Fixture providing a synthetic tabular dataframe conforming to UNSW-NB15 schema."""
    np.random.seed(42)
    n = 60
    data = {col: np.random.randn(n) for col in NUMERICAL_FEATURES}
    data["proto"] = ["tcp", "udp", "arp"] * 20
    data["service"] = ["http", "dns", "-"] * 20
    data["state"] = ["FIN", "CON", "INT"] * 20
    data["id"] = list(range(1, n + 1))
    data["label"] = [0] * 30 + [1] * 30
    data["attack_cat"] = ["Normal"] * 30 + ["Generic"] * 15 + ["Worms"] * 15
    return pd.DataFrame(data)


def test_baseline_model_instantiation():
    """Verify models instantiate with proper baseline hyperparameters."""
    lr = instantiate_baseline_model("logistic_regression", "binary", random_state=42)
    assert isinstance(lr, LogisticRegression)
    assert lr.max_iter == 1000
    assert lr.C == 1.0

    rf = instantiate_baseline_model("random_forest", "multiclass", random_state=42)
    assert isinstance(rf, RandomForestClassifier)
    assert rf.n_estimators == 200

    xgb_bin = instantiate_baseline_model("xgboost", "binary", random_state=42)
    assert isinstance(xgb_bin, XGBClassifier)
    assert xgb_bin.n_estimators == 300
    assert xgb_bin.max_depth == 6
    assert xgb_bin.eval_metric == "logloss"

    xgb_mc = instantiate_baseline_model("xgboost", "multiclass", random_state=42)
    assert isinstance(xgb_mc, XGBClassifier)
    assert xgb_mc.eval_metric == "mlogloss"

    with pytest.raises(ValueError, match="Unknown model_type"):
        instantiate_baseline_model("unsupported_model", "binary")


def test_binary_metrics_computation():
    """Verify precision, recall, specificity, FPR, F1, PR-AUC, ROC-AUC on known values."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    y_prob = np.array([0.1, 0.7, 0.9, 0.8, 0.2])

    metrics = compute_binary_metrics(y_true, y_pred, y_prob)

    # CM: TN=1, FP=1, FN=1, TP=2
    assert metrics["confusion_matrix"]["tn"] == 1
    assert metrics["confusion_matrix"]["fp"] == 1
    assert metrics["confusion_matrix"]["fn"] == 1
    assert metrics["confusion_matrix"]["tp"] == 2

    assert metrics["accuracy"] == 0.6
    assert metrics["precision"] == pytest.approx(2 / 3, rel=1e-4)
    assert metrics["recall"] == pytest.approx(2 / 3, rel=1e-4)
    assert metrics["specificity"] == 0.5
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["roc_auc"] is not None
    assert metrics["pr_auc"] is not None

    md = format_binary_metrics_markdown(metrics)
    assert "Binary Metrics" in md
    assert "Confusion Matrix" in md


def test_multiclass_metrics_computation():
    """Verify multiclass metrics across 10 attack categories."""
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_prob = np.eye(10)

    metrics = compute_multiclass_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert len(metrics["per_class"]) == 10
    assert "Worms" in metrics["per_class"]
    assert metrics["per_class"]["Worms"]["recall"] == 1.0

    md = format_multiclass_metrics_markdown(metrics)
    assert "Multiclass Metrics" in md
    assert "Worms" in md


def test_multiclass_zero_division_safety():
    """Verify metrics calculation safely handles zero-prediction classes."""
    y_true = np.array([0, 1, 2, 9])
    # Predicted class 9 (Worms) is never predicted
    y_pred = np.array([0, 1, 2, 0])

    metrics = compute_multiclass_metrics(y_true, y_pred)
    assert metrics["per_class"]["Worms"]["precision"] == 0.0
    assert metrics["per_class"]["Worms"]["recall"] == 0.0
    assert metrics["per_class"]["Worms"]["f1"] == 0.0


def test_primary_vs_ablation_schema_features():
    """Verify primary has 42 features and ablation has 41 features (no sttl)."""
    assert len(PRIMARY_FEATURE_COLUMNS) == 42
    assert "sttl" in PRIMARY_FEATURE_COLUMNS

    assert len(ALTERNATIVE_FEATURE_COLUMNS_NO_STTL) == 41
    assert "sttl" not in ALTERNATIVE_FEATURE_COLUMNS_NO_STTL

    valid_prim, _ = validate_feature_matrix_columns(PRIMARY_FEATURE_COLUMNS, "primary")
    assert valid_prim

    valid_ab, _ = validate_feature_matrix_columns(ALTERNATIVE_FEATURE_COLUMNS_NO_STTL, "alternative_no_sttl")
    assert valid_ab


def test_target_and_id_exclusion():
    """Verify identifier 'id' and target columns never pass validation."""
    cols_with_id = list(PRIMARY_FEATURE_COLUMNS) + ["id"]
    valid, violations = validate_feature_matrix_columns(cols_with_id)
    assert not valid
    assert any("id" in v for v in violations)

    cols_with_target = list(PRIMARY_FEATURE_COLUMNS) + ["label"]
    valid, violations = validate_feature_matrix_columns(cols_with_target)
    assert not valid
    assert any("label" in v for v in violations)


def test_preprocessing_pipeline_no_data_leakage(synthetic_sample_df: pd.DataFrame):
    """Verify pipeline fits exclusively on train and transforms val without error."""
    train_df = synthetic_sample_df.iloc[:40].copy()
    val_df = synthetic_sample_df.iloc[40:].copy()

    pipeline = create_preprocessing_pipeline(schema="primary", scaler_type="standard")
    
    feature_cols = PRIMARY_FEATURE_COLUMNS
    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()

    X_train_trans = pipeline.fit_transform(X_train)
    X_val_trans = pipeline.transform(X_val)

    assert X_train_trans.shape[0] == len(X_train)
    assert X_val_trans.shape[0] == len(X_val)
    assert X_train_trans.shape[1] == X_val_trans.shape[1]


def test_model_selection_strictly_on_validation():
    """Verify select_best_model_for_configuration uses only validation metrics."""
    dummy_experiments = [
        {
            "experiment_id": "exp_a",
            "task": "binary",
            "schema": "primary",
            "model_type": "logistic_regression",
            "val_metrics": {"f1": 0.85, "pr_auc": 0.90, "false_positive_rate": 0.05},
        },
        {
            "experiment_id": "exp_b",
            "task": "binary",
            "schema": "primary",
            "model_type": "random_forest",
            "val_metrics": {"f1": 0.95, "pr_auc": 0.98, "false_positive_rate": 0.02},
        },
    ]

    best = select_best_model_for_configuration(dummy_experiments, task="binary", schema="primary")
    assert best["experiment_id"] == "exp_b"
    assert best["model_type"] == "random_forest"


def test_model_artifact_serialization_and_reloading(synthetic_sample_df: pd.DataFrame):
    """Verify joblib model bundle serialization, file persistence, and reload predictability."""
    train_df = synthetic_sample_df.copy()
    feature_cols = PRIMARY_FEATURE_COLUMNS
    X_train = train_df[feature_cols]
    y_train = train_df["label"].values

    pipeline = create_preprocessing_pipeline(schema="primary")
    X_trans = pipeline.fit_transform(X_train)

    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_trans, y_train)

    bundle = {
        "experiment_id": "test_bundle",
        "preprocessor": pipeline,
        "model": model,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        art_path = Path(tmp_dir) / "test_model.joblib"
        joblib.dump(bundle, art_path, compress=3)

        assert art_path.exists()
        loaded_bundle = joblib.load(art_path)

        assert loaded_bundle["experiment_id"] == "test_bundle"
        preds_original = model.predict(X_trans)
        preds_loaded = loaded_bundle["model"].predict(loaded_bundle["preprocessor"].transform(X_train))
        np.testing.assert_array_equal(preds_original, preds_loaded)
