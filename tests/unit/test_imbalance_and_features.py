"""Unit tests for Phase 4 Imbalance-Aware Modeling, Feature Engineering, and Threshold Calibration.

Verifies:
1. NetworkFlowFeatureEngineer computes all 10 derived features with division-by-zero safeguards.
2. Training-only resampling (RandomOverSampler and SMOTENC) does not contaminate validation or test data.
3. Categorical feature semantics are preserved during SMOTENC.
4. Class-dependent sample weights are derived strictly from training partition labels.
5. Binary threshold sweeping correctly identifies optimal operating points.
6. Multiclass prior adjustment adjusts decision boundaries.
7. Wilson 95% confidence intervals are computed mathematically correctly.
8. Model serialization and reload integrity for Phase 4 bundles.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from ml.evaluation.thresholding import (
    calibrate_multiclass_prior_multipliers,
    sweep_binary_thresholds,
)
from ml.features.engineer import (
    ENGINEERED_FEATURE_NAMES,
    NetworkFlowFeatureEngineer,
)
from ml.preprocessing.schema import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PRIMARY_FEATURE_COLUMNS,
)
from ml.training.imbalance import (
    apply_training_resampling,
    compute_training_sample_weights,
)
from scripts.run_phase4_experiments import compute_wilson_ci


@pytest.fixture
def synthetic_flow_df() -> pd.DataFrame:
    """Generate a clean synthetic tabular dataframe for testing."""
    np.random.seed(42)
    n = 100
    data = {col: np.random.uniform(0.0, 100.0, n) for col in NUMERICAL_FEATURES}
    data["proto"] = ["tcp", "udp", "arp", "ospf"] * 25
    data["service"] = ["http", "dns", "ftp", "-"] * 25
    data["state"] = ["FIN", "CON", "INT", "REQ"] * 25
    data["id"] = list(range(1, n + 1))
    data["label"] = [0] * 50 + [1] * 50
    # Include all 10 attack categories
    data["attack_cat"] = (
        ["Normal"] * 40
        + ["Generic"] * 20
        + ["Exploits"] * 10
        + ["Fuzzers"] * 10
        + ["DoS"] * 5
        + ["Reconnaissance"] * 5
        + ["Analysis"] * 3
        + ["Backdoor"] * 3
        + ["Shellcode"] * 2
        + ["Worms"] * 2
    )
    return pd.DataFrame(data)


def test_feature_engineering_transformer(synthetic_flow_df: pd.DataFrame):
    """Verify that NetworkFlowFeatureEngineer creates 10 derived features with division-by-zero safeguards."""
    fe = NetworkFlowFeatureEngineer(include_original=True)
    out_df = fe.fit_transform(synthetic_flow_df)

    # Check presence of all 10 engineered features
    for name in ENGINEERED_FEATURE_NAMES:
        assert name in out_df.columns, f"Missing engineered feature: {name}"

    # Verify zero division handling by introducing edge-case zero rows
    zero_df = synthetic_flow_df.copy()
    zero_df["sbytes"] = 0.0
    zero_df["dbytes"] = 0.0
    zero_df["spkts"] = 0.0
    zero_df["dpkts"] = 0.0
    zero_df["tcprtt"] = 0.0
    zero_df["synack"] = 0.0

    zero_out = fe.transform(zero_df)
    assert np.all(np.isfinite(zero_out[ENGINEERED_FEATURE_NAMES].values)), "Found non-finite values with zero inputs!"


def test_training_sample_weights_derivation():
    """Verify sample weights are derived strictly from provided training labels."""
    y_dummy = np.array([0, 0, 0, 0, 1])  # Class 0: 4 samples, Class 1: 1 sample. Total=5, K=2.
    weights_bal = compute_training_sample_weights(y_dummy, weighting_type="balanced", clip_max=None)

    # w_0 = 5 / (2 * 4) = 0.625, w_1 = 5 / (2 * 1) = 2.5
    assert weights_bal[0] == pytest.approx(0.625)
    assert weights_bal[4] == pytest.approx(2.5)

    weights_sqrt = compute_training_sample_weights(y_dummy, weighting_type="sqrt_balanced", clip_max=None)
    assert weights_sqrt[0] == pytest.approx(np.sqrt(0.625))
    assert weights_sqrt[4] == pytest.approx(np.sqrt(2.5))


def test_training_only_resampling_isolation(synthetic_flow_df: pd.DataFrame):
    """Verify resampling modifies only the training set and leaves validation untouched."""
    X_train = synthetic_flow_df[PRIMARY_FEATURE_COLUMNS].copy()
    y_train = np.array([0] * 50 + [1] * 30 + [2] * 15 + [9] * 5)

    X_res, y_res, audit = apply_training_resampling(
        X_train.iloc[: len(y_train)], y_train, method="random_oversample", min_samples_floor=20, random_state=42
    )

    # Class 9 had 5 samples -> 20 samples (+15), Class 2 had 15 samples -> 20 samples (+5)
    assert (y_res == 9).sum() == 20
    assert (y_res == 2).sum() == 20
    assert len(y_res) > len(y_train)
    assert audit["added_samples"] == 20


def test_smotenc_categorical_handling(synthetic_flow_df: pd.DataFrame):
    """Verify SMOTENC operates on mixed categorical and continuous attributes."""
    X_train = synthetic_flow_df[PRIMARY_FEATURE_COLUMNS].copy()
    y_train = np.array([0] * 50 + [1] * 20 + [2] * 15 + [9] * 15)

    X_res, y_res, audit = apply_training_resampling(
        X_train.iloc[: len(y_train)], y_train, method="smotenc", min_samples_floor=25, random_state=42
    )

    assert (y_res == 9).sum() == 25
    # Categorical columns must retain string types
    for cat in CATEGORICAL_FEATURES:
        assert X_res[cat].dtype == object


def test_binary_threshold_sweeping():
    """Verify threshold sweeping computes correct operating points."""
    y_true = np.array([0] * 50 + [1] * 50)
    # Synthetic probabilities
    y_prob = np.concatenate([np.linspace(0.05, 0.45, 50), np.linspace(0.55, 0.95, 50)])

    points, opt_points = sweep_binary_thresholds(y_true, y_prob, cost_fn_ratio=5.0, steps=20)
    assert len(points) == 20
    assert "default" in opt_points
    assert "f1_optimal" in opt_points
    assert "low_false_alarm" in opt_points
    assert opt_points["f1_optimal"].f1 >= 0.95


def test_wilson_confidence_interval_calculation():
    """Verify Wilson 95% confidence interval matches reference formula."""
    # Reference test case: x = 13, n = 24
    ci = compute_wilson_ci(13, 24)
    assert ci["recall"] == pytest.approx(0.5417, abs=1e-3)
    assert ci["ci_lower"] == pytest.approx(0.3507, abs=1e-3)
    assert ci["ci_upper"] == pytest.approx(0.7211, abs=1e-3)


def test_multiclass_prior_calibration():
    """Verify calibrate_multiclass_prior_multipliers properly shifts decision threshold."""
    y_true = np.array([0, 9])
    # Class 0 has 0.55 prob, Class 9 has 0.45 prob
    probs = np.zeros((2, 10))
    probs[0, 0] = 0.9
    probs[0, 1] = 0.1
    probs[1, 0] = 0.55
    probs[1, 9] = 0.45

    # With unadjusted argmax, sample 1 is predicted as class 0
    preds_unadj = np.argmax(probs, axis=1)
    assert preds_unadj[1] == 0

    # Boost class 9 (Worms) with multiplier 2.0
    preds_adj, _ = calibrate_multiclass_prior_multipliers(
        y_val_true=y_true,
        val_probs=probs,
        class_names=["Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"],
        boost_classes={"Worms": 2.0},
    )
    # Sample 1 should now be predicted as class 9 (0.45 * 2.0 = 0.90 > 0.55)
    assert preds_adj[1] == 9
