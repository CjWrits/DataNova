"""Unit tests for Phase 5 Explainability, SHAP Attributions, and Error Diagnostics.

Verifies:
1. SHAP TreeExplainer output matrix dimensions (N, D_features, K_classes).
2. Multiclass class-index alignment (indices 0..9 correspond to canonical class mapping).
3. Preprocessed feature-name extraction and dimensional alignment.
4. Correct aggregation of one-hot encoded columns back to original parent features.
5. Invariance of model parameters and predictions before and after explanation.
6. Non-finite, NaN, and missing value checks on computed SHAP attributions.
7. Correct mapping and attribution of the 10 domain-engineered flow features.
8. Serialization, reload, and attribution generation from frozen joblib model bundles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import pytest
import shap

from ml.explainability.shap_explainer import (
    ExplanationArtifact,
    audit_engineered_features_importance,
    compute_class_specific_importance,
    compute_global_feature_importance,
    compute_shap_explanations,
    extract_error_subsets,
    extract_features_and_transform,
)
from ml.features.engineer import ENGINEERED_FEATURE_NAMES
from ml.preprocessing.schema import (
    CATEGORICAL_FEATURES,
    MULTICLASS_LABEL_MAPPING,
    NUMERICAL_FEATURES,
    PRIMARY_FEATURE_COLUMNS,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def sample_validation_slice() -> pd.DataFrame:
    """Load a clean 50-row validation slice representing all classes."""
    val_csv = PROJECT_ROOT / "data" / "processed" / "unsw_nb15" / "multiclass" / "val.csv"
    if val_csv.exists():
        df_val = pd.read_csv(val_csv)
        # Select balanced subset of 50 rows
        classes = list(MULTICLASS_LABEL_MAPPING.keys())
        sub_indices = []
        for c in classes:
            c_rows = df_val[df_val["attack_cat"] == c]
            sub_indices.extend(c_rows.head(5).index.tolist())
        return df_val.loc[sub_indices].copy()
    else:
        # Fallback synthetic frame
        np.random.seed(42)
        n = 50
        data = {col: np.random.uniform(0.0, 100.0, n) for col in NUMERICAL_FEATURES}
        data["proto"] = ["tcp", "udp", "arp", "ospf", "sctp"] * 10
        data["service"] = ["http", "dns", "ftp", "-", "smtp"] * 10
        data["state"] = ["FIN", "CON", "INT", "REQ", "RST"] * 10
        data["attack_cat"] = (list(MULTICLASS_LABEL_MAPPING.keys()) * 5)
        return pd.DataFrame(data)


@pytest.fixture(scope="module")
def c8_model_bundle() -> Dict[str, Any]:
    """Load the frozen Phase 4 C8 model bundle."""
    bundle_path = PROJECT_ROOT / "artifacts" / "models" / "phase4" / "multiclass_primary_winner.joblib"
    assert bundle_path.exists(), f"Phase 4 C8 bundle not found at {bundle_path}"
    return joblib.load(bundle_path)


def test_shap_output_dimensions(c8_model_bundle: Dict[str, Any], sample_validation_slice: pd.DataFrame):
    """Verify SHAP attributions have shape (N, D_features, K_classes) matching classes and features."""
    artifact = compute_shap_explanations(
        bundle=c8_model_bundle,
        df_eval=sample_validation_slice,
        model_id="test_c8",
        model_name="Test C8 Model",
        max_eval_samples=25,
        random_state=42,
    )

    n_samples = len(artifact.raw_df)
    n_features = len(artifact.transformed_feature_names)
    n_classes = len(MULTICLASS_LABEL_MAPPING)

    assert artifact.shap_values.ndim == 3, "SHAP values for multiclass model must be 3-dimensional"
    assert artifact.shap_values.shape == (n_samples, n_features, n_classes), (
        f"Expected shape ({n_samples}, {n_features}, {n_classes}), got {artifact.shap_values.shape}"
    )


def test_multiclass_class_alignment(c8_model_bundle: Dict[str, Any], sample_validation_slice: pd.DataFrame):
    """Verify that class index corresponds strictly to MULTICLASS_LABEL_MAPPING."""
    artifact = compute_shap_explanations(
        bundle=c8_model_bundle,
        df_eval=sample_validation_slice,
        model_id="test_c8",
        model_name="Test C8 Model",
        max_eval_samples=20,
    )

    for cname, cidx in MULTICLASS_LABEL_MAPPING.items():
        assert artifact.class_names[cidx] == cname, (
            f"Class index mismatch: expected {cname} at index {cidx}, found {artifact.class_names[cidx]}"
        )


def test_feature_name_alignment_after_preprocessing(c8_model_bundle: Dict[str, Any], sample_validation_slice: pd.DataFrame):
    """Verify transformed feature names match the columns of X_processed exactly."""
    X_proc, trans_names, feat_map, unique_parents = extract_features_and_transform(
        c8_model_bundle, sample_validation_slice
    )

    assert X_proc.shape[1] == len(trans_names), (
        f"Feature dimension mismatch: X_proc has {X_proc.shape[1]} cols, but {len(trans_names)} names provided"
    )

    # Verify all 10 domain-engineered features are present
    for eng_feat in ENGINEERED_FEATURE_NAMES:
        assert eng_feat in trans_names, f"Engineered feature {eng_feat} missing from transformed names"


def test_original_feature_aggregation(c8_model_bundle: Dict[str, Any], sample_validation_slice: pd.DataFrame):
    """Verify aggregation folds one-hot dummy variables back into original parent features."""
    artifact = compute_shap_explanations(
        bundle=c8_model_bundle,
        df_eval=sample_validation_slice,
        model_id="test_c8",
        model_name="Test C8 Model",
        max_eval_samples=20,
    )

    df_agg = compute_global_feature_importance(artifact, aggregate_to_original=True)
    agg_features = set(df_agg["feature"])

    # Categorical parent columns must appear as unified parents, not individual one-hot dummies
    for cat in CATEGORICAL_FEATURES:
        assert cat in agg_features, f"Parent categorical feature {cat} missing from aggregated importance"

    # One-hot dummy prefixes should not be standalone features in aggregated table
    for f in agg_features:
        assert not f.startswith("cat__"), f"Aggregated feature list contains raw dummy name: {f}"


def test_model_prediction_invariance_under_explanation(c8_model_bundle: Dict[str, Any], sample_validation_slice: pd.DataFrame):
    """Verify running SHAP TreeExplainer does not alter model predictions or probability outputs."""
    model = c8_model_bundle["model"]
    X_proc, _, _, _ = extract_features_and_transform(c8_model_bundle, sample_validation_slice)

    preds_before = model.predict(X_proc)
    probs_before = model.predict_proba(X_proc)

    # Compute SHAP
    artifact = compute_shap_explanations(
        bundle=c8_model_bundle,
        df_eval=sample_validation_slice,
        model_id="test_invariance",
        model_name="Invariance Test",
        max_eval_samples=len(sample_validation_slice),
    )

    preds_after = model.predict(X_proc)
    probs_after = model.predict_proba(X_proc)

    np.testing.assert_array_equal(preds_before, preds_after, err_msg="Model predictions changed after SHAP explanation!")
    np.testing.assert_allclose(probs_before, probs_after, rtol=1e-6, err_msg="Model probabilities shifted after SHAP explanation!")


def test_missing_and_nonfinite_shap_values(c8_model_bundle: Dict[str, Any], sample_validation_slice: pd.DataFrame):
    """Verify all computed SHAP values are finite numbers (no NaN or Inf)."""
    artifact = compute_shap_explanations(
        bundle=c8_model_bundle,
        df_eval=sample_validation_slice,
        model_id="test_finite",
        model_name="Finite Test",
        max_eval_samples=25,
    )

    assert not np.isnan(artifact.shap_values).any(), "Found NaN in computed SHAP values!"
    assert not np.isinf(artifact.shap_values).any(), "Found Inf in computed SHAP values!"


def test_explanation_from_serialized_and_reloaded_bundle():
    """Verify serialized bundle reloaded from disk executes explanation without error."""
    bundle_path = PROJECT_ROOT / "artifacts" / "models" / "phase4" / "multiclass_nosttl_winner.joblib"
    reloaded_bundle = joblib.load(bundle_path)

    val_csv = PROJECT_ROOT / "data" / "processed" / "unsw_nb15" / "multiclass" / "val.csv"
    df_mini = pd.read_csv(val_csv, nrows=15)

    artifact = compute_shap_explanations(
        bundle=reloaded_bundle,
        df_eval=df_mini,
        model_id="test_reloaded_c10",
        model_name="Reloaded C10",
        max_eval_samples=10,
    )

    assert artifact.shap_values.shape[0] == 10
    assert "sttl" not in artifact.unique_original_features, "No-STTL model should not contain sttl feature"


def test_engineered_features_audit_comparison(c8_model_bundle: Dict[str, Any], sample_validation_slice: pd.DataFrame):
    """Verify engineered features audit correctly compares C8 and C10 models."""
    c10_bundle = joblib.load(PROJECT_ROOT / "artifacts" / "models" / "phase4" / "multiclass_nosttl_winner.joblib")

    art_c8 = compute_shap_explanations(c8_model_bundle, sample_validation_slice, "c8", "C8", max_eval_samples=20)
    art_c10 = compute_shap_explanations(c10_bundle, sample_validation_slice, "c10", "C10", max_eval_samples=20)

    df_audit = audit_engineered_features_importance(art_c8, art_c10)
    assert len(df_audit) == 10, "Audit must evaluate exactly 10 engineered features"
    assert set(df_audit["feature"]) == set(ENGINEERED_FEATURE_NAMES)


def test_c10_sttl_schema_isolation(sample_validation_slice: pd.DataFrame):
    """Verify modifying only STTL leaves C10 transformed matrix and predictions strictly identical."""
    bundle_path = PROJECT_ROOT / "artifacts" / "models" / "phase4" / "multiclass_nosttl_winner.joblib"
    c10_bundle = joblib.load(bundle_path)

    df1 = sample_validation_slice.head(10).copy()
    df1["sttl"] = 31.0

    df2 = sample_validation_slice.head(10).copy()
    df2["sttl"] = 254.0

    X1, names1, _, _ = extract_features_and_transform(c10_bundle, df1, schema="alternative_no_sttl")
    X2, names2, _, _ = extract_features_and_transform(c10_bundle, df2, schema="alternative_no_sttl")

    # Transformed representations must be strictly identical
    assert np.array_equal(X1, X2), "C10 transformed feature matrix changed under STTL variation"
    assert names1 == names2, "Transformed column names changed under STTL variation"
    assert "sttl" not in names1, "Transformed names contain STTL in C10 model"

    preds1 = c10_bundle["model"].predict(X1)
    preds2 = c10_bundle["model"].predict(X2)
    assert np.array_equal(preds1, preds2), "C10 predicted classes changed under STTL variation"

    probs1 = c10_bundle["model"].predict_proba(X1)
    probs2 = c10_bundle["model"].predict_proba(X2)
    assert np.allclose(probs1, probs2, atol=1e-7), "C10 output probabilities changed under STTL variation"


def test_shap_raw_margin_reconstruction(c8_model_bundle: Dict[str, Any], sample_validation_slice: pd.DataFrame):
    """Verify SHAP attributions sum to the booster's raw margin output within float32 tolerance."""
    import xgboost as xgb

    model = c8_model_bundle["model"]
    df_eval = sample_validation_slice.head(5)
    X_proc, trans_names, _, _ = extract_features_and_transform(c8_model_bundle, df_eval, schema="primary")

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_proc)
    expected_vals = explainer.expected_value

    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X_proc)
    raw_margins = booster.predict(dmatrix, output_margin=True)

    max_err = 0.0
    for i in range(len(df_eval)):
        for k in range(10):
            recon = expected_vals[k] + np.sum(shap_vals[i, :, k])
            actual = raw_margins[i, k]
            err = abs(recon - actual)
            if err > max_err:
                max_err = err

    assert max_err < 1e-4, f"SHAP raw-margin reconstruction error too large: {max_err}"


def test_shap_parent_attribution_sum_preservation(c8_model_bundle: Dict[str, Any], sample_validation_slice: pd.DataFrame):
    """Verify aggregating transformed feature attributions to parent features preserves exact sum."""
    df_eval = sample_validation_slice.head(5)
    X_proc, trans_names, feat_map, unique_parents = extract_features_and_transform(
        c8_model_bundle, df_eval, schema="primary"
    )

    explainer = shap.TreeExplainer(c8_model_bundle["model"])
    shap_vals = explainer.shap_values(X_proc)

    for i in range(len(df_eval)):
        for k in range(10):
            trans_sum = float(np.sum(shap_vals[i, :, k]))

            parent_attributions: Dict[str, float] = {p: 0.0 for p in unique_parents}
            for trans_idx, tname in enumerate(trans_names):
                parent = feat_map.get(tname, tname)
                parent_attributions[parent] += float(shap_vals[i, trans_idx, k])

            parent_sum = float(sum(parent_attributions.values()))
            diff = abs(trans_sum - parent_sum)
            assert diff < 1e-5, f"Parent attribution sum discrepancy: {diff}"
