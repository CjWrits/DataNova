"""Baseline model training and evaluation engine.

Implements the 12 Phase 3 baseline machine learning experiments:
- 2 Tasks: Binary ('label') vs Multiclass ('attack_cat')
- 2 Schemas: Primary (42 features, includes 'sttl') vs No-STTL Ablation (41 features)
- 3 Models: Logistic Regression (linear), Random Forest (bagged tree), XGBoost (boosted tree)

Strict phase adherence:
- No resampling (SMOTE / ADASYN)
- No class weighting / focal loss
- No hyperparameter tuning
- No threshold manipulation
- Strict test set isolation during model selection
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from ml.evaluation.metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
)
from ml.preprocessing.pipeline import create_preprocessing_pipeline
from ml.preprocessing.schema import (
    ALTERNATIVE_FEATURE_COLUMNS_NO_STTL,
    MULTICLASS_LABEL_MAPPING,
    PRIMARY_FEATURE_COLUMNS,
    validate_feature_matrix_columns,
)


def instantiate_baseline_model(
    model_type: str,
    task: str,
    random_state: int = 42,
) -> Any:
    """Instantiate a baseline model with fixed unweighted hyperparameters.

    Parameters:
        model_type: 'logistic_regression', 'random_forest', or 'xgboost'.
        task: 'binary' or 'multiclass'.
        random_state: Fixed random seed for reproducibility.

    Returns:
        Unfitted Scikit-learn or XGBoost estimator.
    """
    if model_type == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "xgboost":
        eval_metric = "logloss" if task == "binary" else "mlogloss"
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            eval_metric=eval_metric,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Choose from 'logistic_regression', 'random_forest', 'xgboost'.")


def extract_feature_importances(
    model: Any,
    feature_names: List[str],
    top_n: int = 25,
) -> List[Dict[str, Any]]:
    """Extract and rank feature importances or linear coefficients.

    Parameters:
        model: Fitted estimator.
        feature_names: List of preprocessed column names.
        top_n: Number of top features to return.

    Returns:
        List of dicts with feature name, importance weight, and rank.
    """
    if hasattr(model, "feature_importances_"):
        weights = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim == 2:
            weights = np.mean(np.abs(coef), axis=0)
        else:
            weights = np.abs(coef)
    else:
        return []

    if len(weights) != len(feature_names):
        return []

    sorted_indices = np.argsort(weights)[::-1]
    results = []
    for rank, idx in enumerate(sorted_indices[:top_n], start=1):
        results.append({
            "rank": rank,
            "feature": str(feature_names[idx]),
            "importance": round(float(weights[idx]), 6),
        })
    return results


def run_single_baseline_experiment(
    experiment_id: str,
    task: str,
    schema: str,
    model_type: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train and evaluate a single baseline model on train and val partitions.

    Guarantees:
    - Zero data leakage: preprocessor fitted exclusively on train_df.
    - Test partition is not touched.
    - Identifiers and targets excluded from predictor matrices.

    Parameters:
        experiment_id: Unique string identifier.
        task: 'binary' or 'multiclass'.
        schema: 'primary' (42 features) or 'alternative_no_sttl' (41 features).
        model_type: 'logistic_regression', 'random_forest', 'xgboost'.
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        random_state: Random seed.

    Returns:
        Dictionary containing experiment configuration, metrics, timings,
        feature importances, and the fitted pipeline and model objects.
    """
    target_col = "label" if task == "binary" else "attack_cat"
    feature_cols = PRIMARY_FEATURE_COLUMNS if schema == "primary" else ALTERNATIVE_FEATURE_COLUMNS_NO_STTL

    # Validate schema integrity
    valid, violations = validate_feature_matrix_columns(feature_cols, schema=schema)
    if not valid:
        raise ValueError(f"Feature validation failed for {experiment_id}: {violations}")

    # Prepare features and targets
    X_train_raw = train_df[feature_cols].copy()
    X_val_raw = val_df[feature_cols].copy()

    if task == "binary":
        y_train = train_df[target_col].astype(int).values
        y_val = val_df[target_col].astype(int).values
    else:
        # Encode multiclass string labels to 0..9
        y_train = train_df[target_col].map(MULTICLASS_LABEL_MAPPING).astype(int).values
        y_val = val_df[target_col].map(MULTICLASS_LABEL_MAPPING).astype(int).values

    # Preprocessing pipeline: fit ONLY on training set
    preprocessor = create_preprocessing_pipeline(schema=schema, scaler_type="standard")
    
    t0_prep = time.perf_counter()
    X_train_trans = preprocessor.fit_transform(X_train_raw)
    X_val_trans = preprocessor.transform(X_val_raw)
    prep_time = time.perf_counter() - t0_prep

    # Retrieve preprocessed feature column names
    feature_names = preprocessor.named_steps["preprocessor"].get_feature_names_out().tolist()

    # Instantiate model
    model = instantiate_baseline_model(model_type=model_type, task=task, random_state=random_state)

    # Train model
    t0_train = time.perf_counter()
    model.fit(X_train_trans, y_train)
    train_time = time.perf_counter() - t0_train

    # Validation inference
    t0_val = time.perf_counter()
    val_preds = model.predict(X_val_trans)
    val_probs = model.predict_proba(X_val_trans) if hasattr(model, "predict_proba") else None
    val_inf_time = time.perf_counter() - t0_val

    # Compute validation metrics
    if task == "binary":
        val_metrics = compute_binary_metrics(y_true=y_val, y_pred=val_preds, y_prob=val_probs)
    else:
        val_metrics = compute_multiclass_metrics(y_true=y_val, y_pred=val_preds, y_prob=val_probs)

    # Extract feature importances
    top_features = extract_feature_importances(model, feature_names, top_n=25)

    return {
        "experiment_id": experiment_id,
        "task": task,
        "schema": schema,
        "model_type": model_type,
        "n_features_raw": len(feature_cols),
        "n_features_transformed": len(feature_names),
        "train_samples": len(X_train_raw),
        "val_samples": len(X_val_raw),
        "preprocessing_seconds": round(prep_time, 3),
        "training_seconds": round(train_time, 3),
        "val_inference_seconds": round(val_inf_time, 3),
        "val_latency_ms_per_sample": round((val_inf_time / len(X_val_raw)) * 1000, 4),
        "val_metrics": val_metrics,
        "top_features": top_features,
        "fitted_pipeline": preprocessor,
        "fitted_model": model,
        "val_preds": val_preds,
        "val_probs": val_probs,
        "val_true": y_val,
    }


def evaluate_frozen_model_on_test(
    experiment_id: str,
    preprocessor: Any,
    model: Any,
    test_df: pd.DataFrame,
    task: str,
    schema: str,
) -> Dict[str, Any]:
    """Evaluate a frozen, selected baseline model strictly on the untouched test partition.

    Parameters:
        experiment_id: Experiment ID.
        preprocessor: Fitted preprocessor.
        model: Fitted model.
        test_df: Test DataFrame.
        task: 'binary' or 'multiclass'.
        schema: 'primary' or 'alternative_no_sttl'.

    Returns:
        Dictionary of test metrics, latency, and sample counts.
    """
    target_col = "label" if task == "binary" else "attack_cat"
    feature_cols = PRIMARY_FEATURE_COLUMNS if schema == "primary" else ALTERNATIVE_FEATURE_COLUMNS_NO_STTL

    X_test_raw = test_df[feature_cols].copy()

    if task == "binary":
        y_test = test_df[target_col].astype(int).values
    else:
        y_test = test_df[target_col].map(MULTICLASS_LABEL_MAPPING).astype(int).values

    # Transform using pre-fitted preprocessor (no fitting)
    t0_inf = time.perf_counter()
    X_test_trans = preprocessor.transform(X_test_raw)
    test_preds = model.predict(X_test_trans)
    test_probs = model.predict_proba(X_test_trans) if hasattr(model, "predict_proba") else None
    test_inf_time = time.perf_counter() - t0_inf

    if task == "binary":
        test_metrics = compute_binary_metrics(y_true=y_test, y_pred=test_preds, y_prob=test_probs)
    else:
        test_metrics = compute_multiclass_metrics(y_true=y_test, y_pred=test_preds, y_prob=test_probs)

    return {
        "test_samples": len(X_test_raw),
        "test_inference_seconds": round(test_inf_time, 3),
        "test_latency_ms_per_sample": round((test_inf_time / len(X_test_raw)) * 1000, 4),
        "test_metrics": test_metrics,
        "test_preds": test_preds,
        "test_probs": test_probs,
        "test_true": y_test,
    }


def select_best_model_for_configuration(
    experiments: List[Dict[str, Any]],
    task: str,
    schema: str,
) -> Dict[str, Any]:
    """Select the best baseline model for a given task and schema strictly using validation metrics.

    Selection Criteria:
    - Primary metric: Macro F1 (multiclass) or F1 / Balanced Accuracy (binary).
    - Secondary metric: PR-AUC (Average Precision) and false positive rate.
    - Tie-breaker / sanity: Minority class recall.
    """
    candidates = [
        exp for exp in experiments
        if exp["task"] == task and exp["schema"] == schema
    ]
    if not candidates:
        raise ValueError(f"No experiment candidates found for task={task}, schema={schema}")

    def score_candidate(exp: Dict[str, Any]) -> Tuple[float, float, float]:
        vm = exp["val_metrics"]
        if task == "binary":
            # (F1, PR-AUC, -FPR)
            f1 = vm.get("f1", 0.0)
            prauc = vm.get("pr_auc") or 0.0
            fpr = vm.get("false_positive_rate", 1.0)
            return (f1, prauc, -fpr)
        else:
            # (Macro F1, Macro PR-AUC, Worms Recall)
            mf1 = vm.get("macro_f1", 0.0)
            mprauc = vm.get("macro_pr_auc") or 0.0
            worms_recall = vm.get("per_class", {}).get("Worms", {}).get("recall", 0.0)
            return (mf1, mprauc, worms_recall)

    best_candidate = max(candidates, key=score_candidate)
    return best_candidate
