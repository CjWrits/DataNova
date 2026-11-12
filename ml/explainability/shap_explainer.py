"""SHAP Explainability and Interpretability Engine for Network Intrusion Detection.

Provides unified, memory-efficient model explanation workflows:
1. TreeExplainer attribution computation for Multiclass and Binary XGBoost models.
2. Robust feature-name mapping and aggregation from one-hot dummy variables to parent features.
3. Global and class-specific feature attribution rankings.
4. Systematic error analysis: False Positives (benign flagged as attacks) and False Negatives (missed attacks).
5. Comprehensive domain-engineered feature contribution auditing.
6. Comparative explanation analysis across Phase 3 Baseline, Phase 4 C8 (Winner), and Phase 4 C10 (No-STTL).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.features.engineer import ENGINEERED_FEATURE_NAMES, NetworkFlowFeatureEngineer
from ml.preprocessing.schema import (
    ALTERNATIVE_FEATURE_COLUMNS_NO_STTL,
    CATEGORICAL_FEATURES,
    MULTICLASS_LABEL_MAPPING,
    PRIMARY_FEATURE_COLUMNS,
)


@dataclass
class ExplanationArtifact:
    """Encapsulates computed SHAP values, feature mappings, and predictions."""
    model_id: str
    model_name: str
    schema: str
    class_names: List[str]
    shap_values: np.ndarray  # Shape: (N, D_features, K_classes) or (N, D_features) for binary
    base_values: np.ndarray  # Shape: (K_classes,) or (N, K_classes)
    transformed_feature_names: List[str]
    original_feature_mapping: Dict[str, str]  # trans_name -> original_parent_name
    unique_original_features: List[str]
    X_processed: np.ndarray
    raw_df: pd.DataFrame
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray


def extract_features_and_transform(
    bundle: Dict[str, Any],
    df: pd.DataFrame,
    schema: Optional[str] = None,
) -> Tuple[np.ndarray, List[str], Dict[str, str], List[str]]:
    """Transform raw dataframe using the model bundle's preprocessor and return feature names.

    Returns:
        (X_processed, transformed_feature_names, original_feature_mapping, unique_original_features)
    """
    preprocessor = bundle["preprocessor"]
    resolved_schema = schema or bundle.get("schema", "primary")
    feature_cols = PRIMARY_FEATURE_COLUMNS if resolved_schema == "primary" else ALTERNATIVE_FEATURE_COLUMNS_NO_STTL

    X_raw = df[feature_cols].copy()

    # Case 1: Phase 4 bundle (tuple of Pipeline, NetworkFlowFeatureEngineer, StandardScaler)
    if isinstance(preprocessor, (tuple, list)):
        prep_pipe, fe_transformer, fe_scaler = preprocessor
        X_fe = fe_transformer.transform(X_raw)

        base_cols = [c for c in X_fe.columns if not c.startswith("feat_")]
        new_cols = [c for c in X_fe.columns if c.startswith("feat_")]

        X_base = prep_pipe.transform(X_fe[base_cols])
        X_new = fe_scaler.transform(X_fe[new_cols])
        X_proc = np.hstack([X_base, X_new])

        # Feature names
        base_names = list(prep_pipe.get_feature_names_out())
        eng_names = list(new_cols)
        transformed_names = base_names + eng_names
    else:
        # Case 2: Phase 3 bundle (single Pipeline)
        prep_pipe = preprocessor
        X_proc = prep_pipe.transform(X_raw)
        transformed_names = list(prep_pipe.get_feature_names_out())

    # Build mapping from transformed dummy/scaled feature to parent original feature
    feature_mapping: Dict[str, str] = {}
    for tname in transformed_names:
        if tname.startswith("num__"):
            feature_mapping[tname] = tname[5:]
        elif tname.startswith("cat__"):
            col_part = tname[5:]
            for cat_col in CATEGORICAL_FEATURES:
                if col_part.startswith(cat_col):
                    feature_mapping[tname] = cat_col
                    break
            else:
                feature_mapping[tname] = col_part.split("_")[0]
        elif tname.startswith("feat_"):
            feature_mapping[tname] = tname
        else:
            feature_mapping[tname] = tname

    unique_parents = sorted(list(set(feature_mapping.values())))
    return X_proc, transformed_names, feature_mapping, unique_parents


def compute_shap_explanations(
    bundle: Dict[str, Any],
    df_eval: pd.DataFrame,
    model_id: str,
    model_name: str,
    class_names: Optional[List[str]] = None,
    max_eval_samples: int = 1000,
    random_state: int = 42,
) -> ExplanationArtifact:
    """Compute exact SHAP tree attributions for a serialized model bundle."""
    model = bundle["model"]
    resolved_classes = class_names or list(MULTICLASS_LABEL_MAPPING.keys())
    schema = bundle.get("schema", "primary")

    # Stratified subsampling if evaluation sample exceeds max_eval_samples
    target_col = "attack_cat" if "attack_cat" in df_eval.columns else ("label" if "label" in df_eval.columns else None)
    if len(df_eval) > max_eval_samples and target_col is not None:
        # Sample with stratification across available classes without groupby.apply warnings
        sampled_indices = []
        for _, g in df_eval.groupby(target_col, observed=False):
            n_g = max(1, int(np.round(len(g) / len(df_eval) * max_eval_samples)))
            sampled_indices.extend(g.sample(n=min(len(g), n_g), random_state=random_state).index.tolist())
        eval_sample = df_eval.loc[sampled_indices].copy()
        if len(eval_sample) > max_eval_samples:
            eval_sample = eval_sample.iloc[:max_eval_samples].copy()
    elif len(df_eval) > max_eval_samples:
        eval_sample = df_eval.sample(n=max_eval_samples, random_state=random_state).copy()
    else:
        eval_sample = df_eval.copy()

    X_proc, trans_names, feat_map, unique_parents = extract_features_and_transform(bundle, eval_sample, schema=schema)

    # Predictions & probabilities
    preds = model.predict(X_proc)
    probs = model.predict_proba(X_proc)

    if target_col == "attack_cat":
        y_true = eval_sample["attack_cat"].map(MULTICLASS_LABEL_MAPPING).astype(int).values
    elif target_col == "label":
        y_true = eval_sample["label"].astype(int).values
    else:
        y_true = preds.copy()

    # Instantiate TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_vals_raw = explainer(X_proc)

    # Extract numpy values
    if hasattr(shap_vals_raw, "values"):
        shap_values = shap_vals_raw.values
        base_values = shap_vals_raw.base_values
    else:
        shap_values = np.asarray(shap_vals_raw)
        base_values = np.asarray(explainer.expected_value)

    return ExplanationArtifact(
        model_id=model_id,
        model_name=model_name,
        schema=schema,
        class_names=resolved_classes,
        shap_values=shap_values,
        base_values=base_values,
        transformed_feature_names=trans_names,
        original_feature_mapping=feat_map,
        unique_original_features=unique_parents,
        X_processed=X_proc,
        raw_df=eval_sample,
        y_true=y_true,
        y_pred=preds,
        y_prob=probs,
    )


def compute_global_feature_importance(
    artifact: ExplanationArtifact,
    aggregate_to_original: bool = True,
) -> pd.DataFrame:
    """Compute global mean absolute SHAP values across all samples and classes.

    Returns:
        DataFrame sorted by descending mean absolute SHAP importance.
    """
    sv = artifact.shap_values  # Shape: (N, D, K) or (N, D)
    if sv.ndim == 3:
        # Multiclass: mean absolute SHAP averaged across all classes
        # Mean across classes, then mean across samples
        mean_abs_per_feat = np.mean(np.mean(np.abs(sv), axis=2), axis=0)
    else:
        # Binary: mean absolute SHAP across samples
        mean_abs_per_feat = np.mean(np.abs(sv), axis=0)

    if not aggregate_to_original:
        df_imp = pd.DataFrame({
            "feature": artifact.transformed_feature_names,
            "mean_abs_shap": mean_abs_per_feat,
        })
        return df_imp.sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    # Aggregate one-hot dummy variables to original parent features
    parent_map = artifact.original_feature_mapping
    parent_imp: Dict[str, float] = {}

    for tname, val in zip(artifact.transformed_feature_names, mean_abs_per_feat):
        parent = parent_map[tname]
        parent_imp[parent] = parent_imp.get(parent, 0.0) + val

    df_agg = pd.DataFrame(list(parent_imp.items()), columns=["feature", "mean_abs_shap"])
    total_shap = df_agg["mean_abs_shap"].sum()
    df_agg["relative_pct"] = (df_agg["mean_abs_shap"] / total_shap) * 100.0 if total_shap > 0 else 0.0
    return df_agg.sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)


def compute_class_specific_importance(
    artifact: ExplanationArtifact,
    class_idx: int,
    aggregate_to_original: bool = True,
) -> pd.DataFrame:
    """Compute feature importance specifically for a designated class index (0..9)."""
    sv = artifact.shap_values
    if sv.ndim == 3:
        class_sv = sv[:, :, class_idx]
    else:
        class_sv = sv if class_idx == 1 else -sv

    mean_abs_per_feat = np.mean(np.abs(class_sv), axis=0)
    mean_signed_per_feat = np.mean(class_sv, axis=0)

    if not aggregate_to_original:
        df_class = pd.DataFrame({
            "feature": artifact.transformed_feature_names,
            "mean_abs_shap": mean_abs_per_feat,
            "mean_signed_shap": mean_signed_per_feat,
        })
        return df_class.sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    parent_map = artifact.original_feature_mapping
    parent_abs: Dict[str, float] = {}
    parent_signed: Dict[str, float] = {}

    for tname, abs_val, sign_val in zip(artifact.transformed_feature_names, mean_abs_per_feat, mean_signed_per_feat):
        parent = parent_map[tname]
        parent_abs[parent] = parent_abs.get(parent, 0.0) + abs_val
        parent_signed[parent] = parent_signed.get(parent, 0.0) + sign_val

    data = []
    for parent in parent_abs:
        data.append({
            "feature": parent,
            "mean_abs_shap": parent_abs[parent],
            "mean_signed_shap": parent_signed[parent],
        })

    df_out = pd.DataFrame(data)
    total_shap = df_out["mean_abs_shap"].sum()
    df_out["relative_pct"] = (df_out["mean_abs_shap"] / total_shap) * 100.0 if total_shap > 0 else 0.0
    return df_out.sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)


def extract_error_subsets(
    artifact: ExplanationArtifact,
) -> Dict[str, Dict[str, Any]]:
    """Segment explanation samples into key diagnostic error subsets:

    1. Benign False Alarms: True Normal ($y=0$) predicted as Attack ($y \ne 0$), with focus on Fuzzers ($y=3$).
    2. Missed Attacks: True Attack ($y \ne 0$) predicted as Normal ($y=0$), focusing on Fuzzers, Exploits, DoS.
    3. Minority Class Misclassifications: Analysis and DoS confusion patterns.
    """
    y_true = artifact.y_true
    y_pred = artifact.y_pred
    sv = artifact.shap_values
    classes = artifact.class_names
    feat_map = artifact.original_feature_mapping

    error_analyses: Dict[str, Dict[str, Any]] = {}

    # Subset A: False Alarms on Benign Traffic (y_true == 0, y_pred != 0)
    fa_mask = (y_true == 0) & (y_pred != 0)
    fa_fuzzers_mask = (y_true == 0) & (y_pred == 3)  # Predicted Fuzzers
    fa_exploits_mask = (y_true == 0) & (y_pred == 2)  # Predicted Exploits

    def summarize_drivers(mask: np.ndarray, target_class_idx: int) -> List[Dict[str, Any]]:
        if not np.any(mask):
            return []
        sub_sv = sv[mask, :, target_class_idx]
        mean_signed = np.mean(sub_sv, axis=0)

        # Aggregate to original parent features
        parent_signed: Dict[str, float] = {}
        for tname, sval in zip(artifact.transformed_feature_names, mean_signed):
            p = feat_map[tname]
            parent_signed[p] = parent_signed.get(p, 0.0) + sval

        sorted_drivers = sorted(parent_signed.items(), key=lambda x: abs(x[1]), reverse=True)
        return [{"feature": k, "mean_shap_pull": round(float(v), 5)} for k, v in sorted_drivers[:10]]

    error_analyses["benign_flagged_as_fuzzers"] = {
        "count": int(np.sum(fa_fuzzers_mask)),
        "description": "True Normal flows incorrectly classified as Fuzzers",
        "drivers_pushing_towards_fuzzers": summarize_drivers(fa_fuzzers_mask, target_class_idx=3),
    }

    error_analyses["benign_flagged_as_exploits"] = {
        "count": int(np.sum(fa_exploits_mask)),
        "description": "True Normal flows incorrectly classified as Exploits",
        "drivers_pushing_towards_exploits": summarize_drivers(fa_exploits_mask, target_class_idx=2),
    }

    # Subset B: Attacks Missed as Normal (y_true != 0, y_pred == 0)
    miss_fuzzers = (y_true == 3) & (y_pred == 0)
    miss_exploits = (y_true == 2) & (y_pred == 0)
    miss_dos = (y_true == 4) & (y_pred == 0)
    miss_analysis = (y_true == 6) & (y_pred == 0)

    error_analyses["fuzzers_missed_as_normal"] = {
        "count": int(np.sum(miss_fuzzers)),
        "description": "True Fuzzers attack flows misclassified as benign Normal traffic",
        "drivers_pushing_towards_normal": summarize_drivers(miss_fuzzers, target_class_idx=0),
    }

    error_analyses["exploits_missed_as_normal"] = {
        "count": int(np.sum(miss_exploits)),
        "description": "True Exploits attack flows misclassified as benign Normal traffic",
        "drivers_pushing_towards_normal": summarize_drivers(miss_exploits, target_class_idx=0),
    }

    error_analyses["dos_missed_as_normal"] = {
        "count": int(np.sum(miss_dos)),
        "description": "True DoS attacks misclassified as benign Normal traffic",
        "drivers_pushing_towards_normal": summarize_drivers(miss_dos, target_class_idx=0),
    }

    error_analyses["analysis_missed_as_normal"] = {
        "count": int(np.sum(miss_analysis)),
        "description": "True Analysis probes misclassified as benign Normal traffic",
        "drivers_pushing_towards_normal": summarize_drivers(miss_analysis, target_class_idx=0),
    }

    # Subset C: Attack-to-Attack Confusion (DoS and Worms confused with Exploits)
    dos_as_expl = (y_true == 4) & (y_pred == 2)
    worms_as_expl = (y_true == 9) & (y_pred == 2)

    error_analyses["dos_misclassified_as_exploits"] = {
        "count": int(np.sum(dos_as_expl)),
        "description": "True DoS attacks misclassified as Exploits",
        "drivers_pushing_towards_exploits": summarize_drivers(dos_as_expl, target_class_idx=2),
    }

    error_analyses["worms_misclassified_as_exploits"] = {
        "count": int(np.sum(worms_as_expl)),
        "description": "True Worms attacks misclassified as Exploits",
        "drivers_pushing_towards_exploits": summarize_drivers(worms_as_expl, target_class_idx=2),
    }

    return error_analyses


def audit_engineered_features_importance(
    artifact_c8: ExplanationArtifact,
    artifact_c10: ExplanationArtifact,
) -> pd.DataFrame:
    """Compare the SHAP contribution and rank of all 10 domain engineered features

    between Candidate 8 (Primary Schema with STTL) and Candidate 10 (No-STTL Schema).
    """
    imp_c8 = compute_global_feature_importance(artifact_c8, aggregate_to_original=False)
    imp_c10 = compute_global_feature_importance(artifact_c10, aggregate_to_original=False)

    total_c8 = imp_c8["mean_abs_shap"].sum()
    total_c10 = imp_c10["mean_abs_shap"].sum()

    records = []
    for feat in ENGINEERED_FEATURE_NAMES:
        # Candidate 8
        c8_row = imp_c8[imp_c8["feature"] == feat]
        if not c8_row.empty:
            c8_val = float(c8_row["mean_abs_shap"].values[0])
            c8_rank = int(c8_row.index[0]) + 1
            c8_pct = (c8_val / total_c8) * 100.0 if total_c8 > 0 else 0.0
        else:
            c8_val, c8_rank, c8_pct = 0.0, 999, 0.0

        # Candidate 10
        c10_row = imp_c10[imp_c10["feature"] == feat]
        if not c10_row.empty:
            c10_val = float(c10_row["mean_abs_shap"].values[0])
            c10_rank = int(c10_row.index[0]) + 1
            c10_pct = (c10_val / total_c10) * 100.0 if total_c10 > 0 else 0.0
        else:
            c10_val, c10_rank, c10_pct = 0.0, 999, 0.0

        records.append({
            "feature": feat,
            "c8_mean_abs_shap": round(c8_val, 5),
            "c8_rank": c8_rank,
            "c8_relative_pct": round(c8_pct, 2),
            "c10_mean_abs_shap": round(c10_val, 5),
            "c10_rank": c10_rank,
            "c10_relative_pct": round(c10_pct, 2),
            "rank_shift": c8_rank - c10_rank,  # positive means higher rank in C10
        })

    df_out = pd.DataFrame(records)
    return df_out.sort_values(by="c8_mean_abs_shap", ascending=False).reset_index(drop=True)
