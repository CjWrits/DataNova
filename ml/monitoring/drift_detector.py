"""Core Statistical Drift Detection Engine for Network Telemetry and NIDS Predictions.

Provides deterministic, reproducible distribution drift metrics:
1. Population Stability Index (PSI) for continuous numerical features (quantile-binned).
2. Kolmogorov-Smirnov (KS) two-sample divergence test.
3. Categorical distribution divergence and unseen / novel category detection.
4. Data quality metrics: missing-value rate, non-finite value rate, and zero-variance checks.
5. Model prediction behavior: prediction class distribution, prediction entropy, confidence degradation.
6. Small-batch safeguards to prevent spurious alert triggering on short bursts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from ml.preprocessing.schema import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PRIMARY_FEATURE_COLUMNS,
)


@dataclass
class NumericalDriftResult:
    feature_name: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    drift_level: str  # 'STABLE', 'MODERATE_DRIFT', 'CRITICAL_DRIFT'
    ref_mean: float
    batch_mean: float
    ref_std: float
    batch_std: float
    missing_rate: float
    non_finite_rate: float


@dataclass
class CategoricalDriftResult:
    feature_name: str
    psi: float
    drift_level: str
    unknown_category_count: int
    unknown_category_rate: float
    novel_categories: List[str]
    missing_rate: float


@dataclass
class PredictionDriftResult:
    model_id: str
    batch_size: int
    predicted_distribution: Dict[str, float]
    prediction_psi: float
    mean_confidence: float
    min_confidence: float
    low_confidence_rate: float  # Fraction of samples with max_prob < 0.60
    mean_entropy: float
    drift_level: str


@dataclass
class BatchMonitoringSummary:
    batch_id: str
    batch_size: int
    is_valid_size: bool
    overall_drift_status: str  # 'STABLE', 'LOW_DRIFT', 'MODERATE_DRIFT', 'CRITICAL_DRIFT'
    drifting_features_count: int
    critical_features_count: int
    numerical_metrics: Dict[str, NumericalDriftResult]
    categorical_metrics: Dict[str, CategoricalDriftResult]
    prediction_metrics: Optional[PredictionDriftResult] = None
    warnings: List[str] = field(default_factory=list)


class FeatureDriftDetector:
    """Computes statistical drift between a frozen reference dataset and incoming evaluation batches."""

    def __init__(
        self,
        ref_df: pd.DataFrame,
        psi_moderate_threshold: float = 0.10,
        psi_critical_threshold: float = 0.25,
        ks_pvalue_threshold: float = 0.01,
        min_batch_size: int = 30,
        num_quantile_bins: int = 10,
        eps: float = 1e-4,
    ) -> None:
        self.ref_df = ref_df.copy()
        self.psi_mod_th = psi_moderate_threshold
        self.psi_crit_th = psi_critical_threshold
        self.ks_pval_th = ks_pvalue_threshold
        self.min_batch_size = min_batch_size
        self.num_bins = num_quantile_bins
        self.eps = eps

        # Precompute reference quantiles and categorical frequency distributions
        self._ref_quantiles: Dict[str, np.ndarray] = {}
        self._ref_cat_freqs: Dict[str, Dict[str, float]] = {}

        self._fit_reference()

    def _fit_reference(self) -> None:
        """Precompute reference distribution bins and category tables."""
        # Numerical quantiles
        for col in self.ref_df.columns:
            if col in NUMERICAL_FEATURES:
                vals = self.ref_df[col].dropna().values.astype(float)
                if len(vals) > 0:
                    percentiles = np.linspace(0, 100, self.num_bins + 1)
                    q_bins = np.percentile(vals, percentiles)
                    # Deduplicate in case of heavy ties (e.g. zeros)
                    q_bins = np.unique(q_bins)
                    if len(q_bins) < 2:
                        q_bins = np.array([vals.min() - 1.0, vals.max() + 1.0])
                    else:
                        q_bins[0] = -np.inf
                        q_bins[-1] = np.inf
                    self._ref_quantiles[col] = q_bins

            elif col in CATEGORICAL_FEATURES:
                cat_counts = self.ref_df[col].dropna().astype(str).value_counts(normalize=True)
                self._ref_cat_freqs[col] = cat_counts.to_dict()

    def compute_numerical_psi(self, ref_vals: np.ndarray, batch_vals: np.ndarray, q_bins: np.ndarray) -> float:
        """Compute Population Stability Index across precomputed reference quantile bins."""
        # Filter non-finite for binning
        clean_ref = ref_vals[np.isfinite(ref_vals)]
        clean_batch = batch_vals[np.isfinite(batch_vals)]

        if len(clean_ref) == 0 or len(clean_batch) == 0:
            return 0.0

        ref_counts, _ = np.histogram(clean_ref, bins=q_bins)
        batch_counts, _ = np.histogram(clean_batch, bins=q_bins)

        p = (ref_counts / len(clean_ref)) + self.eps
        q = (batch_counts / len(clean_batch)) + self.eps

        p /= p.sum()
        q /= q.sum()

        psi = np.sum((q - p) * np.log(q / p))
        return float(max(0.0, psi))

    def evaluate_numerical_feature(self, col: str, batch_series: pd.Series) -> NumericalDriftResult:
        """Compute drift statistics for a single continuous numerical feature."""
        ref_series = self.ref_df[col] if col in self.ref_df.columns else pd.Series(dtype=float)

        ref_vals = ref_series.values.astype(float)
        batch_vals = batch_series.values.astype(float)

        n_samples = len(batch_vals)
        missing_cnt = np.isnan(batch_vals).sum()
        missing_rate = float(missing_cnt / n_samples) if n_samples > 0 else 0.0

        non_finite_cnt = (~np.isfinite(batch_vals)).sum()
        non_finite_rate = float(non_finite_cnt / n_samples) if n_samples > 0 else 0.0

        clean_ref = ref_vals[np.isfinite(ref_vals)]
        clean_batch = batch_vals[np.isfinite(batch_vals)]

        if len(clean_ref) == 0 or len(clean_batch) == 0:
            return NumericalDriftResult(
                feature_name=col,
                psi=0.0,
                ks_statistic=0.0,
                ks_pvalue=1.0,
                drift_level="STABLE",
                ref_mean=0.0,
                batch_mean=0.0,
                ref_std=0.0,
                batch_std=0.0,
                missing_rate=missing_rate,
                non_finite_rate=non_finite_rate,
            )

        # PSI
        q_bins = self._ref_quantiles.get(col, np.array([-np.inf, np.inf]))
        psi = self.compute_numerical_psi(clean_ref, clean_batch, q_bins)

        # Two-sample KS test
        ks_res = stats.ks_2samp(clean_ref, clean_batch)
        ks_stat = float(ks_res.statistic)
        ks_pval = float(ks_res.pvalue)

        # Determine drift severity
        if psi >= self.psi_crit_th or (ks_pval < self.ks_pval_th and psi >= self.psi_mod_th * 1.5):
            drift_level = "CRITICAL_DRIFT"
        elif psi >= self.psi_mod_th or ks_pval < self.ks_pval_th:
            drift_level = "MODERATE_DRIFT"
        else:
            drift_level = "STABLE"

        return NumericalDriftResult(
            feature_name=col,
            psi=round(psi, 4),
            ks_statistic=round(ks_stat, 4),
            ks_pvalue=round(ks_pval, 6),
            drift_level=drift_level,
            ref_mean=round(float(np.mean(clean_ref)), 4),
            batch_mean=round(float(np.mean(clean_batch)), 4),
            ref_std=round(float(np.std(clean_ref)), 4),
            batch_std=round(float(np.std(clean_batch)), 4),
            missing_rate=round(missing_rate, 4),
            non_finite_rate=round(non_finite_rate, 4),
        )

    def evaluate_categorical_feature(self, col: str, batch_series: pd.Series) -> CategoricalDriftResult:
        """Compute categorical frequency drift and identify unseen novel categories."""
        ref_freqs = self._ref_cat_freqs.get(col, {})
        n_samples = len(batch_series)

        missing_cnt = batch_series.isna().sum()
        missing_rate = float(missing_cnt / n_samples) if n_samples > 0 else 0.0

        clean_batch = batch_series.dropna().astype(str)
        if len(clean_batch) == 0:
            return CategoricalDriftResult(
                feature_name=col,
                psi=0.0,
                drift_level="STABLE",
                unknown_category_count=0,
                unknown_category_rate=0.0,
                novel_categories=[],
                missing_rate=missing_rate,
            )

        batch_freqs = clean_batch.value_counts(normalize=True).to_dict()

        # Identify novel categories
        known_cats = set(ref_freqs.keys())
        batch_cats = set(batch_freqs.keys())
        novel_cats = sorted(list(batch_cats - known_cats))

        novel_count = int(clean_batch.isin(novel_cats).sum())
        novel_rate = float(novel_count / len(clean_batch))

        # Categorical PSI across union of categories
        all_cats = sorted(list(known_cats | batch_cats))
        p = np.array([ref_freqs.get(c, 0.0) + self.eps for c in all_cats])
        q = np.array([batch_freqs.get(c, 0.0) + self.eps for c in all_cats])

        p /= p.sum()
        q /= q.sum()

        psi = float(np.sum((q - p) * np.log(q / p)))
        psi = max(0.0, psi)

        if psi >= self.psi_crit_th or novel_rate >= 0.05:
            drift_level = "CRITICAL_DRIFT"
        elif psi >= self.psi_mod_th or novel_rate > 0.0:
            drift_level = "MODERATE_DRIFT"
        else:
            drift_level = "STABLE"

        return CategoricalDriftResult(
            feature_name=col,
            psi=round(psi, 4),
            drift_level=drift_level,
            unknown_category_count=novel_count,
            unknown_category_rate=round(novel_rate, 4),
            novel_categories=novel_cats,
            missing_rate=round(missing_rate, 4),
        )

    def evaluate_batch(
        self,
        batch_df: pd.DataFrame,
        batch_id: str = "batch_001",
        features_to_monitor: Optional[List[str]] = None,
    ) -> BatchMonitoringSummary:
        """Run comprehensive feature distribution monitoring on an incoming batch."""
        n_samples = len(batch_df)
        is_valid = n_samples >= self.min_batch_size
        warnings = []

        if not is_valid:
            warnings.append(
                f"Batch size ({n_samples}) is below minimum reliable safeguard ({self.min_batch_size}). "
                "Statistical metrics may exhibit small-sample variance."
            )

        target_features = features_to_monitor or [
            c for c in batch_df.columns if c in NUMERICAL_FEATURES or c in CATEGORICAL_FEATURES or c.startswith("feat_")
        ]

        num_results: Dict[str, NumericalDriftResult] = {}
        cat_results: Dict[str, CategoricalDriftResult] = {}

        drifting_count = 0
        critical_count = 0

        for col in target_features:
            if col not in batch_df.columns:
                continue

            if col in CATEGORICAL_FEATURES or batch_df[col].dtype == object:
                res_c = self.evaluate_categorical_feature(col, batch_df[col])
                cat_results[col] = res_c
                if res_c.drift_level in ("MODERATE_DRIFT", "CRITICAL_DRIFT"):
                    drifting_count += 1
                if res_c.drift_level == "CRITICAL_DRIFT":
                    critical_count += 1
            else:
                res_n = self.evaluate_numerical_feature(col, batch_df[col])
                num_results[col] = res_n
                if res_n.drift_level in ("MODERATE_DRIFT", "CRITICAL_DRIFT"):
                    drifting_count += 1
                if res_n.drift_level == "CRITICAL_DRIFT":
                    critical_count += 1

        # Determine overall batch status
        if critical_count >= 3 or (critical_count >= 1 and drifting_count >= 5):
            overall_status = "CRITICAL_DRIFT"
        elif critical_count >= 1 or drifting_count >= 3:
            overall_status = "MODERATE_DRIFT"
        elif drifting_count >= 1:
            overall_status = "LOW_DRIFT"
        else:
            overall_status = "STABLE"

        return BatchMonitoringSummary(
            batch_id=batch_id,
            batch_size=n_samples,
            is_valid_size=is_valid,
            overall_drift_status=overall_status,
            drifting_features_count=drifting_count,
            critical_features_count=critical_count,
            numerical_metrics=num_results,
            categorical_metrics=cat_results,
            warnings=warnings,
        )


class ModelPredictionMonitor:
    """Monitors model inference outputs, class distribution shifts, confidence, and prediction entropy."""

    def __init__(
        self,
        class_names: List[str],
        ref_class_distribution: Optional[Dict[str, float]] = None,
        confidence_warning_threshold: float = 0.60,
    ) -> None:
        self.class_names = class_names
        self.ref_class_dist = ref_class_distribution or {c: 1.0 / len(class_names) for c in class_names}
        self.conf_warn_th = confidence_warning_threshold

    def evaluate_predictions(
        self,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        model_id: str = "model",
    ) -> PredictionDriftResult:
        """Analyze batch predictions for confidence degradation and distribution divergence."""
        n_samples = len(y_pred)
        if n_samples == 0:
            return PredictionDriftResult(
                model_id=model_id,
                batch_size=0,
                predicted_distribution={},
                prediction_psi=0.0,
                mean_confidence=0.0,
                min_confidence=0.0,
                low_confidence_rate=0.0,
                mean_entropy=0.0,
                drift_level="STABLE",
            )

        # 1. Prediction class distribution
        unique_preds, counts = np.unique(y_pred, return_counts=True)
        pred_dist: Dict[str, float] = {c: 0.0 for c in self.class_names}
        for u, cnt in zip(unique_preds, counts):
            if int(u) < len(self.class_names):
                pred_dist[self.class_names[int(u)]] = float(cnt / n_samples)

        # 2. Prediction PSI against reference distribution
        eps = 1e-4
        p = np.array([self.ref_class_dist.get(c, 0.0) + eps for c in self.class_names])
        q = np.array([pred_dist.get(c, 0.0) + eps for c in self.class_names])
        p /= p.sum()
        q /= q.sum()
        pred_psi = float(max(0.0, np.sum((q - p) * np.log(q / p))))

        # 3. Confidence statistics
        max_probs = np.max(y_prob, axis=1) if y_prob.ndim == 2 else np.maximum(y_prob, 1.0 - y_prob)
        mean_conf = float(np.mean(max_probs))
        min_conf = float(np.min(max_probs))
        low_conf_cnt = int((max_probs < self.conf_warn_th).sum())
        low_conf_rate = float(low_conf_cnt / n_samples)

        # 4. Prediction Entropy
        if y_prob.ndim == 2:
            clipped_probs = np.clip(y_prob, 1e-9, 1.0)
            sample_entropies = -np.sum(clipped_probs * np.log(clipped_probs), axis=1)
            mean_entropy = float(np.mean(sample_entropies))
        else:
            p1 = np.clip(y_prob, 1e-9, 1.0 - 1e-9)
            p0 = 1.0 - p1
            mean_entropy = float(np.mean(-(p1 * np.log(p1) + p0 * np.log(p0))))

        # 5. Determine prediction drift level
        if pred_psi >= 0.25 or low_conf_rate >= 0.35:
            drift_level = "CRITICAL_DRIFT"
        elif pred_psi >= 0.10 or low_conf_rate >= 0.15:
            drift_level = "MODERATE_DRIFT"
        else:
            drift_level = "STABLE"

        return PredictionDriftResult(
            model_id=model_id,
            batch_size=n_samples,
            predicted_distribution={k: round(v, 4) for k, v in pred_dist.items()},
            prediction_psi=round(pred_psi, 4),
            mean_confidence=round(mean_conf, 4),
            min_confidence=round(min_conf, 4),
            low_confidence_rate=round(low_conf_rate, 4),
            mean_entropy=round(mean_entropy, 4),
            drift_level=drift_level,
        )
