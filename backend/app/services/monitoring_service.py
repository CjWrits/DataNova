"""Centralized Drift and Data Quality Monitoring Service.

Pre-computes and holds baseline reference distributions from the sealed training partition.
Evaluates incoming network telemetry batches for:
1. Continuous Numerical Feature Drift (Quantile-binned Population Stability Index & Kolmogorov-Smirnov).
2. Categorical Distribution Frequency Skew & Novel Unseen Categories.
3. Data Pipeline Quality: Missing-value (NaN) and non-finite (Inf) rates.
4. Model Prediction Drift: Class distribution divergence and confidence degradation.
5. Adaptive Operational Recommendations: Structured rule-based guidance for SOC operations.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.app.core.config import settings
from backend.app.schemas.flow import DriftDecisionItem, DriftMonitoringResponse
from ml.explainability.shap_explainer import extract_features_and_transform
from ml.monitoring.adaptive_engine import (
    AdaptiveActionType,
    AdaptiveDecision,
    AdaptiveResponseEngine,
    ComprehensiveAdaptivePolicyResult,
)
from ml.monitoring.drift_detector import (
    BatchMonitoringSummary,
    FeatureDriftDetector,
    ModelPredictionMonitor,
    PredictionDriftResult,
)
from ml.preprocessing.schema import MULTICLASS_LABEL_MAPPING

logger = logging.getLogger("backend.services.monitoring_service")


class MonitoringService:
    """Singleton service for evaluating feature drift and prediction shifts."""

    _instance: Optional[MonitoringService] = None

    def __init__(self) -> None:
        self.is_loaded: bool = False
        self.detector: Optional[FeatureDriftDetector] = None
        self.prediction_monitor: Optional[ModelPredictionMonitor] = None
        self.adaptive_engine: AdaptiveResponseEngine = AdaptiveResponseEngine()
        self.ref_row_count: int = 0
        self.classes: List[str] = list(MULTICLASS_LABEL_MAPPING.keys())

    @classmethod
    def get_instance(cls) -> MonitoringService:
        if cls._instance is None:
            cls._instance = MonitoringService()
        return cls._instance

    def initialize_reference(self) -> None:
        """Load baseline reference distributions strictly from the training partition."""
        logger.info("Initializing MonitoringService: fitting FeatureDriftDetector on training partition...")
        ref_path = settings.REFERENCE_DATA_PATH
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference training dataset not found at {ref_path}")

        df_train_ref = pd.read_csv(ref_path)
        self.ref_row_count = len(df_train_ref)

        self.detector = FeatureDriftDetector(
            ref_df=df_train_ref,
            psi_moderate_threshold=0.10,
            psi_critical_threshold=0.25,
            ks_pvalue_threshold=0.01,
            min_batch_size=settings.MIN_DRIFT_BATCH_SIZE,
        )

        # Baseline uniform / prior prediction monitor
        self.prediction_monitor = ModelPredictionMonitor(
            class_names=self.classes,
            confidence_warning_threshold=0.60,
        )

        self.is_loaded = True
        logger.info(
            f"MonitoringService initialized successfully with {self.ref_row_count:,} reference training rows."
        )

    def evaluate_telemetry_batch(
        self,
        raw_flows: List[Dict[str, Any]],
        batch_id: str = "api_batch_001",
        c8_bundle: Optional[Dict[str, Any]] = None,
        c10_bundle: Optional[Dict[str, Any]] = None,
    ) -> DriftMonitoringResponse:
        """Evaluate a batch of flow telemetry for drift, data quality, and adaptive policy."""
        t0 = time.perf_counter()
        if not self.is_loaded or self.detector is None or self.prediction_monitor is None:
            self.initialize_reference()

        df_batch = pd.DataFrame(raw_flows)

        # 1. Feature Drift & Data Quality Evaluation
        feature_summary: BatchMonitoringSummary = self.detector.evaluate_batch(
            batch_df=df_batch,
            batch_id=batch_id,
        )

        # 2. Prediction Drift Evaluation (if models are provided and flows are suitable)
        pred_result: Optional[PredictionDriftResult] = None
        c8_vs_c10_disagreement: Optional[float] = None

        if c8_bundle is not None:
            try:
                # Sanitized slice for inference if non-finites exist
                clean_df = df_batch.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                X_proc_c8, _, _, _ = extract_features_and_transform(c8_bundle, clean_df, schema="primary")
                preds_c8 = c8_bundle["model"].predict(X_proc_c8)
                probs_c8 = c8_bundle["model"].predict_proba(X_proc_c8)

                pred_result = self.prediction_monitor.evaluate_predictions(
                    y_pred=preds_c8,
                    y_prob=probs_c8,
                    model_id="C8_Primary",
                )

                if c10_bundle is not None:
                    X_proc_c10, _, _, _ = extract_features_and_transform(c10_bundle, clean_df, schema="alternative_no_sttl")
                    preds_c10 = c10_bundle["model"].predict(X_proc_c10)
                    disagreement = float(np.mean(preds_c8 != preds_c10))
                    c8_vs_c10_disagreement = round(disagreement, 4)
            except Exception as e:
                logger.warning(f"Could not compute prediction drift on batch: {e}")

        # 3. Rule-Based Adaptive Policy Evaluation
        policy_result: ComprehensiveAdaptivePolicyResult = self.adaptive_engine.evaluate_batch_signals(
            feature_summary=feature_summary,
            prediction_result=pred_result,
            c8_vs_c10_disagreement_rate=c8_vs_c10_disagreement,
        )

        # 4. Serialize Feature Summary
        num_summary = {
            col: {
                "psi": res.psi,
                "ks_statistic": res.ks_statistic,
                "ks_pvalue": res.ks_pvalue,
                "drift_level": res.drift_level,
                "missing_rate": res.missing_rate,
                "non_finite_rate": res.non_finite_rate,
            }
            for col, res in feature_summary.numerical_metrics.items()
        }
        cat_summary = {
            col: {
                "psi": res.psi,
                "drift_level": res.drift_level,
                "unknown_category_count": res.unknown_category_count,
                "unknown_category_rate": res.unknown_category_rate,
                "novel_categories": res.novel_categories,
            }
            for col, res in feature_summary.categorical_metrics.items()
        }

        # 5. Serialize Prediction Summary
        pred_summary_dict = None
        if pred_result is not None:
            pred_summary_dict = {
                "model_id": pred_result.model_id,
                "prediction_psi": pred_result.prediction_psi,
                "mean_confidence": pred_result.mean_confidence,
                "min_confidence": pred_result.min_confidence,
                "low_confidence_rate": pred_result.low_confidence_rate,
                "mean_entropy": pred_result.mean_entropy,
                "drift_level": pred_result.drift_level,
                "predicted_distribution": pred_result.predicted_distribution,
                "c8_c10_disagreement_rate": c8_vs_c10_disagreement,
            }

        decision_items = [
            DriftDecisionItem(
                action=d.action.value,
                priority=d.priority,
                trigger_reason=d.trigger_reason,
                evidence_metrics=d.evidence_metrics,
                operational_guidance=d.operational_guidance,
            )
            for d in policy_result.decisions
        ]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return DriftMonitoringResponse(
            batch_id=batch_id,
            batch_size=feature_summary.batch_size,
            is_valid_size=feature_summary.is_valid_size,
            overall_drift_status=feature_summary.overall_drift_status,
            drifting_features_count=feature_summary.drifting_features_count,
            critical_features_count=feature_summary.critical_features_count,
            primary_recommendation=policy_result.primary_recommendation.value,
            decisions=decision_items,
            requires_soc_escalation=policy_result.requires_soc_escalation,
            requires_pipeline_investigation=policy_result.requires_pipeline_investigation,
            feature_drift_summary={"numerical": num_summary, "categorical": cat_summary},
            prediction_drift_summary=pred_summary_dict,
            warnings=feature_summary.warnings,
            reference_dataset_info={
                "source": "train.csv",
                "sample_count": self.ref_row_count,
                "sealed_test_partition_status": "Strictly Untouched",
            },
            processing_time_ms=round(elapsed_ms, 2),
        )
