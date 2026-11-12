"""Rule-Based Adaptive Response Decision Engine for Network Intrusion Monitoring.

Evaluates multi-source monitoring signals (feature distribution drift, novel categories,
data corruption, prediction entropy, confidence degradation, and model disagreement)
to recommend actionable, non-destructive operational interventions.

Strict Constraints:
- Operates purely as decision support; never triggers automatic retraining or silent replacement.
- Model weights, hyperparameters, thresholds, and serialized artifacts remain 100% immutable.
- Every adaptive response includes explicit trigger conditions, quantitative evidence, and operational rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ml.monitoring.drift_detector import BatchMonitoringSummary, PredictionDriftResult


class AdaptiveActionType(str, Enum):
    CONTINUE_MONITORING = "CONTINUE_MONITORING"
    INCREASE_MONITORING_FREQUENCY = "INCREASE_MONITORING_FREQUENCY"
    FLAG_AFFECTED_FEATURES = "FLAG_AFFECTED_FEATURES"
    RECOMMEND_ANALYST_REVIEW = "RECOMMEND_ANALYST_REVIEW"
    RECOMMEND_NO_STTL_FALLBACK = "RECOMMEND_NO_STTL_FALLBACK"
    RECOMMEND_DATA_COLLECTION = "RECOMMEND_DATA_COLLECTION"
    RECOMMEND_OFFLINE_EVALUATION = "RECOMMEND_OFFLINE_EVALUATION"
    RECOMMEND_RETRAINING_EXPERIMENT = "RECOMMEND_RETRAINING_EXPERIMENT"


@dataclass
class AdaptiveDecision:
    action: AdaptiveActionType
    priority: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    trigger_reason: str
    evidence_metrics: Dict[str, Any]
    operational_guidance: str


@dataclass
class ComprehensiveAdaptivePolicyResult:
    batch_id: str
    overall_status: str
    primary_recommendation: AdaptiveActionType
    decisions: List[AdaptiveDecision]
    requires_soc_escalation: bool
    requires_pipeline_investigation: bool


class AdaptiveResponseEngine:
    """Evaluates monitoring telemetry and produces structured operational guidance."""

    def __init__(
        self,
        psi_warning_th: float = 0.10,
        psi_critical_th: float = 0.25,
        confidence_drop_th: float = 0.65,
        low_conf_rate_th: float = 0.20,
        novel_category_rate_th: float = 0.02,
        missing_rate_th: float = 0.01,
    ) -> None:
        self.psi_warn_th = psi_warning_th
        self.psi_crit_th = psi_critical_th
        self.conf_drop_th = confidence_drop_th
        self.low_conf_th = low_conf_rate_th
        self.novel_cat_th = novel_category_rate_th
        self.missing_th = missing_rate_th

    def evaluate_batch_signals(
        self,
        feature_summary: BatchMonitoringSummary,
        prediction_result: Optional[PredictionDriftResult] = None,
        c8_vs_c10_disagreement_rate: Optional[float] = None,
    ) -> ComprehensiveAdaptivePolicyResult:
        """Apply deterministic, rule-based adaptive response logic."""
        decisions: List[AdaptiveDecision] = []

        # Rule 1: Small-batch safeguard check
        if not feature_summary.is_valid_size:
            decisions.append(
                AdaptiveDecision(
                    action=AdaptiveActionType.CONTINUE_MONITORING,
                    priority="LOW",
                    trigger_reason=f"Batch size ({feature_summary.batch_size}) below statistical safeguard threshold.",
                    evidence_metrics={"batch_size": feature_summary.batch_size},
                    operational_guidance="Accumulate additional telemetry flows before drawing drift inferences.",
                )
            )
            return ComprehensiveAdaptivePolicyResult(
                batch_id=feature_summary.batch_id,
                overall_status=feature_summary.overall_drift_status,
                primary_recommendation=AdaptiveActionType.CONTINUE_MONITORING,
                decisions=decisions,
                requires_soc_escalation=False,
                requires_pipeline_investigation=False,
            )

        # Rule 2: Sensor / Data Quality Corruption (Missing or Non-Finite values)
        corrupted_cols = []
        for col, res in feature_summary.numerical_metrics.items():
            if res.missing_rate >= self.missing_th or res.non_finite_rate >= self.missing_th:
                corrupted_cols.append({"col": col, "missing_rate": res.missing_rate, "non_finite_rate": res.non_finite_rate})

        if corrupted_cols:
            decisions.append(
                AdaptiveDecision(
                    action=AdaptiveActionType.FLAG_AFFECTED_FEATURES,
                    priority="HIGH",
                    trigger_reason="Data pipeline corruption detected with elevated missing or non-finite rates.",
                    evidence_metrics={"corrupted_features": corrupted_cols},
                    operational_guidance="Inspect upstream flow capture appliances and sensor telemetry parsers.",
                )
            )

        # Rule 3: Novel / Unseen Categorical Protocol or Service Injection
        novel_cat_cols = []
        for col, res_c in feature_summary.categorical_metrics.items():
            if res_c.unknown_category_rate >= self.novel_cat_th:
                novel_cat_cols.append({
                    "col": col,
                    "novel_rate": res_c.unknown_category_rate,
                    "novel_categories": res_c.novel_categories,
                })

        if novel_cat_cols:
            decisions.append(
                AdaptiveDecision(
                    action=AdaptiveActionType.RECOMMEND_DATA_COLLECTION,
                    priority="HIGH",
                    trigger_reason="New uncataloged protocols or services detected in network traffic.",
                    evidence_metrics={"novel_categories": novel_cat_cols},
                    operational_guidance="Collect and quarantine sample flows for protocol labeling and offline schema extension.",
                )
            )

        # Rule 4: STTL Artifact Divergence (Specific Trigger for No-STTL Fallback Recommendation)
        sttl_metric = feature_summary.numerical_metrics.get("sttl")
        sttl_is_divergent = False
        if sttl_metric:
            if sttl_metric.psi >= self.psi_crit_th:
                sttl_is_divergent = True
            elif sttl_metric.ks_statistic >= 0.30 and sttl_metric.ks_pvalue < 0.01:
                sttl_is_divergent = True
            elif (
                sttl_metric.drift_level in ("MODERATE_DRIFT", "CRITICAL_DRIFT")
                and (c8_vs_c10_disagreement_rate is not None and c8_vs_c10_disagreement_rate >= 0.15)
            ):
                sttl_is_divergent = True

        if sttl_is_divergent and sttl_metric:
            decisions.append(
                AdaptiveDecision(
                    action=AdaptiveActionType.RECOMMEND_NO_STTL_FALLBACK,
                    priority="HIGH",
                    trigger_reason=(
                        f"Severe divergence detected on synthetic testbed artifact `sttl` "
                        f"(PSI={sttl_metric.psi:.3f}, KS={sttl_metric.ks_statistic:.3f}, p={sttl_metric.ks_pvalue:.2e})."
                    ),
                    evidence_metrics={
                        "sttl_psi": sttl_metric.psi,
                        "sttl_ks_stat": sttl_metric.ks_statistic,
                        "sttl_ks_pvalue": sttl_metric.ks_pvalue,
                        "c8_c10_disagreement": c8_vs_c10_disagreement_rate,
                    },
                    operational_guidance="Evaluate parallel inference against the No-STTL model (Candidate 10) to mitigate synthetic TTL sensitivity.",
                )
            )

        # Rule 5: Severe Multivariate Distribution Drift
        if feature_summary.critical_features_count >= 3:
            crit_names = [k for k, v in feature_summary.numerical_metrics.items() if v.drift_level == "CRITICAL_DRIFT"]
            decisions.append(
                AdaptiveDecision(
                    action=AdaptiveActionType.RECOMMEND_OFFLINE_EVALUATION,
                    priority="CRITICAL",
                    trigger_reason=f"Systemic multivariate drift across {len(crit_names)} critical network features.",
                    evidence_metrics={"critical_features": crit_names},
                    operational_guidance="Initiate controlled offline evaluation against recent labeled captures; do not alter live models.",
                )
            )

        # Rule 6: Model Prediction Confidence Collapse & Entropy Spike
        if prediction_result:
            if prediction_result.mean_confidence < self.conf_drop_th or prediction_result.low_confidence_rate >= self.low_conf_th:
                decisions.append(
                    AdaptiveDecision(
                        action=AdaptiveActionType.RECOMMEND_ANALYST_REVIEW,
                        priority="HIGH",
                        trigger_reason="Prediction confidence degraded below operational reliability threshold.",
                        evidence_metrics={
                            "mean_confidence": prediction_result.mean_confidence,
                            "low_confidence_rate": prediction_result.low_confidence_rate,
                            "prediction_entropy": prediction_result.mean_entropy,
                        },
                        operational_guidance="Escalate low-confidence flow predictions to SOC tier-2 analysts for manual verification.",
                    )
                )

        # Rule 7: Moderate Drifting Features without Critical Collapse
        if 1 <= feature_summary.drifting_features_count < 3 and not decisions:
            mod_names = [k for k, v in feature_summary.numerical_metrics.items() if v.drift_level == "MODERATE_DRIFT"]
            decisions.append(
                AdaptiveDecision(
                    action=AdaptiveActionType.INCREASE_MONITORING_FREQUENCY,
                    priority="MEDIUM",
                    trigger_reason=f"Moderate drift identified on {len(mod_names)} features.",
                    evidence_metrics={"drifting_features": mod_names},
                    operational_guidance="Shorten monitoring window from hourly to 15-minute intervals to observe trajectory.",
                )
            )

        # Default: Everything Stable
        if not decisions:
            decisions.append(
                AdaptiveDecision(
                    action=AdaptiveActionType.CONTINUE_MONITORING,
                    priority="LOW",
                    trigger_reason="Telemetry and model inference metrics remain within baseline bounds.",
                    evidence_metrics={"status": "STABLE"},
                    operational_guidance="Maintain standard passive telemetry observation cadence.",
                )
            )

        # Determine primary recommendation (highest priority decision)
        priority_map = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        primary_dec = max(decisions, key=lambda d: priority_map.get(d.priority, 0))

        requires_soc = any(d.action in (AdaptiveActionType.RECOMMEND_ANALYST_REVIEW, AdaptiveActionType.RECOMMEND_NO_STTL_FALLBACK) for d in decisions)
        requires_pipe = any(d.action in (AdaptiveActionType.FLAG_AFFECTED_FEATURES, AdaptiveActionType.RECOMMEND_DATA_COLLECTION) for d in decisions)

        return ComprehensiveAdaptivePolicyResult(
            batch_id=feature_summary.batch_id,
            overall_status=feature_summary.overall_drift_status,
            primary_recommendation=primary_dec.action,
            decisions=decisions,
            requires_soc_escalation=requires_soc,
            requires_pipeline_investigation=requires_pipe,
        )
