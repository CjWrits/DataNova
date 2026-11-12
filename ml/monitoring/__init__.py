"""Drift Detection, Monitoring & Adaptive Response Simulation Module."""

from ml.monitoring.adaptive_engine import (
    AdaptiveActionType,
    AdaptiveDecision,
    AdaptiveResponseEngine,
    ComprehensiveAdaptivePolicyResult,
)
from ml.monitoring.drift_detector import (
    BatchMonitoringSummary,
    CategoricalDriftResult,
    FeatureDriftDetector,
    ModelPredictionMonitor,
    NumericalDriftResult,
    PredictionDriftResult,
)
from ml.monitoring.simulation import (
    SimulationScenarioMetadata,
    generate_categorical_frequency_shift,
    generate_combined_multivariate_shift,
    generate_corrupted_missing_and_invalid_values,
    generate_novel_categories_injection,
    generate_numerical_mean_scale_shift,
    generate_stable_batch,
    generate_sttl_artifact_perturbation,
)
from ml.monitoring.visualizer import (
    plot_c8_vs_c10_sttl_sensitivity_comparison,
    plot_categorical_frequency_shift,
    plot_confidence_degradation_distribution,
    plot_feature_drift_psi_summary,
    plot_numerical_distribution_shift_overlay,
    plot_prediction_distribution_comparison,
)

__all__ = [
    "FeatureDriftDetector",
    "ModelPredictionMonitor",
    "NumericalDriftResult",
    "CategoricalDriftResult",
    "PredictionDriftResult",
    "BatchMonitoringSummary",
    "SimulationScenarioMetadata",
    "generate_stable_batch",
    "generate_numerical_mean_scale_shift",
    "generate_categorical_frequency_shift",
    "generate_novel_categories_injection",
    "generate_corrupted_missing_and_invalid_values",
    "generate_sttl_artifact_perturbation",
    "generate_combined_multivariate_shift",
    "AdaptiveActionType",
    "AdaptiveDecision",
    "ComprehensiveAdaptivePolicyResult",
    "AdaptiveResponseEngine",
    "plot_feature_drift_psi_summary",
    "plot_numerical_distribution_shift_overlay",
    "plot_categorical_frequency_shift",
    "plot_prediction_distribution_comparison",
    "plot_c8_vs_c10_sttl_sensitivity_comparison",
    "plot_confidence_degradation_distribution",
]
