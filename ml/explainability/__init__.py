"""Explainability engines including SHAP tree explainers and local attributions."""

from ml.explainability.shap_explainer import (
    ExplanationArtifact,
    audit_engineered_features_importance,
    compute_class_specific_importance,
    compute_global_feature_importance,
    compute_shap_explanations,
    extract_error_subsets,
    extract_features_and_transform,
)
from ml.explainability.visualizer import (
    plot_beeswarm_summary,
    plot_class_specific_attributions,
    plot_engineered_features_audit,
    plot_error_attribution_drivers,
    plot_global_feature_importance_comparison,
)

__all__ = [
    "ExplanationArtifact",
    "extract_features_and_transform",
    "compute_shap_explanations",
    "compute_global_feature_importance",
    "compute_class_specific_importance",
    "extract_error_subsets",
    "audit_engineered_features_importance",
    "plot_global_feature_importance_comparison",
    "plot_class_specific_attributions",
    "plot_error_attribution_drivers",
    "plot_engineered_features_audit",
    "plot_beeswarm_summary",
]
