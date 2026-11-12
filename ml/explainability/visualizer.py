"""Publication-Grade Visualization Generator for Phase 5 Explainability and Diagnostics.

Generates diagnostic plots saved under research/visualizations/phase5/:
1. Global feature importance comparisons (Phase 3 Baseline vs Phase 4 C8 Winner vs Phase 4 C10 No-STTL).
2. Class-specific SHAP attribution rankings for all 10 classes (especially Normal, Fuzzers, DoS, Analysis, Worms).
3. False-Positive attribution breakdown (benign flows misclassified as Fuzzers/Exploits).
4. False-Negative attribution breakdown (missed attacks misclassified as Normal).
5. 10 Engineered features comparison chart (Primary C8 vs No-STTL C10).
6. Multi-model Beeswarm plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from ml.explainability.shap_explainer import (
    ExplanationArtifact,
    compute_class_specific_importance,
    compute_global_feature_importance,
)


def plot_global_feature_importance_comparison(
    artifact_p3: ExplanationArtifact,
    artifact_c8: ExplanationArtifact,
    artifact_c10: ExplanationArtifact,
    output_path: Path,
    top_n: int = 15,
) -> None:
    """Compare global aggregated feature importances across Phase 3 Baseline, C8 Winner, and C10 No-STTL."""
    imp_p3 = compute_global_feature_importance(artifact_p3, aggregate_to_original=True).set_index("feature")
    imp_c8 = compute_global_feature_importance(artifact_c8, aggregate_to_original=True).set_index("feature")
    imp_c10 = compute_global_feature_importance(artifact_c10, aggregate_to_original=True).set_index("feature")

    # Select top features based on Candidate 8
    top_features = list(imp_c8.head(top_n).index)

    df_plot = pd.DataFrame({
        "Phase 3 Baseline (Unweighted)": [imp_p3.loc[f, "mean_abs_shap"] if f in imp_p3.index else 0.0 for f in top_features],
        "Phase 4 C8 (FE + Sqrt Winner)": [imp_c8.loc[f, "mean_abs_shap"] if f in imp_c8.index else 0.0 for f in top_features],
        "Phase 4 C10 (No-STTL Robustness)": [imp_c10.loc[f, "mean_abs_shap"] if f in imp_c10.index else 0.0 for f in top_features],
    }, index=top_features)

    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid", font="sans-serif")
    df_plot.plot(kind="barh", figsize=(12, 8), width=0.8, color=["#7f7f7f", "#1f77b4", "#2ca02c"])

    plt.title(f"Global Feature Attribution Comparison (Top {top_n} Features)\nPhase 3 Baseline vs. Phase 4 C8 Winner vs. Phase 4 C10 No-STTL", fontsize=13, fontweight="bold")
    plt.xlabel("Mean Absolute SHAP Attribution (log scale or raw impact)", fontsize=11)
    plt.ylabel("Network Flow Feature (Aggregated)", fontsize=11)
    plt.gca().invert_yaxis()
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_class_specific_attributions(
    artifact: ExplanationArtifact,
    class_idx: int,
    output_path: Path,
    top_n: int = 12,
) -> None:
    """Plot top positive and negative feature attributions for a designated class."""
    cname = artifact.class_names[class_idx]
    df_class = compute_class_specific_importance(artifact, class_idx=class_idx, aggregate_to_original=True)
    top_df = df_class.head(top_n).copy()

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid", font="sans-serif")
    colors = ["#1f77b4" if s >= 0 else "#d62728" for s in top_df["mean_signed_shap"]]

    bars = plt.barh(top_df["feature"], top_df["mean_abs_shap"], color=colors, alpha=0.85)
    plt.title(f"Class Attribution Profile: '{cname}'\nModel: {artifact.model_name}", fontsize=13, fontweight="bold")
    plt.xlabel("Mean Absolute SHAP Value (Blue = Positive Pull, Red = Negative Pull)", fontsize=10)
    plt.ylabel("Network Feature", fontsize=10)
    plt.gca().invert_yaxis()

    for idx, (bar, row) in enumerate(zip(bars, top_df.itertuples())):
        sign_str = "+" if row.mean_signed_shap >= 0 else "-"
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f"{sign_str}{row.mean_abs_shap:.3f} ({row.relative_pct:.1f}%)", va="center", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_error_attribution_drivers(
    error_dict: Dict[str, Any],
    title: str,
    output_path: Path,
) -> None:
    """Plot features pushing predictions towards the erroneous class."""
    drivers = error_dict.get("drivers_pushing_towards_fuzzers") or error_dict.get("drivers_pushing_towards_normal") or error_dict.get("drivers_pushing_towards_exploits") or []
    if not drivers:
        return

    features = [d["feature"] for d in drivers[:10]]
    values = [d["mean_shap_pull"] for d in drivers[:10]]

    plt.figure(figsize=(10, 5))
    sns.set_theme(style="whitegrid", font="sans-serif")
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in values]

    bars = plt.barh(features, values, color=colors, alpha=0.85)
    plt.title(f"Diagnostic Error Analysis: {title}\n(Sample Count: {error_dict.get('count', 0):,})", fontsize=12, fontweight="bold")
    plt.xlabel("Mean SHAP Attribution Pull (Positive = Pushing Towards Erroneous Class)", fontsize=10)
    plt.ylabel("Feature", fontsize=10)
    plt.gca().invert_yaxis()
    plt.axvline(0, color="black", linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_engineered_features_audit(
    df_audit: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot comparative contribution of the 10 domain engineered flow features."""
    plt.figure(figsize=(11, 6))
    sns.set_theme(style="whitegrid", font="sans-serif")

    x = np.arange(len(df_audit))
    width = 0.35

    plt.bar(x - width/2, df_audit["c8_mean_abs_shap"], width, label="Candidate 8 (Primary w/ STTL)", color="#1f77b4", alpha=0.85)
    plt.bar(x + width/2, df_audit["c10_mean_abs_shap"], width, label="Candidate 10 (Robustness No-STTL)", color="#2ca02c", alpha=0.85)

    plt.xticks(x, [f.replace("feat_", "") for f in df_audit["feature"]], rotation=35, ha="right", fontsize=9)
    plt.ylabel("Mean Absolute SHAP Value", fontsize=10)
    plt.title("Domain-Engineered Flow Features: Attribution Audit & Model Stability", fontsize=12, fontweight="bold")
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_beeswarm_summary(
    artifact: ExplanationArtifact,
    output_path: Path,
    class_idx: int = 3,  # Fuzzers by default
    max_display: int = 15,
) -> None:
    """Generate SHAP beeswarm summary plot for a specific target class."""
    sv = artifact.shap_values
    if sv.ndim == 3:
        class_sv = sv[:, :, class_idx]
    else:
        class_sv = sv

    plt.figure(figsize=(12, 7))
    shap.summary_plot(
        class_sv,
        artifact.X_processed,
        feature_names=artifact.transformed_feature_names,
        max_display=max_display,
        show=False,
    )
    cname = artifact.class_names[class_idx]
    plt.title(f"SHAP Attribution Beeswarm Plot for Class '{cname}'\nModel: {artifact.model_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
