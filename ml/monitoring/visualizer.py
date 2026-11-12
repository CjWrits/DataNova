"""Publication-Grade Visualization Generator for Drift Monitoring and Adaptive Intelligence.

Saves diagnostic visualization figures to research/visualizations/phase6/:
1. Feature Drift PSI Summary Bar Chart (highlighting Moderate vs Critical thresholds).
2. Numerical Distribution Shift Overlays (Reference vs Drifted Batches).
3. Categorical Protocol and Service Frequency Skew Plots.
4. Model Prediction Distribution Divergence across Classes.
5. Prediction Confidence Degradation Density Curves.
6. C8 vs C10 Prediction Disagreement and Resilience under STTL Perturbation.
7. Drift Severity Overview & Adaptive Policy Matrix.
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

from ml.monitoring.drift_detector import BatchMonitoringSummary, PredictionDriftResult


def plot_feature_drift_psi_summary(
    summary: BatchMonitoringSummary,
    output_path: Path,
    top_n: int = 15,
) -> None:
    """Plot descending Population Stability Index (PSI) across monitored features with threshold bands."""
    items = []
    for col, res in summary.numerical_metrics.items():
        items.append({"feature": col, "psi": res.psi, "type": "numerical", "level": res.drift_level})
    for col, res in summary.categorical_metrics.items():
        items.append({"feature": col, "psi": res.psi, "type": "categorical", "level": res.drift_level})

    df_plot = pd.DataFrame(items).sort_values(by="psi", ascending=False).head(top_n)

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid", font="sans-serif")

    colors = [
        "#d62728" if lvl == "CRITICAL_DRIFT" else ("#ff7f0e" if lvl == "MODERATE_DRIFT" else "#2ca02c")
        for lvl in df_plot["level"]
    ]

    bars = plt.barh(df_plot["feature"], df_plot["psi"], color=colors, alpha=0.85)

    plt.axvline(0.10, color="#ff7f0e", linestyle="--", lw=1.5, label="Moderate Drift Threshold (PSI = 0.10)")
    plt.axvline(0.25, color="#d62728", linestyle="-.", lw=1.5, label="Critical Drift Threshold (PSI = 0.25)")

    plt.title(f"Feature Population Stability Index (PSI) Summary\nBatch: '{summary.batch_id}' (Status: {summary.overall_drift_status})", fontsize=12, fontweight="bold")
    plt.xlabel("Population Stability Index (PSI)", fontsize=10)
    plt.ylabel("Network Flow Feature", fontsize=10)
    plt.gca().invert_yaxis()
    plt.legend(frameon=True, fontsize=9, loc="lower right")

    for bar, psi_val in zip(bars, df_plot["psi"]):
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f"{psi_val:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_numerical_distribution_shift_overlay(
    ref_series: pd.Series,
    batch_series: pd.Series,
    feature_name: str,
    output_path: Path,
    log_scale: bool = False,
) -> None:
    """Plot KDE and histogram distribution overlay comparing reference vs drifted batch."""
    plt.figure(figsize=(10, 5))
    sns.set_theme(style="whitegrid", font="sans-serif")

    clean_ref = ref_series.dropna().values.astype(float)
    clean_batch = batch_series.dropna().values.astype(float)

    clean_ref = clean_ref[np.isfinite(clean_ref)]
    clean_batch = clean_batch[np.isfinite(clean_batch)]

    if log_scale:
        clean_ref = np.log1p(np.maximum(0, clean_ref))
        clean_batch = np.log1p(np.maximum(0, clean_batch))
        xlabel_suffix = " (log1p scale)"
    else:
        xlabel_suffix = ""

    sns.kdeplot(clean_ref, label="Reference Distribution (Train)", color="#1f77b4", lw=2, fill=True, alpha=0.25)
    sns.kdeplot(clean_batch, label="Monitored Batch (Simulation)", color="#d62728", lw=2, fill=True, alpha=0.25)

    plt.title(f"Numerical Distribution Divergence: `{feature_name}`", fontsize=12, fontweight="bold")
    plt.xlabel(f"{feature_name}{xlabel_suffix}", fontsize=10)
    plt.ylabel("Probability Density", fontsize=10)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_categorical_frequency_shift(
    ref_series: pd.Series,
    batch_series: pd.Series,
    feature_name: str,
    output_path: Path,
    top_n: int = 10,
) -> None:
    """Plot comparative category frequencies and highlight novel unseen categories."""
    ref_counts = ref_series.astype(str).value_counts(normalize=True)
    batch_counts = batch_series.astype(str).value_counts(normalize=True)

    top_cats = list(batch_counts.head(top_n).index)
    known_cats = set(ref_counts.index)

    df_plot = pd.DataFrame({
        "Reference Proportion": [ref_counts.get(c, 0.0) for c in top_cats],
        "Monitored Batch Proportion": [batch_counts.get(c, 0.0) for c in top_cats],
    }, index=top_cats)

    plt.figure(figsize=(11, 5))
    sns.set_theme(style="whitegrid", font="sans-serif")
    df_plot.plot(kind="bar", figsize=(11, 5), width=0.75, color=["#1f77b4", "#ff7f0e"])

    plt.title(f"Categorical Frequency Divergence: `{feature_name}`", fontsize=12, fontweight="bold")
    plt.xlabel("Category Value", fontsize=10)
    plt.ylabel("Normalized Frequency", fontsize=10)
    plt.xticks(rotation=30, ha="right")
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_prediction_distribution_comparison(
    ref_dist: Dict[str, float],
    batch_dist: Dict[str, float],
    output_path: Path,
) -> None:
    """Plot predicted class distribution shift across all 10 classes."""
    classes = list(ref_dist.keys())
    df_pred = pd.DataFrame({
        "Reference Prediction Distribution": [ref_dist.get(c, 0.0) for c in classes],
        "Drifted Batch Prediction Distribution": [batch_dist.get(c, 0.0) for c in classes],
    }, index=classes)

    plt.figure(figsize=(12, 5))
    sns.set_theme(style="whitegrid", font="sans-serif")
    df_pred.plot(kind="bar", figsize=(12, 5), width=0.75, color=["#7f7f7f", "#d62728"])

    plt.title("Model Prediction Distribution Divergence (Prediction Drift)", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Attack Category", fontsize=10)
    plt.ylabel("Proportion of Total Predictions", fontsize=10)
    plt.xticks(rotation=35, ha="right")
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_c8_vs_c10_sttl_sensitivity_comparison(
    c8_preds_ref: np.ndarray,
    c8_preds_drift: np.ndarray,
    c10_preds_ref: np.ndarray,
    c10_preds_drift: np.ndarray,
    output_path: Path,
) -> None:
    """Compare prediction shift between Candidate 8 (with STTL) and Candidate 10 (No-STTL) under STTL perturbation."""
    c8_shift_rate = float(np.mean(c8_preds_ref != c8_preds_drift)) * 100.0
    c10_shift_rate = float(np.mean(c10_preds_ref != c10_preds_drift)) * 100.0

    plt.figure(figsize=(8, 5))
    sns.set_theme(style="whitegrid", font="sans-serif")

    models = ["Candidate 8\n(Primary with STTL)", "Candidate 10\n(Robustness No-STTL)"]
    rates = [c8_shift_rate, c10_shift_rate]
    colors = ["#d62728", "#2ca02c"]

    bars = plt.bar(models, rates, color=colors, width=0.5, alpha=0.85)
    plt.title("Model Sensitivity to Adversarial / Synthetic STTL Perturbation\n(% of Predictions Altered Solely by Modifying STTL)", fontsize=12, fontweight="bold")
    plt.ylabel("Prediction Alteration Rate (%)", fontsize=10)
    plt.ylim([0, max(rates) * 1.25 if max(rates) > 0 else 10])

    for bar, val in zip(bars, rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confidence_degradation_distribution(
    ref_conf: np.ndarray,
    batch_conf: np.ndarray,
    output_path: Path,
) -> None:
    """Plot distribution of maximum predicted probability comparing reference to degraded batch."""
    plt.figure(figsize=(10, 5))
    sns.set_theme(style="whitegrid", font="sans-serif")

    sns.kdeplot(ref_conf, label="Reference In-Distribution Confidence", color="#2ca02c", lw=2, fill=True, alpha=0.25)
    sns.kdeplot(batch_conf, label="Drifted Batch Degraded Confidence", color="#d62728", lw=2, fill=True, alpha=0.25)

    plt.axvline(0.60, color="black", linestyle="--", lw=1.5, label="Low-Confidence Threshold (0.60)")

    plt.title("Model Prediction Confidence Degradation Density", fontsize=12, fontweight="bold")
    plt.xlabel("Maximum Predicted Class Probability $\\max_c P(y=c \\mid x)$", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.xlim([0.0, 1.05])
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
