"""UNSW-NB15 Reproducible Exploratory Data Analysis (EDA) Script.

Performs comprehensive empirical data analysis on the acquired UNSW-NB15
raw benchmark dataset, computes distributional properties, detects leakage risks,
and generates publication-quality visualization figures.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

logger = logging.getLogger("eda")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_datasets(raw_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the acquired benchmark CSV files and metadata."""
    train_file = raw_dir / "UNSW_NB15_training-set.csv"
    test_file = raw_dir / "UNSW_NB15_testing-set.csv"
    feat_file = raw_dir / "NUSW-NB15_features.csv"
    event_file = raw_dir / "UNSW-NB15_LIST_EVENTS.csv"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(f"Missing benchmark CSVs in {raw_dir}. Run scripts/download_unsw_nb15.py first.")

    logger.info("Loading training set: %s", train_file)
    df_train = pd.read_csv(train_file)
    logger.info("Loading testing set:  %s", test_file)
    df_test = pd.read_csv(test_file)

    df_feats = pd.read_csv(feat_file, encoding="latin1") if feat_file.exists() else pd.DataFrame()
    df_events = pd.read_csv(event_file) if event_file.exists() else pd.DataFrame()

    return df_train, df_test, df_feats, df_events


def compute_dataset_structure(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict[str, Any]:
    """Compute dimensional, type, missing, and duplicate properties."""
    structure = {
        "train_rows": len(df_train),
        "train_cols": len(df_train.columns),
        "test_rows": len(df_test),
        "test_cols": len(df_test.columns),
        "total_rows": len(df_train) + len(df_test),
        "columns_match": list(df_train.columns) == list(df_test.columns),
        "train_missing_cells": int(df_train.isnull().sum().sum()),
        "test_missing_cells": int(df_test.isnull().sum().sum()),
        "train_duplicates_with_id": int(df_train.duplicated().sum()),
        "test_duplicates_with_id": int(df_test.duplicated().sum()),
        "train_duplicates_no_id": int(df_train.drop(columns=["id"]).duplicated().sum()),
        "test_duplicates_no_id": int(df_test.drop(columns=["id"]).duplicated().sum()),
    }

    # Cross-dataset duplication/overlap
    train_no_id = df_train.drop(columns=["id"])
    test_no_id = df_test.drop(columns=["id"])
    overlap_count = int(len(pd.merge(train_no_id, test_no_id, how="inner")))
    structure["train_test_overlap_exact_rows"] = overlap_count
    structure["train_test_overlap_pct_of_train"] = round((overlap_count / len(df_train)) * 100, 2)

    feature_cols = [c for c in df_train.columns if c not in ["id", "label", "attack_cat"]]
    train_feats = df_train[feature_cols]
    test_feats = df_test[feature_cols]
    overlap_feats_count = int(len(pd.merge(train_feats, test_feats, how="inner")))
    structure["train_test_feature_overlap_rows"] = overlap_feats_count

    return structure


def compute_class_distributions(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict[str, Any]:
    """Compute binary and multiclass counts and percentages."""
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    def extract_dist(df: pd.DataFrame, col: str) -> Dict[str, Dict[str, Any]]:
        counts = df[col].value_counts()
        pcts = df[col].value_counts(normalize=True) * 100
        return {
            str(k): {"count": int(counts[k]), "percentage": round(float(pcts[k]), 4)}
            for k in counts.index
        }

    return {
        "binary_train": extract_dist(df_train, "label"),
        "binary_test": extract_dist(df_test, "label"),
        "binary_combined": extract_dist(df_all, "label"),
        "multiclass_train": extract_dist(df_train, "attack_cat"),
        "multiclass_test": extract_dist(df_test, "attack_cat"),
        "multiclass_combined": extract_dist(df_all, "attack_cat"),
    }


def compute_feature_statistics(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict[str, Any]:
    """Compute correlation, collinearity, and categorical differences."""
    num_cols = [c for c in df_train.select_dtypes(include=[np.number]).columns if c not in ["id", "label"]]
    cat_cols = df_train.select_dtypes(include=["object"]).columns.tolist()

    # Numerical correlations with label
    corr_train = df_train[num_cols].apply(lambda s: s.corr(df_train["label"])).to_dict()
    corr_test = df_test[num_cols].apply(lambda s: s.corr(df_test["label"])).to_dict()

    # Identify collinear pairs (> 0.90)
    corr_mat = df_train[num_cols].corr().abs()
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    collinear_pairs = []
    for col in upper.columns:
        for row in upper.index:
            v = upper.loc[row, col]
            if v > 0.90:
                collinear_pairs.append({
                    "feature_1": row,
                    "feature_2": col,
                    "correlation": round(float(v), 4),
                })
    collinear_pairs.sort(key=lambda x: x["correlation"], reverse=True)

    # Categorical domain drift between train and test
    cat_diffs = {}
    for c in cat_cols:
        tr_u = set(df_train[c].dropna().unique())
        te_u = set(df_test[c].dropna().unique())
        cat_diffs[c] = {
            "train_unique": len(tr_u),
            "test_unique": len(te_u),
            "in_test_not_train": sorted(list(te_u - tr_u)),
            "in_train_not_test": sorted(list(tr_u - te_u)),
        }

    # ID target correlation (Leakage risk)
    id_leakage = {
        "train_id_label_corr": round(float(df_train["id"].corr(df_train["label"])), 4),
        "test_id_label_corr": round(float(df_test["id"].corr(df_test["label"])), 4),
    }

    return {
        "num_features_count": len(num_cols),
        "cat_features_count": len(cat_cols),
        "correlations_train": corr_train,
        "correlations_test": corr_test,
        "collinear_pairs": collinear_pairs,
        "categorical_differences": cat_diffs,
        "id_leakage": id_leakage,
    }


def generate_visualizations(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    vis_dir: Path,
) -> List[Path]:
    """Generate and save publication-grade analysis plots."""
    vis_dir.mkdir(parents=True, exist_ok=True)
    generated_plots: List[Path] = []

    # 1. Binary Class Imbalance Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["Train (82k)", "Test (175k)", "Combined (258k)"]
    normal_pcts = [
        (df_train["label"] == 0).mean() * 100,
        (df_test["label"] == 0).mean() * 100,
        ((df_train["label"] == 0).sum() + (df_test["label"] == 0).sum()) / (len(df_train) + len(df_test)) * 100,
    ]
    attack_pcts = [
        (df_train["label"] == 1).mean() * 100,
        (df_test["label"] == 1).mean() * 100,
        ((df_train["label"] == 1).sum() + (df_test["label"] == 1).sum()) / (len(df_train) + len(df_test)) * 100,
    ]
    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax.bar(x - width/2, normal_pcts, width, label="Normal / Benign (0)", color="#2b5c8f")
    rects2 = ax.bar(x + width/2, attack_pcts, width, label="Attack / Malicious (1)", color="#d9534f")

    ax.set_ylabel("Class Proportion (%)")
    ax.set_title("UNSW-NB15 Binary Class Proportions Across Partitions")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(frameon=True)
    ax.set_ylim(0, 100)

    for rect in rects1 + rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    p1 = vis_dir / "class_imbalance_binary.png"
    fig.savefig(p1)
    plt.close(fig)
    generated_plots.append(p1)

    # 2. Multiclass Attack Category Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    order = df_all["attack_cat"].value_counts().index.tolist()

    df_train_counts = df_train["attack_cat"].value_counts().reindex(order, fill_value=0)
    df_test_counts = df_test["attack_cat"].value_counts().reindex(order, fill_value=0)

    y = np.arange(len(order))
    h = 0.38
    ax.barh(y - h/2, df_train_counts.values, h, label="Training Set (82,332)", color="#337ab7")
    ax.barh(y + h/2, df_test_counts.values, h, label="Testing Set (175,341)", color="#e67e22")

    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("Flow Count")
    ax.set_title("UNSW-NB15 Attack Category Counts (Train vs. Test)")
    ax.legend(frameon=True)

    # Format x-axis with comma separators
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, loc: f"{int(val):,}"))
    plt.tight_layout()
    p2 = vis_dir / "attack_category_distribution.png"
    fig.savefig(p2)
    plt.close(fig)
    generated_plots.append(p2)

    # 3. Minority Class Deep-Dive (Rare Attack Families)
    minority_classes = ["Analysis", "Backdoor", "Shellcode", "Worms"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    tr_min = df_train["attack_cat"].value_counts().reindex(minority_classes, fill_value=0)
    te_min = df_test["attack_cat"].value_counts().reindex(minority_classes, fill_value=0)

    x_min = np.arange(len(minority_classes))
    ax.bar(x_min - width/2, tr_min.values, width, label="Train (82,332)", color="#17a2b8")
    ax.bar(x_min + width/2, te_min.values, width, label="Test (175,341)", color="#dc3545")

    ax.set_ylabel("Raw Instance Count")
    ax.set_title("Extreme Minority Attack Categories (< 1% Frequency)")
    ax.set_xticks(x_min)
    ax.set_xticklabels(minority_classes)
    ax.legend(frameon=True)

    for i, (tr_v, te_v) in enumerate(zip(tr_min.values, te_min.values)):
        ax.annotate(f"{tr_v:,}", (x_min[i] - width/2, tr_v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
        ax.annotate(f"{te_v:,}", (x_min[i] + width/2, te_v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    p3 = vis_dir / "minority_class_breakdown.png"
    fig.savefig(p3)
    plt.close(fig)
    generated_plots.append(p3)

    # 4. Correlation Heatmap of High-Impact Features
    key_features = [
        "sttl", "ct_state_ttl", "ct_dst_sport_ltm", "rate", "dload", "swin",
        "dwin", "sbytes", "dbytes", "sloss", "dloss", "ct_srv_src", "label"
    ]
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_subset = df_train[key_features].corr()
    sns.heatmap(corr_subset, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax, cbar_kws={"label": "Pearson Correlation"})
    ax.set_title("Correlation Matrix of Key Flow Features and Target Label (Train)")
    plt.tight_layout()
    p4 = vis_dir / "feature_correlation_heatmap.png"
    fig.savefig(p4)
    plt.close(fig)
    generated_plots.append(p4)

    # 5. Synthetic Artifact Analysis: Source TTL (sttl) Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    df_train[df_train["label"] == 0]["sttl"].hist(bins=50, ax=ax1, color="#2b5c8f", alpha=0.75, density=True)
    ax1.set_title("Benign / Normal Traffic (sttl)")
    ax1.set_xlabel("Source Time-to-Live (sttl)")
    ax1.set_ylabel("Empirical Density")
    ax1.set_xlim(0, 260)

    df_train[df_train["label"] == 1]["sttl"].hist(bins=50, ax=ax2, color="#d9534f", alpha=0.75, density=True)
    ax2.set_title("Malicious Attack Traffic (sttl)")
    ax2.set_xlabel("Source Time-to-Live (sttl)")
    ax2.set_xlim(0, 260)

    fig.suptitle("Synthetic Artifact Discovery: Source TTL Bimodal Separation", y=1.02)
    plt.tight_layout()
    p5 = vis_dir / "synthetic_artifact_leakage_sttl.png"
    fig.savefig(p5)
    plt.close(fig)
    generated_plots.append(p5)

    # 6. ID Sequence Leakage Plot
    fig, ax = plt.subplots(figsize=(10, 4.5))
    window = 1000
    roll_train = df_train["label"].rolling(window=window).mean()
    roll_test = df_test["label"].rolling(window=window).mean()

    ax.plot(df_train["id"], roll_train, label="Train Set (Rolling Mean, window=1000)", color="#337ab7", linewidth=1.2)
    ax.plot(df_test["id"], roll_test, label="Test Set (Rolling Mean, window=1000)", color="#d9534f", linewidth=1.2, linestyle="--")

    ax.set_xlabel("Row Sequence / ID Index")
    ax.set_ylabel("Rolling Attack Proportion")
    ax.set_title("Target Leakage Risk: Sequential Block Structure of 'id' vs Label")
    ax.legend(frameon=True, loc="center right")
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    p6 = vis_dir / "id_target_leakage.png"
    fig.savefig(p6)
    plt.close(fig)
    generated_plots.append(p6)

    logger.info("Successfully saved 6 visualization plots to %s", vis_dir)
    return generated_plots


def run_full_eda(
    raw_dir: Path,
    vis_dir: Path,
    report_json_path: Path | None = None,
) -> Dict[str, Any]:
    """Execute complete EDA pipeline and aggregate findings."""
    df_train, df_test, df_feats, df_events = load_datasets(raw_dir)

    logger.info("Computing dataset structure metrics...")
    structure = compute_dataset_structure(df_train, df_test)

    logger.info("Computing class distributions...")
    classes = compute_class_distributions(df_train, df_test)

    logger.info("Computing feature statistics and leakage indicators...")
    features = compute_feature_statistics(df_train, df_test)

    logger.info("Generating publication visualizations...")
    plots = generate_visualizations(df_train, df_test, vis_dir)

    results = {
        "dataset_structure": structure,
        "class_distributions": classes,
        "feature_statistics": features,
        "generated_visualizations": [str(p) for p in plots],
    }

    if report_json_path:
        report_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved EDA JSON summary to %s", report_json_path)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UNSW-NB15 Exploratory Data Analysis.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/unsw_nb15"))
    parser.add_argument("--vis-dir", type=Path, default=Path("research/visualizations"))
    parser.add_argument("--output-json", type=Path, default=Path("research/experiments/unsw_nb15_eda_summary.json"))
    args = parser.parse_args()

    results = run_full_eda(args.raw_dir, args.vis_dir, args.output_json)
    print("\nEDA Execution Complete!")
    print(f"Train Rows: {results['dataset_structure']['train_rows']:,}")
    print(f"Test Rows:  {results['dataset_structure']['test_rows']:,}")
    print(f"Overlap:    {results['dataset_structure']['train_test_overlap_exact_rows']:,} rows")
