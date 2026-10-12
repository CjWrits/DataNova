"""Group-aware stratified dataset partitioning module.

Constructs reproducible, group-aware evaluation splits (Train ~70%, Val ~15%, Test ~15%)
ensuring that identical feature vectors never cross partition boundaries.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from ml.preprocessing.duplicate_audit import compute_feature_hashes
from ml.preprocessing.schema import (
    PRIMARY_FEATURE_COLUMNS,
    TARGET_BINARY,
    TARGET_MULTICLASS,
)

logger = logging.getLogger("split")


def partition_groups_stratified(
    group_df: pd.DataFrame,
    target_col: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, str]:
    """Assign feature group hashes to train, val, or test partitions using stratified group allocation."""
    assert abs(train_ratio + val_ratio + (1.0 - train_ratio - val_ratio) - 1.0) < 1e-5

    rng = np.random.RandomState(random_seed)
    group_to_partition: Dict[str, str] = {}

    unique_classes = group_df[target_col].unique()

    for cls in unique_classes:
        cls_groups = group_df[group_df[target_col] == cls].copy()
        shuffled_indices = rng.permutation(len(cls_groups))
        cls_groups = cls_groups.iloc[shuffled_indices]

        n = len(cls_groups)
        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))

        # Ensure representation for rare classes if count >= 3
        if n >= 3:
            n_train = max(1, min(n - 2, n_train))
            n_val = max(1, min(n - n_train - 1, n_val))
        elif n == 2:
            n_train = 1
            n_val = 1
        else:
            n_train = 1
            n_val = 0

        n_test = n - n_train - n_val

        train_hashes = cls_groups.iloc[:n_train]["feature_hash"].tolist()
        val_hashes = cls_groups.iloc[n_train : n_train + n_val]["feature_hash"].tolist()
        test_hashes = cls_groups.iloc[n_train + n_val :]["feature_hash"].tolist()

        for h in train_hashes:
            group_to_partition[h] = "train"
        for h in val_hashes:
            group_to_partition[h] = "val"
        for h in test_hashes:
            group_to_partition[h] = "test"

    return group_to_partition


def create_leakage_resistant_split(
    df: pd.DataFrame,
    target_col: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create a leakage-resistant dataset with partition labels, excluding ambiguous groups."""
    if "feature_hash" not in df.columns:
        df = df.copy()
        df["feature_hash"] = compute_feature_hashes(df)

    # 1. Identify and exclude conflicting label groups for this task
    label_nunique = df.groupby("feature_hash")[target_col].nunique()
    conflict_hashes: Set[str] = set(label_nunique[label_nunique > 1].index)

    clean_df = df[~df["feature_hash"].isin(conflict_hashes)].copy()

    # 2. Group metadata for stratification
    group_summary = (
        clean_df.groupby("feature_hash")
        .agg(
            target=(target_col, "first"),
            row_count=("feature_hash", "count"),
        )
        .reset_index()
    )

    # 3. Stratified group allocation
    group_assignments = partition_groups_stratified(
        group_df=group_summary,
        target_col="target",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed,
    )

    clean_df["partition"] = clean_df["feature_hash"].map(group_assignments)

    # 4. Rigorous verification: assert zero group leakage across partitions
    train_h = set(clean_df[clean_df["partition"] == "train"]["feature_hash"])
    val_h = set(clean_df[clean_df["partition"] == "val"]["feature_hash"])
    test_h = set(clean_df[clean_df["partition"] == "test"]["feature_hash"])

    assert len(train_h & val_h) == 0, "Train-Val group leakage detected!"
    assert len(train_h & test_h) == 0, "Train-Test group leakage detected!"
    assert len(val_h & test_h) == 0, "Val-Test group leakage detected!"

    # 5. Compile split statistics
    stats: Dict[str, Any] = {
        "task": target_col,
        "total_source_records": len(df),
        "quarantined_records": len(df) - len(clean_df),
        "clean_records": len(clean_df),
        "total_groups": len(group_summary),
        "train_rows": int((clean_df["partition"] == "train").sum()),
        "val_rows": int((clean_df["partition"] == "val").sum()),
        "test_rows": int((clean_df["partition"] == "test").sum()),
        "train_pct": round(float((clean_df["partition"] == "train").mean() * 100), 2),
        "val_pct": round(float((clean_df["partition"] == "val").mean() * 100), 2),
        "test_pct": round(float((clean_df["partition"] == "test").mean() * 100), 2),
        "class_breakdown": {},
    }

    for cls in clean_df[target_col].unique():
        cls_sub = clean_df[clean_df[target_col] == cls]
        stats["class_breakdown"][str(cls)] = {
            "total_rows": len(cls_sub),
            "train_rows": int((cls_sub["partition"] == "train").sum()),
            "val_rows": int((cls_sub["partition"] == "val").sum()),
            "test_rows": int((cls_sub["partition"] == "test").sum()),
            "train_pct": round(float((cls_sub["partition"] == "train").mean() * 100), 2),
            "val_pct": round(float((cls_sub["partition"] == "val").mean() * 100), 2),
            "test_pct": round(float((cls_sub["partition"] == "test").mean() * 100), 2),
            "total_groups": cls_sub["feature_hash"].nunique(),
        }

    return clean_df, stats


def export_evaluation_splits(
    clean_df: pd.DataFrame,
    output_dir: Path,
    export_format: str = "csv",
) -> Dict[str, Path]:
    """Export train, val, and test splits into processed storage directories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_files: Dict[str, Path] = {}

    for part in ["train", "val", "test"]:
        sub_df = clean_df[clean_df["partition"] == part].copy()
        if export_format == "parquet":
            out_file = output_dir / f"{part}.parquet"
            sub_df.to_parquet(out_file, index=False)
        else:
            out_file = output_dir / f"{part}.csv"
            sub_df.to_csv(out_file, index=False)
        exported_files[part] = out_file
        logger.info("Saved %s partition: %s (%d rows)", part, out_file, len(sub_df))

    return exported_files
