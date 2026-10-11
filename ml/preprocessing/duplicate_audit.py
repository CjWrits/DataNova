"""Duplicate and conflicting-label audit module.

Performs group-level hashing on predictor feature vectors, identifies internal
and cross-partition duplicates, flags conflicting label groups, and quarantines
ambiguous examples independently for binary and multiclass tasks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

from ml.preprocessing.schema import (
    IDENTIFIER_COLUMNS,
    PRIMARY_FEATURE_COLUMNS,
    TARGET_BINARY,
    TARGET_MULTICLASS,
)


def compute_feature_hashes(
    df: pd.DataFrame,
    feature_cols: List[str] | None = None,
) -> pd.Series:
    """Compute deterministic SHA-256 group identifiers for each feature vector."""
    if feature_cols is None:
        feature_cols = [c for c in PRIMARY_FEATURE_COLUMNS if c in df.columns]

    # Verify no identifiers or targets are present
    forbidden = set(IDENTIFIER_COLUMNS) | {TARGET_BINARY, TARGET_MULTICLASS}
    assert not any(c in forbidden for c in feature_cols), "Forbidden column in hash columns!"

    # Ensure consistent column ordering for deterministic hashing
    canonical_cols = sorted(feature_cols)

    # Construct canonical representation per row
    hashes = [
        hashlib.sha256(
            "|".join(f"{col}={val}" for col, val in zip(canonical_cols, tup)).encode("utf-8")
        ).hexdigest()
        for tup in zip(*[df[col] for col in canonical_cols])
    ]
    return pd.Series(hashes, index=df.index, name="feature_hash")


def audit_duplicates_and_conflicts(
    df: pd.DataFrame,
    feature_cols: List[str] | None = None,
) -> Dict[str, Any]:
    """Perform a comprehensive duplicate and label conflict audit across the dataset."""
    if "feature_hash" not in df.columns:
        df = df.copy()
        df["feature_hash"] = compute_feature_hashes(df, feature_cols=feature_cols)

    total_rows = len(df)
    unique_groups = df["feature_hash"].nunique()

    # Binary conflict analysis (label: 0 vs 1)
    bin_grp = df.groupby("feature_hash")[TARGET_BINARY].agg(["nunique", "count", "unique"])
    bin_conflicts_df = bin_grp[bin_grp["nunique"] > 1]
    bin_conflict_hashes: Set[str] = set(bin_conflicts_df.index)
    bin_conflict_rows_count = int(bin_conflicts_df["count"].sum())

    # Multiclass conflict analysis (attack_cat)
    multi_grp = df.groupby("feature_hash")[TARGET_MULTICLASS].agg(["nunique", "count", "unique"])
    multi_conflicts_df = multi_grp[multi_grp["nunique"] > 1]
    multi_conflict_hashes: Set[str] = set(multi_conflicts_df.index)
    multi_conflict_rows_count = int(multi_conflicts_df["count"].sum())

    # Groups consistent in binary but conflicting in multiclass
    multi_only_conflicts = multi_conflict_hashes - bin_conflict_hashes
    multi_only_rows_count = int(df[df["feature_hash"].isin(multi_only_conflicts)].shape[0])

    # Category breakdown for multiclass clean vs conflicting
    category_audit: Dict[str, Dict[str, Any]] = {}
    for cat in df[TARGET_MULTICLASS].unique():
        cat_df = df[df[TARGET_MULTICLASS] == cat]
        raw_rows = len(cat_df)
        raw_groups = cat_df["feature_hash"].nunique()
        conf_rows = int(cat_df[cat_df["feature_hash"].isin(multi_conflict_hashes)].shape[0])
        clean_rows = raw_rows - conf_rows
        clean_groups = cat_df[~cat_df["feature_hash"].isin(multi_conflict_hashes)]["feature_hash"].nunique()
        category_audit[str(cat)] = {
            "raw_rows": raw_rows,
            "raw_unique_groups": raw_groups,
            "conflicting_rows": conf_rows,
            "eligible_clean_rows": clean_rows,
            "eligible_clean_groups": clean_groups,
        }

    audit_summary = {
        "total_records": total_rows,
        "unique_feature_groups": unique_groups,
        "internal_duplicates_count": total_rows - unique_groups,
        "internal_duplicates_percentage": round(((total_rows - unique_groups) / total_rows) * 100, 2),
        "binary_task": {
            "conflicting_groups_count": len(bin_conflict_hashes),
            "conflicting_rows_quarantined": bin_conflict_rows_count,
            "eligible_clean_rows": total_rows - bin_conflict_rows_count,
            "eligible_clean_groups": unique_groups - len(bin_conflict_hashes),
        },
        "multiclass_task": {
            "conflicting_groups_count": len(multi_conflict_hashes),
            "conflicting_rows_quarantined": multi_conflict_rows_count,
            "eligible_clean_rows": total_rows - multi_conflict_rows_count,
            "eligible_clean_groups": unique_groups - len(multi_conflict_hashes),
            "consistent_in_binary_but_conflicting_in_multiclass_groups": len(multi_only_conflicts),
            "consistent_in_binary_but_conflicting_in_multiclass_rows": multi_only_rows_count,
        },
        "category_breakdown": category_audit,
    }

    return audit_summary


def save_quarantine_manifests(
    df: pd.DataFrame,
    manifest_dir: Path,
) -> Tuple[Path, Path]:
    """Generate and save quarantine manifests listing excluded records for each task."""
    manifest_dir.mkdir(parents=True, exist_ok=True)

    if "feature_hash" not in df.columns:
        df = df.copy()
        df["feature_hash"] = compute_feature_hashes(df)

    bin_grp = df.groupby("feature_hash")[TARGET_BINARY].nunique()
    bin_conflict_hashes = set(bin_grp[bin_grp > 1].index)
    bin_quarantine = df[df["feature_hash"].isin(bin_conflict_hashes)]

    multi_grp = df.groupby("feature_hash")[TARGET_MULTICLASS].nunique()
    multi_conflict_hashes = set(multi_grp[multi_grp > 1].index)
    multi_quarantine = df[df["feature_hash"].isin(multi_conflict_hashes)]

    bin_path = manifest_dir / "binary_quarantine_manifest.csv"
    multi_path = manifest_dir / "multiclass_quarantine_manifest.csv"

    bin_quarantine[["id", "feature_hash", TARGET_BINARY, TARGET_MULTICLASS]].to_csv(bin_path, index=False)
    multi_quarantine[["id", "feature_hash", TARGET_BINARY, TARGET_MULTICLASS]].to_csv(multi_path, index=False)

    return bin_path, multi_path
