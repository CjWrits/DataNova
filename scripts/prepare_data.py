"""End-to-end Data Preparation, Duplicate Auditing, and Evaluation Split Generation Script.

Executes:
1. Feature vector hashing and duplicate audit across combined records.
2. Quarantine of conflicting-label records independently for binary and multiclass tasks.
3. Group-aware stratified 70/15/15 partition construction.
4. Saving processed partitions and manifest artifacts to data/processed/unsw_nb15/.
5. Saving audit metrics to artifacts/metrics/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict

import pandas as pd

from ml.preprocessing.duplicate_audit import (
    audit_duplicates_and_conflicts,
    compute_feature_hashes,
    save_quarantine_manifests,
)
from ml.preprocessing.schema import (
    PRIMARY_FEATURE_COLUMNS,
    TARGET_BINARY,
    TARGET_MULTICLASS,
)
from ml.preprocessing.split import (
    create_leakage_resistant_split,
    export_evaluation_splits,
)

logger = logging.getLogger("prepare_data")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_data_preparation(
    raw_dir: Path,
    processed_dir: Path,
    artifacts_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Execute end-to-end data preparation workflow."""
    train_file = raw_dir / "UNSW_NB15_training-set.csv"
    test_file = raw_dir / "UNSW_NB15_testing-set.csv"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(f"Missing raw files in {raw_dir}. Run scripts/download_unsw_nb15.py first.")

    logger.info("Loading original raw benchmark files...")
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    df_train["origin_split"] = "author_train"
    df_test["origin_split"] = "author_test"

    df_all = pd.concat([df_train, df_test], ignore_index=True)
    logger.info("Combined raw record pool: %d rows x %d cols", len(df_all), len(df_all.columns))

    # 1. Compute deterministic feature hashes
    logger.info("Computing deterministic SHA-256 group identifiers on 42 predictor attributes...")
    df_all["feature_hash"] = compute_feature_hashes(df_all, feature_cols=PRIMARY_FEATURE_COLUMNS)

    # 2. Perform duplicate and label-conflict audit
    logger.info("Performing duplicate and conflicting-label audit...")
    audit_stats = audit_duplicates_and_conflicts(df_all, feature_cols=PRIMARY_FEATURE_COLUMNS)

    # Save duplicate audit metrics
    metrics_dir = artifacts_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    dup_stats_path = metrics_dir / "duplicate_statistics.json"
    with open(dup_stats_path, "w", encoding="utf-8") as f:
        json.dump(audit_stats, f, indent=2)
    logger.info("Saved duplicate statistics to %s", dup_stats_path)

    # Save quarantine manifests
    manifest_dir = processed_dir / "manifests"
    bin_quarantine_path, multi_quarantine_path = save_quarantine_manifests(df_all, manifest_dir)
    logger.info("Saved binary quarantine manifest:     %s", bin_quarantine_path)
    logger.info("Saved multiclass quarantine manifest: %s", multi_quarantine_path)

    # 3. Construct Binary Classification Evaluation Split
    logger.info("Constructing group-aware stratified split for BINARY task (label)...")
    clean_bin_df, bin_split_stats = create_leakage_resistant_split(
        df=df_all,
        target_col=TARGET_BINARY,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed,
    )
    bin_dir = processed_dir / "binary"
    export_evaluation_splits(clean_bin_df, bin_dir, export_format="csv")

    # 4. Construct Multiclass Classification Evaluation Split
    logger.info("Constructing group-aware stratified split for MULTICLASS task (attack_cat)...")
    clean_multi_df, multi_split_stats = create_leakage_resistant_split(
        df=df_all,
        target_col=TARGET_MULTICLASS,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed,
    )
    multi_dir = processed_dir / "multiclass"
    export_evaluation_splits(clean_multi_df, multi_dir, export_format="csv")

    # Save split metrics
    split_stats_path = metrics_dir / "split_statistics.json"
    combined_split_stats = {
        "binary_task": bin_split_stats,
        "multiclass_task": multi_split_stats,
    }
    with open(split_stats_path, "w", encoding="utf-8") as f:
        json.dump(combined_split_stats, f, indent=2)
    logger.info("Saved split statistics to %s", split_stats_path)

    return {
        "audit_statistics": audit_stats,
        "split_statistics": combined_split_stats,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UNSW-NB15 Data Preparation and Splitting.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/unsw_nb15"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/unsw_nb15"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run_data_preparation(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        artifacts_dir=args.artifacts_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.seed,
    )
    print("\nData Preparation Complete!")
    print(f"Binary Clean Rows:     {results['split_statistics']['binary_task']['clean_records']:,}")
    print(f"Multiclass Clean Rows: {results['split_statistics']['multiclass_task']['clean_records']:,}")
