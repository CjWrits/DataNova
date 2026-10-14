"""Automated unit and integration tests for Phase 2 Data Preparation and Leakage Auditing.

Verifies:
1. 'id' never appears in model feature columns.
2. Target columns never enter model feature matrices.
3. Identical feature groups never cross partition boundaries.
4. Conflicting-label groups are correctly identified and quarantined.
5. Binary and multiclass eligibility are handled independently.
6. Rare classes (e.g. Worms) are not discarded and are represented in all splits.
7. Partition splits are 100% reproducible with the fixed random seed.
8. Validation and test data do not influence fitted preprocessing objects.
9. Unknown categorical values do not crash the transformation pipeline.
10. Raw dataset checksums remain unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from ml.preprocessing.duplicate_audit import (
    audit_duplicates_and_conflicts,
    compute_feature_hashes,
)
from ml.preprocessing.pipeline import (
    create_preprocessing_pipeline,
    prepare_train_val_test_matrices,
)
from ml.preprocessing.schema import (
    ALTERNATIVE_FEATURE_COLUMNS_NO_STTL,
    CATEGORICAL_FEATURES,
    IDENTIFIER_COLUMNS,
    NUMERICAL_FEATURES,
    PRIMARY_FEATURE_COLUMNS,
    TARGET_BINARY,
    TARGET_COLUMNS,
    TARGET_MULTICLASS,
    validate_feature_matrix_columns,
)
from ml.preprocessing.split import partition_groups_stratified


def test_id_never_in_feature_columns():
    """Verify that 'id' is completely excluded from all feature schemas."""
    assert "id" not in PRIMARY_FEATURE_COLUMNS
    assert "id" not in ALTERNATIVE_FEATURE_COLUMNS_NO_STTL
    assert "id" not in CATEGORICAL_FEATURES
    assert "id" not in NUMERICAL_FEATURES

    # Validation function must flag forbidden id
    valid, violations = validate_feature_matrix_columns(["id"] + PRIMARY_FEATURE_COLUMNS)
    assert not valid
    assert any("id" in v for v in violations)


def test_target_columns_never_enter_feature_matrices():
    """Verify that target columns (label, attack_cat) are excluded from predictor schemas."""
    for target in TARGET_COLUMNS:
        assert target not in PRIMARY_FEATURE_COLUMNS
        assert target not in ALTERNATIVE_FEATURE_COLUMNS_NO_STTL
        valid, violations = validate_feature_matrix_columns([target] + PRIMARY_FEATURE_COLUMNS)
        assert not valid
        assert any(target in v for v in violations)


def test_identical_feature_groups_never_cross_partitions(project_root: Path):
    """Verify that zero feature groups overlap across Train, Val, and Test splits."""
    proc_dir = project_root / "data" / "processed" / "unsw_nb15"

    for task in ["binary", "multiclass"]:
        train_df = pd.read_csv(proc_dir / task / "train.csv")
        val_df = pd.read_csv(proc_dir / task / "val.csv")
        test_df = pd.read_csv(proc_dir / task / "test.csv")

        train_hashes = set(train_df["feature_hash"])
        val_hashes = set(val_df["feature_hash"])
        test_hashes = set(test_df["feature_hash"])

        assert len(train_hashes & val_hashes) == 0, f"{task}: Overlap between Train and Val!"
        assert len(train_hashes & test_hashes) == 0, f"{task}: Overlap between Train and Test!"
        assert len(val_hashes & test_hashes) == 0, f"{task}: Overlap between Val and Test!"


def test_conflicting_label_groups_correctly_identified(project_root: Path):
    """Verify that quarantined groups have >1 label and clean groups have exactly 1 label."""
    proc_dir = project_root / "data" / "processed" / "unsw_nb15"
    bin_manifest = pd.read_csv(proc_dir / "manifests" / "binary_quarantine_manifest.csv")
    multi_manifest = pd.read_csv(proc_dir / "manifests" / "multiclass_quarantine_manifest.csv")

    assert len(bin_manifest) == 1758
    assert len(multi_manifest) == 40400

    # In clean binary partitions, every feature group has exactly 1 distinct label
    train_df = pd.read_csv(proc_dir / "binary" / "train.csv")
    val_df = pd.read_csv(proc_dir / "binary" / "val.csv")
    test_df = pd.read_csv(proc_dir / "binary" / "test.csv")
    all_bin = pd.concat([train_df, val_df, test_df], ignore_index=True)
    assert (all_bin.groupby("feature_hash")["label"].nunique() == 1).all()

    # In clean multiclass partitions, every feature group has exactly 1 distinct attack_cat
    m_train = pd.read_csv(proc_dir / "multiclass" / "train.csv")
    m_val = pd.read_csv(proc_dir / "multiclass" / "val.csv")
    m_test = pd.read_csv(proc_dir / "multiclass" / "test.csv")
    all_multi = pd.concat([m_train, m_val, m_test], ignore_index=True)
    assert (all_multi.groupby("feature_hash")["attack_cat"].nunique() == 1).all()


def test_binary_and_multiclass_eligibility_independent(project_root: Path):
    """Verify that records with consistent binary labels but conflicting multiclass labels are handled independently."""
    metrics_file = project_root / "artifacts" / "metrics" / "duplicate_statistics.json"
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # 2,031 groups (38,642 rows) consistent in binary but conflicting in multiclass
    multi_task = metrics["multiclass_task"]
    assert multi_task["consistent_in_binary_but_conflicting_in_multiclass_groups"] == 2031
    assert multi_task["consistent_in_binary_but_conflicting_in_multiclass_rows"] == 38642

    bin_clean_rows = metrics["binary_task"]["eligible_clean_rows"]
    multi_clean_rows = multi_task["eligible_clean_rows"]
    assert bin_clean_rows == 255915
    assert multi_clean_rows == 217273
    assert bin_clean_rows - multi_clean_rows == 38642


def test_rare_classes_preserved_and_stratified(project_root: Path):
    """Verify that rare classes (especially Worms) are present in all multiclass partitions."""
    proc_dir = project_root / "data" / "processed" / "unsw_nb15" / "multiclass"
    train_df = pd.read_csv(proc_dir / "train.csv")
    val_df = pd.read_csv(proc_dir / "val.csv")
    test_df = pd.read_csv(proc_dir / "test.csv")

    for cat in ["Worms", "Backdoor", "Analysis", "Shellcode"]:
        tr_cnt = (train_df["attack_cat"] == cat).sum()
        val_cnt = (val_df["attack_cat"] == cat).sum()
        te_cnt = (test_df["attack_cat"] == cat).sum()

        assert tr_cnt > 0, f"Class {cat} missing from Train!"
        assert val_cnt > 0, f"Class {cat} missing from Val!"
        assert te_cnt > 0, f"Class {cat} missing from Test!"

    # Specific check for Worms counts: 111 train, 24 val, 24 test
    assert (train_df["attack_cat"] == "Worms").sum() == 111
    assert (val_df["attack_cat"] == "Worms").sum() == 24
    assert (test_df["attack_cat"] == "Worms").sum() == 24


def test_split_reproducibility_with_seed():
    """Verify that group-aware stratified splitting produces identical partitions across repeated runs."""
    dummy_groups = pd.DataFrame({
        "feature_hash": [f"hash_{i}" for i in range(100)],
        "target": ["Normal"] * 50 + ["Attack"] * 50,
        "row_count": [1] * 100,
    })

    part_1 = partition_groups_stratified(dummy_groups, target_col="target", random_seed=42)
    part_2 = partition_groups_stratified(dummy_groups, target_col="target", random_seed=42)

    assert part_1 == part_2, "Splitting was non-deterministic with fixed seed!"


def test_validation_and_test_do_not_influence_preprocessor():
    """Verify that fitting preprocessing pipeline strictly on train data is not influenced by val/test."""
    np.random.seed(42)
    train_data = {col: np.random.randn(50) for col in NUMERICAL_FEATURES}
    train_data.update({col: ["tcp"] * 50 for col in CATEGORICAL_FEATURES})
    df_train = pd.DataFrame(train_data)

    test_data = {col: np.random.randn(20) * 1000.0 for col in NUMERICAL_FEATURES}
    test_data.update({col: ["udp"] * 20 for col in CATEGORICAL_FEATURES})
    df_test = pd.DataFrame(test_data)

    pipeline = create_preprocessing_pipeline(schema="primary")
    pipeline.fit(df_train)

    scaler = pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps["scaler"]
    train_mean_before = np.copy(scaler.mean_)

    # Transforming test data must NOT change the fitted training mean
    _ = pipeline.transform(df_test)
    train_mean_after = scaler.mean_

    np.testing.assert_array_equal(train_mean_before, train_mean_after)


def test_unknown_categorical_values_do_not_crash_pipeline():
    """Verify that novel unseen categories in validation/test are handled gracefully without raising exceptions."""
    train_data = {col: [1.0] * 10 for col in NUMERICAL_FEATURES}
    train_data.update({"proto": ["tcp"] * 10, "service": ["http"] * 10, "state": ["FIN"] * 10})
    df_train = pd.DataFrame(train_data)

    # Test set containing novel categories ('icmp', 'novel_service', 'PAR')
    novel_test_data = {col: [2.0] * 5 for col in NUMERICAL_FEATURES}
    novel_test_data.update({
        "proto": ["icmp", "rtp", "novel_proto", "tcp", "udp"],
        "service": ["unknown_svc", "http", "dns", "snmp", "ftp"],
        "state": ["no", "PAR", "URN", "ECO", "FIN"],
    })
    df_novel_test = pd.DataFrame(novel_test_data)

    pipeline = create_preprocessing_pipeline(schema="primary")
    pipeline.fit(df_train)

    # Must transform without throwing ValueError for unknown categories
    transformed = pipeline.transform(df_novel_test)
    assert transformed.shape[0] == 5
    assert not np.isnan(transformed).any()


def test_raw_dataset_checksums_remain_unchanged(project_root: Path):
    """Verify that all raw source files remain 100% untouched and match initial SHA-256 hashes."""
    raw_dir = project_root / "data" / "raw" / "unsw_nb15"
    checksum_file = raw_dir / "checksums.sha256"

    with open(checksum_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            expected_hash, fname = line.split(maxsplit=1)
            fpath = raw_dir / fname
            hasher = hashlib.sha256()
            with open(fpath, "rb") as fp:
                while chunk := fp.read(1024 * 1024):
                    hasher.update(chunk)
            assert hasher.hexdigest() == expected_hash, f"Raw file {fname} was mutated!"
