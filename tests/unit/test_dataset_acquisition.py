"""Unit tests validating dataset acquisition, integrity checksums, and EDA outputs."""

import hashlib
import json
import subprocess
from pathlib import Path
import pandas as pd
import pytest


def test_checksum_manifest_verification(project_root: Path):
    """Verify that all downloaded files match their SHA-256 checksum manifest."""
    raw_dir = project_root / "data" / "raw" / "unsw_nb15"
    checksum_file = raw_dir / "checksums.sha256"
    assert checksum_file.is_file(), "checksums.sha256 does not exist."

    with open(checksum_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) >= 4, f"Expected at least 4 files in checksum manifest, found {len(lines)}"

    for line in lines:
        expected_hash, filename = line.split(maxsplit=1)
        filepath = raw_dir / filename
        assert filepath.is_file(), f"Acquired file missing: {filepath}"

        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(1024 * 1024):
                hasher.update(chunk)
        calculated_hash = hasher.hexdigest()
        assert calculated_hash == expected_hash, (
            f"Checksum mismatch for {filename}: expected {expected_hash}, calculated {calculated_hash}"
        )


def test_dataset_readability_and_schema(project_root: Path):
    """Verify that train and test CSVs are readable and match expected dimensions."""
    raw_dir = project_root / "data" / "raw" / "unsw_nb15"
    train_path = raw_dir / "UNSW_NB15_training-set.csv"
    test_path = raw_dir / "UNSW_NB15_testing-set.csv"

    # Fast read check
    df_train_head = pd.read_csv(train_path, nrows=10)
    df_test_head = pd.read_csv(test_path, nrows=10)

    assert len(df_train_head.columns) == 45
    assert len(df_test_head.columns) == 45
    assert list(df_train_head.columns) == list(df_test_head.columns)
    assert "label" in df_train_head.columns
    assert "attack_cat" in df_train_head.columns
    assert "id" in df_train_head.columns


def test_eda_visualizations_exist(project_root: Path):
    """Verify that all 6 required EDA visualization artifacts are generated."""
    vis_dir = project_root / "research" / "visualizations"
    expected_plots = [
        "attack_category_distribution.png",
        "class_imbalance_binary.png",
        "feature_correlation_heatmap.png",
        "id_target_leakage.png",
        "minority_class_breakdown.png",
        "synthetic_artifact_leakage_sttl.png",
    ]
    for plot_name in expected_plots:
        plot_path = vis_dir / plot_name
        assert plot_path.is_file(), f"Missing visualization: {plot_name}"
        assert plot_path.stat().st_size > 10_000, f"Visualization {plot_name} appears empty or corrupted"


def test_eda_report_and_notebook_exist(project_root: Path):
    """Verify that the EDA report and Jupyter notebook are present and valid."""
    report_file = project_root / "research" / "reports" / "unsw_nb15_eda_report.md"
    assert report_file.is_file(), "unsw_nb15_eda_report.md is missing."
    assert report_file.stat().st_size > 2_000, "unsw_nb15_eda_report.md is unexpectedly small."

    notebook_file = project_root / "research" / "notebooks" / "01_unsw_nb15_eda.ipynb"
    assert notebook_file.is_file(), "01_unsw_nb15_eda.ipynb is missing."
    with open(notebook_file, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
    assert "cells" in nb_data
    assert len(nb_data["cells"]) > 0


def test_git_excludes_raw_csvs(project_root: Path):
    """Ensure git check-ignore confirms raw CSV files are not accidentally committed."""
    raw_csv = project_root / "data" / "raw" / "unsw_nb15" / "UNSW_NB15_training-set.csv"
    result = subprocess.run(
        ["git", "check-ignore", str(raw_csv)],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Raw CSV {raw_csv} is NOT properly ignored by Git!"
