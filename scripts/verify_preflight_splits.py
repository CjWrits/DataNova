"""Preflight split verification script for Phase 3.

Performs rigorous preflight audits on raw checksums and processed split partitions:
1. Verifies raw UNSW-NB15 file checksums.
2. Audits Binary and Multiclass split partition sizes and feature group counts.
3. Computes complete per-class row and group distributions across Train, Val, and Test.
4. Verifies zero group overlap between all pairs of partitions.
5. Confirms rare class preservation (e.g., Worms, Shellcode, Backdoor).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing.schema import (
    ALTERNATIVE_FEATURE_COLUMNS_NO_STTL,
    PRIMARY_FEATURE_COLUMNS,
    MULTICLASS_LABEL_MAPPING,
)
from ml.preprocessing.duplicate_audit import compute_feature_hashes


def verify_raw_checksums() -> bool:
    """Verify SHA-256 checksums of raw CSV files."""
    manifest_path = PROJECT_ROOT / "data" / "raw" / "unsw_nb15" / "checksums.sha256"
    if not manifest_path.exists():
        print(f"[FAIL] Manifest missing: {manifest_path}")
        return False

    all_matched = True
    print("\n=== Raw Dataset Checksum Verification ===")
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            expected_hash, filename = line.split(maxsplit=1)
            file_path = PROJECT_ROOT / "data" / "raw" / "unsw_nb15" / filename
            if not file_path.exists():
                print(f"[MISSING] {filename}")
                all_matched = False
                continue

            h = hashlib.sha256()
            with open(file_path, "rb") as bf:
                while chunk := bf.read(1024 * 1024):
                    h.update(chunk)
            actual_hash = h.hexdigest()
            if actual_hash == expected_hash:
                print(f"[MATCH] {filename} -> {actual_hash[:16]}...")
            else:
                print(f"[FAIL] {filename} mismatch! Expected {expected_hash}, got {actual_hash}")
                all_matched = False

    return all_matched


def audit_partition_groups(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[bool, Dict[str, Set[str]]]:
    """Compute feature groups and verify complete isolation."""
    train_fps = set(compute_feature_hashes(train_df, feature_cols))
    val_fps = set(compute_feature_hashes(val_df, feature_cols))
    test_fps = set(compute_feature_hashes(test_df, feature_cols))

    leak_tr_val = train_fps & val_fps
    leak_tr_te = train_fps & test_fps
    leak_val_te = val_fps & test_fps

    is_clean = len(leak_tr_val) == 0 and len(leak_tr_te) == 0 and len(leak_val_te) == 0

    fps_dict = {
        "train": train_fps,
        "val": val_fps,
        "test": test_fps,
    }
    return is_clean, fps_dict


def run_preflight_audit() -> Dict:
    """Run full preflight audit and return structured results."""
    print("=" * 80)
    print("PHASE 3 PREFLIGHT VERIFICATION: SPLIT & CLASS INTEGRITY AUDIT")
    print("=" * 80)

    raw_ok = verify_raw_checksums()

    results = {
        "raw_checksums_verified": raw_ok,
        "binary": {},
        "multiclass": {},
    }

    # 1. Binary Audit
    print("\n--- 1. Binary Partitions Audit ---")
    bin_dir = PROJECT_ROOT / "data" / "processed" / "unsw_nb15" / "binary"
    train_bin = pd.read_csv(bin_dir / "train.csv")
    val_bin = pd.read_csv(bin_dir / "val.csv")
    test_bin = pd.read_csv(bin_dir / "test.csv")

    total_bin_rows = len(train_bin) + len(val_bin) + len(test_bin)
    bin_clean, bin_fps = audit_partition_groups(train_bin, val_bin, test_bin, PRIMARY_FEATURE_COLUMNS)
    total_bin_groups = len(bin_fps["train"]) + len(bin_fps["val"]) + len(bin_fps["test"])

    print(f"Total Rows: {total_bin_rows:,} (Train: {len(train_bin):,}, Val: {len(val_bin):,}, Test: {len(test_bin):,})")
    print(f"Total Unique Feature Groups: {total_bin_groups:,} (Train: {len(bin_fps['train']):,}, Val: {len(bin_fps['val']):,}, Test: {len(bin_fps['test']):,})")
    print(f"Partition Group Isolation: {'PASSED (Zero Leakage)' if bin_clean else 'FAILED (Leakage Detected)'}")

    # Binary class distribution
    train_bin['fp'] = compute_feature_hashes(train_bin, PRIMARY_FEATURE_COLUMNS)
    val_bin['fp'] = compute_feature_hashes(val_bin, PRIMARY_FEATURE_COLUMNS)
    test_bin['fp'] = compute_feature_hashes(test_bin, PRIMARY_FEATURE_COLUMNS)

    print("\nBinary Class Distribution (Rows & Groups):")
    bin_classes = [0, 1]
    bin_dist = {}
    for cls in bin_classes:
        name = "Normal" if cls == 0 else "Attack"
        tr_r = int((train_bin['label'] == cls).sum())
        val_r = int((val_bin['label'] == cls).sum())
        te_r = int((test_bin['label'] == cls).sum())
        tot_r = tr_r + val_r + te_r

        tr_g = int(train_bin[train_bin['label'] == cls]['fp'].nunique())
        val_g = int(val_bin[val_bin['label'] == cls]['fp'].nunique())
        te_g = int(test_bin[test_bin['label'] == cls]['fp'].nunique())
        tot_g = tr_g + val_g + te_g

        bin_dist[name] = {
            "total_rows": tot_r,
            "train_rows": tr_r,
            "val_rows": val_r,
            "test_rows": te_r,
            "train_row_pct": round(tr_r / tot_r * 100, 2),
            "val_row_pct": round(val_r / tot_r * 100, 2),
            "test_row_pct": round(te_r / tot_r * 100, 2),
            "total_groups": tot_g,
            "train_groups": tr_g,
            "val_groups": val_g,
            "test_groups": te_g,
            "train_grp_pct": round(tr_g / tot_g * 100, 2),
            "val_grp_pct": round(val_g / tot_g * 100, 2),
            "test_grp_pct": round(te_g / tot_g * 100, 2),
            "rows_per_group": round(tot_r / tot_g, 2),
        }
        print(f"  {name:8s}: {tot_r:>7,d} rows ({tot_g:>6,d} grps, {tot_r/tot_g:.2f} r/g) -> "
              f"Tr: {tr_r:>6,d} ({tr_r/tot_r*100:.1f}%), Val: {val_r:>6,d} ({val_r/tot_r*100:.1f}%), Te: {te_r:>6,d} ({te_r/tot_r*100:.1f}%) | "
              f"Grps Tr: {tr_g:>5,d} ({tr_g/tot_g*100:.1f}%), Val: {val_g:>5,d} ({val_g/tot_g*100:.1f}%), Te: {te_g:>5,d} ({te_g/tot_g*100:.1f}%)")

    results["binary"] = {
        "isolation_verified": bin_clean,
        "total_rows": total_bin_rows,
        "train_rows": len(train_bin),
        "val_rows": len(val_bin),
        "test_rows": len(test_bin),
        "total_groups": total_bin_groups,
        "class_distribution": bin_dist,
    }

    # 2. Multiclass Audit
    print("\n--- 2. Multiclass Partitions Audit ---")
    mc_dir = PROJECT_ROOT / "data" / "processed" / "unsw_nb15" / "multiclass"
    train_mc = pd.read_csv(mc_dir / "train.csv")
    val_mc = pd.read_csv(mc_dir / "val.csv")
    test_mc = pd.read_csv(mc_dir / "test.csv")

    total_mc_rows = len(train_mc) + len(val_mc) + len(test_mc)
    mc_clean, mc_fps = audit_partition_groups(train_mc, val_mc, test_mc, PRIMARY_FEATURE_COLUMNS)
    total_mc_groups = len(mc_fps["train"]) + len(mc_fps["val"]) + len(mc_fps["test"])

    print(f"Total Rows: {total_mc_rows:,} (Train: {len(train_mc):,}, Val: {len(val_mc):,}, Test: {len(test_mc):,})")
    print(f"Total Unique Feature Groups: {total_mc_groups:,} (Train: {len(mc_fps['train']):,}, Val: {len(mc_fps['val']):,}, Test: {len(mc_fps['test']):,})")
    print(f"Partition Group Isolation: {'PASSED (Zero Leakage)' if mc_clean else 'FAILED (Leakage Detected)'}")

    train_mc['fp'] = compute_feature_hashes(train_mc, PRIMARY_FEATURE_COLUMNS)
    val_mc['fp'] = compute_feature_hashes(val_mc, PRIMARY_FEATURE_COLUMNS)
    test_mc['fp'] = compute_feature_hashes(test_mc, PRIMARY_FEATURE_COLUMNS)

    print("\nMulticlass Class Distribution (Rows & Groups across 10 Classes):")
    mc_dist = {}
    rare_classes_ok = True
    for cat in MULTICLASS_LABEL_MAPPING.keys():
        tr_r = int((train_mc['attack_cat'] == cat).sum())
        val_r = int((val_mc['attack_cat'] == cat).sum())
        te_r = int((test_mc['attack_cat'] == cat).sum())
        tot_r = tr_r + val_r + te_r

        tr_g = int(train_mc[train_mc['attack_cat'] == cat]['fp'].nunique())
        val_g = int(val_mc[val_mc['attack_cat'] == cat]['fp'].nunique())
        te_g = int(test_mc[test_mc['attack_cat'] == cat]['fp'].nunique())
        tot_g = tr_g + val_g + te_g

        if tr_r == 0 or val_r == 0 or te_r == 0:
            rare_classes_ok = False

        mc_dist[cat] = {
            "total_rows": tot_r,
            "train_rows": tr_r,
            "val_rows": val_r,
            "test_rows": te_r,
            "train_row_pct": round(tr_r / tot_r * 100, 2),
            "val_row_pct": round(val_r / tot_r * 100, 2),
            "test_row_pct": round(te_r / tot_r * 100, 2),
            "total_groups": tot_g,
            "train_groups": tr_g,
            "val_groups": val_g,
            "test_groups": te_g,
            "train_grp_pct": round(tr_g / tot_g * 100, 2),
            "val_grp_pct": round(val_g / tot_g * 100, 2),
            "test_grp_pct": round(te_g / tot_g * 100, 2),
            "rows_per_group": round(tot_r / tot_g, 2),
        }

        print(f"  {cat:15s}: {tot_r:>7,d} rows ({tot_g:>6,d} grps, {tot_r/tot_g:5.2f} r/g) | "
              f"Rows Tr:{tr_r:>6,d} ({tr_r/tot_r*100:4.1f}%), Val:{val_r:>5,d} ({val_r/tot_r*100:4.1f}%), Te:{te_r:>5,d} ({te_r/tot_r*100:4.1f}%) | "
              f"Grps Tr:{tr_g:>5,d} ({tr_g/tot_g*100:4.1f}%), Val:{val_g:>4,d} ({val_g/tot_g*100:4.1f}%), Te:{te_g:>4,d} ({te_g/tot_g*100:4.1f}%)")

    print(f"\nAll 10 Classes Preserved Across Train/Val/Test: {'PASSED' if rare_classes_ok else 'FAILED'}")

    results["multiclass"] = {
        "isolation_verified": mc_clean,
        "all_classes_preserved": rare_classes_ok,
        "total_rows": total_mc_rows,
        "train_rows": len(train_mc),
        "val_rows": len(val_mc),
        "test_rows": len(test_mc),
        "total_groups": total_mc_groups,
        "class_distribution": mc_dist,
    }

    # Save preflight audit report
    out_path = PROJECT_ROOT / "artifacts" / "metrics" / "preflight_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nPreflight audit report saved to: {out_path}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_preflight_audit()
