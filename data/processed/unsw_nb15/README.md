# Processed & Leakage-Resistant UNSW-NB15 Partitions

## Overview
This directory contains the cleaned, leakage-resistant, group-aware evaluation partitions constructed during Phase 2 of the **Adaptive Threat Intelligence** project.

To eliminate the severe evaluation artifacts discovered in the raw author-provided splits—namely cross-partition duplicate leakage (8,421 test rows matching train), contradictory labeling (414 feature vectors), and sequential ID ordering ($r = +0.7272$)—these partitions enforce strict group-level isolation.

## Directory Structure
- `binary/`:
  - `train.csv` (179,511 rows, 70.14%)
  - `val.csv` (37,976 rows, 14.84%)
  - `test.csv` (38,428 rows, 15.02%)
  - *Clean pool*: 255,915 records across 153,270 unique feature groups.
  - *Quarantined*: 1,758 rows belonging to 414 ambiguous groups with conflicting `label` values (0 and 1).
- `multiclass/`:
  - `train.csv` (151,325 rows, 69.65%)
  - `val.csv` (33,045 rows, 15.21%)
  - `test.csv` (32,903 rows, 15.14%)
  - *Clean pool*: 217,273 records across 151,239 unique feature groups.
  - *Quarantined*: 40,400 rows belonging to 2,445 groups with conflicting `attack_cat` values.
- `manifests/`:
  - `binary_quarantine_manifest.csv`: Audit log of rows excluded from the binary supervised task.
  - `multiclass_quarantine_manifest.csv`: Audit log of rows excluded from the multiclass task.

## Key Properties & Leakage Invariants
1. **Zero Feature-Vector Overlap**: Every unique predictor feature vector belongs to **exactly one partition** ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
2. **Deterministic Hashing**: Groups are identified via canonical SHA-256 digests over the 42 predictor columns (excluding `id`, `label`, `attack_cat`).
3. **No Identifier Predictors**: The sequential `id` column is excluded from all feature matrices.
4. **Rare Class Representation**: The extreme minority class `Worms` (159 clean distinct feature groups) is preserved and stratified across partitions (111 Train, 24 Val, 24 Test).
5. **No Synthetic Retraining**: Raw files remain unaltered in `data/raw/unsw_nb15/` for reference benchmarking.

## Regeneration Command
```bash
python scripts/prepare_data.py --raw-dir data/raw/unsw_nb15 --processed-dir data/processed/unsw_nb15 --artifacts-dir artifacts
```
