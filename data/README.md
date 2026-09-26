# Data Directory Architecture & Governance

## Overview
This directory stores all raw, processed, and external datasets used by the **Adaptive Threat Intelligence** project. To maintain ML scientific integrity and prevent data leakage, strict separation between data tiers is enforced.

## Directory Structure
- `data/raw/` — Read-only, uncompressed or compressed raw benchmark dumps (e.g., CIC-IDS-2017, UNSW-NB15, or NSL-KDD). Files in this folder are **strictly immutable** and excluded from Git tracking.
- `data/processed/` — Partitioned, cleaned, feature-engineered artifacts generated via deterministic pipelines. Contains training, validation, and out-of-time test splits saved in parquet or memory-mapped formats.
- `data/external/` — Threat intelligence feeds, IP reputation blacklists, CVE taxonomies, and protocol metadata dictionaries used for contextual enrichment.

## Data Governance & Leakage Prevention Rules
1. **Zero Raw Modification**: Raw datasets must never be edited, normalized in-place, or overwritten.
2. **Deterministic Partitioning**: Splitting into Train, Validation, and Test must happen **before** any normalization, scaling, imputation, or synthetic balancing (e.g., SMOTE).
3. **Leakage Elimination**: Statistical transformers (MinMaxScaler, StandardScaler, QuantileTransformer) must be fitted **exclusively on the training split** and applied identically to validation, test, and inference payloads.
4. **Temporal Ordering & Split Strategy**: Where timestamps are available (e.g. CIC-IDS-2017), splits must honor temporal boundaries to prevent look-ahead bias and simulate real-world arrival patterns.
5. **Class Imbalance Integrity**: Class rebalancing techniques (SMOTE, ADASYN, RandomUnderSampler) must **never** be applied to validation, test, or evaluation sets. Test splits must preserve natural prior class distributions to accurately reflect operational False Positive Rates (FPR).

## Storage Policy
All large tabular files (`*.csv`, `*.parquet`, `*.pcap`, `*.feather`, `*.h5`) are ignored by `.gitignore`. Only directory structures, documentation, and schema definitions are tracked.
