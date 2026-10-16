# Data Quality & Duplicate Audit Report

**Project**: Adaptive Threat Intelligence  
**Phase**: Phase 2 — Data Preparation and Leakage Auditing  
**Dataset**: UNSW-NB15 Combined Pool (257,673 records)  
**Date**: October 2026  

---

## 1. Overview of the Audit Process

To eliminate evaluation distortions caused by repeated telemetry and ambiguous annotations, a deterministic feature-vector hashing and conflict audit was conducted across the 257,673 combined records of UNSW-NB15.

Each network flow was projected onto its **42 predictor features** (3 categorical: `proto`, `service`, `state`; and 39 numerical flow metrics). Identifiers (`id`) and ground-truth targets (`label`, `attack_cat`) were strictly excluded. A canonical string representation of each feature vector was hashed using SHA-256 to assign a globally unique `feature_hash`.

---

## 2. Global Duplicate Structure

| Metric | Measured Value | Percentage of Combined Pool |
| :--- | :--- | :--- |
| Total Combined Records | 257,673 | 100.00% |
| **Unique Feature Groups (Distinct Vectors)** | **153,684** | **59.64%** |
| Internal Duplicate Records | 103,989 | 40.36% |
| Largest Feature Group Size | 38,126 rows | 14.80% |

> **Finding**: Over 40% of records in the raw UNSW-NB15 benchmark partitions are exact duplicate repetitions of existing flows. A single high-volume flow signature (repeated background DNS queries) accounts for 38,126 identical rows.

---

## 3. Label Conflict & Ambiguity Analysis

A feature group is classified as **conflicting** for a supervised task if rows sharing an identical predictor feature vector are assigned different ground-truth target values. Arbitrarily selecting one label or averaging predictions introduces noise; therefore, conflicting groups are systematically quarantined.

### 3.1 Binary Task Conflicts (`label`: 0 vs. 1)
- **Conflicting Unique Feature Groups**: **414 groups** (0.27% of groups).
- **Affected Quarantined Records**: **1,758 rows** (0.68% of dataset).
- **Eligible Clean Records**: **255,915 rows** across **153,270 unique groups**.
- *Root Cause*: Standard benign traffic (e.g., DNS queries, routine HTTP requests) occurring during an active attack simulation window that were coarsely labeled as malicious due to timestamp-based boundary slicing.

### 3.2 Multiclass Task Conflicts (`attack_cat`: 10 classes)
- **Conflicting Unique Feature Groups**: **2,445 groups** (1.59% of groups).
- **Affected Quarantined Records**: **40,400 rows** (15.68% of dataset).
- **Eligible Clean Records**: **217,273 rows** across **151,239 unique groups**.
- *Root Cause*: Attack tools in the IXIA testbed generated overlapping flow signatures across different threat categories (e.g. DoS vs. Exploits vs. Generic).

### 3.3 Task-Specific Eligibility Independence
Crucially, **2,031 feature groups** (affecting **38,642 rows**) are **100% consistent for binary classification** (every row in the group is `label=1`, Attack), but **conflicting for multiclass classification** (labeled with different attack families).
- **Binary Treatment**: These 38,642 flows are unambiguously malicious and remain fully eligible for binary intrusion detection.
- **Multiclass Treatment**: Because their specific attack family is contradictory, they are quarantined from the multiclass supervised dataset.

---

## 4. Deep-Dive Audit of Rare Classes (Worms, Backdoor, Analysis, Shellcode)

| Attack Category | Raw Rows | Raw Distinct Groups | Conflicting Quarantined Rows | **Clean Eligible Rows** | **Clean Distinct Feature Groups** | Group Diversity Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Normal** | 93,000 | 85,722 | 928 | **92,072** | **85,308** | 92.7% |
| **Generic** | 58,871 | 7,599 | 373 | **58,498** | **7,283** | 12.5% |
| **Exploits** | 44,525 | 27,434 | 16,507 | **28,018** | **25,451** | 90.8% |
| **Fuzzers** | 24,246 | 20,960 | 3,617 | **20,629** | **18,749** | 90.9% |
| **DoS** | 16,353 | 5,500 | 12,663 | **3,690** | **3,624** | 98.2% |
| **Reconnaissance** | 13,987 | 9,991 | 2,246 | **11,741** | **8,419** | 71.7% |
| **Analysis** | 2,677 | 2,032 | 2,061 | **616** | **450** | 73.1% |
| **Backdoor** | 2,329 | 1,880 | 1,977 | **352** | **350** | 99.4% |
| **Shellcode** | 1,511 | 1,456 | 13 | **1,498** | **1,446** | 96.5% |
| **Worms** | 174 | 171 | 15 | **159** | **159** | **100.0%** |

### Special Audit Findings on the Worms Class
1. **Raw Count**: 174 rows.
2. **Conflicting Records**: Exactly 15 rows belonged to 12 feature groups that conflicted with `Exploits` or `DoS`.
3. **Eligible Clean Rows**: Exactly **159 rows**.
4. **Distinct Worms Groups**: Exactly **159 groups**.
5. **Group Multiplicity**: Every single clean Worms flow is a completely unique, distinct feature vector of group size 1.
6. **Evaluation Feasibility**: Because there are 159 independent feature groups, Worms can be divided cleanly across Train (111 groups / 69.8%), Validation (24 groups / 15.1%), and Testing (24 groups / 15.1%) without any cross-partition leakage.

---

## 5. Quarantine Manifests & Reproducibility

Quarantined records are exported to audit manifests:
- `data/processed/unsw_nb15/manifests/binary_quarantine_manifest.csv` (1,758 records)
- `data/processed/unsw_nb15/manifests/multiclass_quarantine_manifest.csv` (40,400 records)

These manifests document the `id`, `feature_hash`, and conflicting label combinations, ensuring full forensic transparency.
