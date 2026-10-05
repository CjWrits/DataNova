# UNSW-NB15 Exploratory Data Analysis & Empirical Investigation Report

**Project**: Adaptive Threat Intelligence (DataNova Global Data-thon 2026)  
**Phase**: Phase 1 — Dataset Acquisition and Exploratory Data Analysis (EDA)  
**Dataset**: UNSW-NB15 Network Intrusion Detection Dataset  
**Date**: October 2026  
**Authors**: ML Research & Engineering Team  

---

## Executive Summary

This report delivers an empirical analysis of the acquired **UNSW-NB15** network intrusion benchmark files. Rather than relying on literature assertions, this analysis directly inspects the underlying binary data, schema structures, label distributions, duplicate records, cross-partition contamination, and potential target leakage vectors.

### Key Discoveries at a Glance
1. **The Inverted Benchmark Class Distribution**: While the raw 2.54M capture has an attack prevalence of only 12.65% (87.35% normal), the author-curated benchmark partitions invert this ratio: **55.06% Attack** in `UNSW_NB15_training-set.csv` and **68.06% Attack** in `UNSW_NB15_testing-set.csv`. Attacks form the majority class in both files.
2. **Extreme Minority Attack Families**: Within the attack population, four classes represent extreme minority distributions: `Worms` (0.05% in train, 44 rows; 0.07% in test, 130 rows), `Shellcode` (0.46% in train), `Backdoor` (0.71% in train), and `Analysis` (0.82% in train).
3. **Partition Size Reversal**: The file named `UNSW_NB15_testing-set.csv` contains **175,341 records** (68.05% of the 257k total), whereas `UNSW_NB15_training-set.csv` contains only **82,332 records** (31.95%).
4. **Cross-Partition Duplicate Contamination**: 8,421 testing rows (4.80% of the test set) have exact feature and label duplicates in the training set. When examining features alone, 10,279 test rows (5.86%) share identical feature vectors with training records.
5. **High-Severity Target Leakage Risks**:
   - The sequential row index `id` has a Pearson correlation of **+0.7272** with the target label in the test set. Records are ordered in contiguous blocks of benign and attack traffic; an index threshold yields artificial classification accuracy without learning flow characteristics.
   - The Source Time-to-Live (`sttl`) feature reflects synthetic IXIA PerfectStorm simulator defaults (e.g. `sttl=254` represents 86.0% of all attacks, while `sttl=31` is 100% benign with 16,702 samples).
6. **Covariate Shift / Sign Inversion**: Network timing metrics (`tcprtt`, `synack`, `ackdat`) exhibit negative correlation with the target in the training set (-0.149), but invert to positive correlation in the test set (+0.082 to +0.097).

---

## 1. Dataset Dimensions & Schema Definition

The acquired benchmark files comprise two main partition CSV files, one comprehensive feature dictionary, and one event catalog.

### 1.1 Dimensional Summary

| File Name | File Size (Bytes) | Row Count | Column Count | Cryptographic Hash (SHA-256) |
| :--- | :--- | :--- | :--- | :--- |
| `UNSW_NB15_training-set.csv` | 15,380,800 | 82,332 | 45 | `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559` |
| `UNSW_NB15_testing-set.csv` | 32,293,018 | 175,341 | 45 | `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa` |
| **Combined Benchmark Total** | **47,673,818** | **257,673** | **45** | — |
| `NUSW-NB15_features.csv` | 4,044 | 49 (metadata) | 4 | `c55f19cceebb6360dc50f44f8a5f246ccefbcf8a6c604ac1ad46e643869cafce` |
| `UNSW-NB15_LIST_EVENTS.csv` | 4,639 | 94 (events) | 3 | `5b40f8128e2c87e691157c76debde04c328b0eb66bab97f1e7e17625f84497f2` |

Both partition files share an identical 45-column schema. There are zero column name or data type mismatches.

### 1.2 Breakdown of Columns (45 Total)
- **1 Identifier Column**: `id` (Sequential integer from 1 to 82,332 in train, 1 to 175,341 in test).
- **2 Target Ground-Truth Columns**:
  - `label`: Binary indicator (`0` = Normal / Benign, `1` = Attack / Malicious).
  - `attack_cat`: Multi-class string denoting attack taxonomy or `Normal`.
- **3 Categorical Feature Columns**:
  - `proto`: Transaction protocol (131 categories in train, 133 in test).
  - `service`: Application layer protocol (`-`, `dns`, `http`, `smtp`, `ftp`, `ftp-data`, `pop3`, `ssh`, `ssl`, `snmp`, `dhcp`, `radius`, `irc`).
  - `state`: Protocol connection state (`FIN`, `INT`, `CON`, `REQ`, `ACC`, `RST`, `CLO`, `no`, `PAR`, `URN`, `ECO`).
- **39 Numerical Flow Feature Columns**: Duration, packet counts, byte counts, flow rates, TTL values, window sizes, TCP round-trip metrics, inter-packet arrival times, and flow connection count aggregates.

*(Note: In the original 49-feature raw capture, features 1 to 4 were `srcip`, `sport`, `dstip`, and `dsport`. The authors omitted these four columns from the benchmark splits to prevent models from memorizing IP topologies).*

---

## 2. Missing Values, Sentinel Encoding & Duplicate Analysis

### 2.1 Explicit Missing Values (NaN / Null)
- **Training Set Nulls**: 0 cells (0.00%)
- **Testing Set Nulls**: 0 cells (0.00%)

There are no missing values represented as `NaN` or `None`.

### 2.2 Implicit Sentinel Values
- **Service Column Sentinel**: The value `'-'` in the `service` column denotes that no application-layer service was detected by Bro/Argus (frequently observed in pure layer-4 UDP, ICMP, or non-standard TCP ports).
  - Training Set: 47,153 occurrences (**57.27%** of flows).
  - Testing Set: 94,168 occurrences (**53.71%** of flows).
  - *Engineering Implication*: `'-'` is not missing data; it carries distinct semantic meaning as "unclassified / raw layer-4 traffic" and should be treated as an explicit category.

### 2.3 Duplicate Records Analysis

| Scope | Train Count | Train % | Test Count | Test % |
| :--- | :--- | :--- | :--- | :--- |
| Exact Duplicates (including `id`) | 0 | 0.00% | 0 | 0.00% |
| Internal Duplicates (excluding `id`) | 26,387 | **32.05%** | 67,601 | **38.55%** |
| Unique Flow Signatures | 55,945 | 67.95% | 107,740 | 61.45% |

### 2.4 Cross-Partition Overlap & Contamination
- **Cross-Partition Duplicate Patterns**: Exactly **940 unique flow feature vectors** appear in both `training-set.csv` and `testing-set.csv`.
- **Affected Testing Records**: These 940 patterns account for **8,421 rows** in the test set (4.80% of all test records) that have identical twins in the training set with matching labels.
- **Feature Vector Label Conflicts**: **414 distinct feature vectors** (affecting 1,758 total rows) appear with contradictory labels (labeled Benign in one instance and Attack in another). This reflects standard network ambiguity (e.g. benign traffic captured during an attack simulation window).

---

## 3. Class Imbalance & Target Distributions

### 3.1 Binary Label Distribution (`label`)

| Partition | Normal (0) Count | Normal (0) % | Attack (1) Count | Attack (1) % | Total Records |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train Set** | 37,000 | 44.94% | 45,332 | **55.06%** | 82,332 |
| **Test Set** | 56,000 | 31.94% | 119,341 | **68.06%** | 175,341 |
| **Combined** | 93,000 | 36.09% | 164,673 | **63.91%** | 257,673 |

![Binary Class Imbalance](file:///C:/Users/Chirag/Desktop/DataNova/research/visualizations/class_imbalance_binary.png)

### 3.2 Multiclass Attack Taxonomy Distribution (`attack_cat`)

| Category | Train Count | Train % | Test Count | Test % | Combined Count | Combined % | Severity Tier |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Normal** | 37,000 | 44.94% | 56,000 | 31.94% | 93,000 | 36.09% | Baseline |
| **Generic** | 18,871 | 22.92% | 40,000 | 22.81% | 58,871 | 22.85% | Majority Attack |
| **Exploits** | 11,132 | 13.52% | 33,393 | 19.04% | 44,525 | 17.28% | Majority Attack |
| **Fuzzers** | 6,062 | 7.36% | 18,184 | 10.37% | 24,246 | 9.41% | Mid-Tier |
| **DoS** | 4,089 | 4.97% | 12,264 | 6.99% | 16,353 | 6.35% | Mid-Tier |
| **Reconnaissance** | 3,496 | 4.25% | 10,491 | 5.98% | 13,987 | 5.43% | Mid-Tier |
| **Analysis** | 677 | 0.82% | 2,000 | 1.14% | 2,677 | 1.04% | **Minority (< 1.5%)** |
| **Backdoor** | 583 | 0.71% | 1,746 | 1.00% | 2,329 | 0.90% | **Minority (< 1.0%)** |
| **Shellcode** | 378 | 0.46% | 1,133 | 0.65% | 1,511 | 0.59% | **Minority (< 0.6%)** |
| **Worms** | 44 | **0.05%** | 130 | **0.07%** | 174 | **0.07%** | **Extreme Minority (< 0.1%)** |

![Multiclass Distribution](file:///C:/Users/Chirag/Desktop/DataNova/research/visualizations/attack_category_distribution.png)

![Minority Class Breakdown](file:///C:/Users/Chirag/Desktop/DataNova/research/visualizations/minority_class_breakdown.png)

### 3.3 Empirical Observations on Imbalance
1. **The ~3x Multiplier Pattern**: Across almost every attack class, the count in `testing-set.csv` is almost exactly three times the count in `training-set.csv` (e.g., DoS: $4,089 \times 3 = 12,267 \approx 12,264$; Exploits: $11,132 \times 3 = 33,396 \approx 33,393$; Worms: $44 \times 3 = 132 \approx 130$).
2. **The "Worms" Vulnerability**: With only 44 instances in the training set and 130 in the test set, any multi-class model evaluated on raw accuracy will completely ignore Worms without penalty. Multi-class cost-sensitive formulations and PR-AUC are mandatory.

---

## 4. Feature Correlations, Collinearity & Redundancy

![Feature Correlation Matrix](file:///C:/Users/Chirag/Desktop/DataNova/research/visualizations/feature_correlation_heatmap.png)

### 4.1 Highly Collinear Feature Pairs ($r > 0.90$)
Seventeen pairs of numerical features display extreme collinearity, introducing redundant degrees of freedom:

| Feature 1 | Feature 2 | Pearson Correlation ($r$) | Physical Interpretation |
| :--- | :--- | :--- | :--- |
| `dbytes` | `dloss` | 0.9971 | Destination bytes directly mirror packet loss |
| `sbytes` | `sloss` | 0.9950 | Source bytes directly mirror packet loss |
| `is_ftp_login` | `ct_ftp_cmd` | 0.9943 | Binary login flag is redundant with FTP command count |
| `dpkts` | `dloss` | 0.9815 | Destination packets and loss count |
| `ct_srv_src` | `ct_srv_dst` | 0.9778 | Service count source vs. destination |
| `dpkts` | `dbytes` | 0.9764 | Destination packet count vs. byte count |
| `spkts` | `sloss` | 0.9736 | Source packet count vs. loss count |
| `spkts` | `sbytes` | 0.9657 | Source packet count vs. byte count |
| `swin` | `dwin` | 0.9601 | Source window size vs. destination window size |
| `sinpkt` | `is_sm_ips_ports` | 0.9445 | Inter-packet time vs. identical IP/port indicator |
| `tcprtt` | `synack` | 0.9395 | Total round trip time vs. SYN-ACK handshake time |

*Engineering Recommendation*: Collinear pairs with $r > 0.98$ (`dbytes/dloss`, `sbytes/sloss`, `is_ftp_login/ct_ftp_cmd`) should be considered for feature pruning during feature engineering to reduce model complexity and eliminate redundant SHAP explanations.

---

## 5. Potential Data Leakage Risks & Synthetic Artifacts

### 5.1 Critical Risk 1: Sequential ID Target Leakage

![ID Target Leakage](file:///C:/Users/Chirag/Desktop/DataNova/research/visualizations/id_target_leakage.png)

- **Measured Correlation**:
  - Training set: $\text{Corr}(\text{id}, \text{label}) = -0.3875$
  - Testing set: $\text{Corr}(\text{id}, \text{label}) = \mathbf{+0.7272}$
- **Mechanism**: The dataset rows were not randomly shuffled prior to index assignment. In `UNSW_NB15_testing-set.csv`, rows 1 through 43,836 are **100.0% Benign**, whereas rows 131,506 through 175,341 are **100.0% Malicious**.
- **Mandatory Safeguard**: The `id` column must be strictly excluded from all training and inference feature sets.

### 5.2 Critical Risk 2: Synthetic Testbed TTL Artifacts (`sttl`)

![Synthetic TTL Artifact](file:///C:/Users/Chirag/Desktop/DataNova/research/visualizations/synthetic_artifact_leakage_sttl.png)

- **Source TTL (`sttl`)**:
  - When `sttl == 31`, 100.0% of records are **Benign** (16,702 samples, 0 attacks).
  - When `sttl == 254`, **39,000 out of 45,332 attacks** occur (86.03% of all malicious traffic).
- **Mechanism**: The IXIA PerfectStorm generator configured simulated attack nodes with distinct default operating system TTLs (254 and 62) compared to normal nodes (31 and 60).
- **Research Risk**: Classifiers trained naively on `sttl` learn the testbed's operating system default settings rather than genuine adversarial payload or behavioral dynamics.

### 5.3 Critical Risk 3: Unseen Categories Across Partitions
- **Protocol (`proto`)**: Test contains `icmp` and `rtp`, which never appear in the training partition.
- **State (`state`)**: Test contains `no`, `PAR`, `URN`, `ECO`, while Train contains `CLO` and `ACC`.
- **Mandatory Safeguard**: Categorical encoders must be configured with `handle_unknown='ignore'` to avoid crashing on novel protocols.

---

## 6. Train/Test Partition Suitability Evaluation

### 6.1 Assessment of the Author Predefined Split
The authors' predefined split (`training-set.csv` [82k] and `testing-set.csv` [175k]) has significant scientific flaws if used naively:
1. **Prior Probability Shift**: The proportion of attacks increases from 55.06% in train to 68.06% in test ($\Delta = +13.0\%$).
2. **Data Leakage via Duplication**: 8,421 exact duplicate rows span between the two files.
3. **Inverted Sizing**: Standard ML practice trains on the larger partition (~70-80%) and evaluates on the smaller partition (~20-30%).

### 6.2 Recommended Strategy for Phase 2
- **Option A (Benchmark Baseline)**: Retain the author's split for benchmark comparison with published literature, but explicitly drop `id`, deduplicate cross-partition records, and report PR-AUC alongside standard metrics.
- **Option B (Scientifically Pure Protocol)**: Combine the 257k benchmark pool, remove duplicate flow patterns, and perform a stratified 70/15/15 train/val/test split with zero cross-contamination.

---

## 7. Limitations & Research Boundaries

1. **Synthetic Generation Artifacts**: As demonstrated by `sttl` and `ct_state_ttl`, certain features reflect simulator configuration rather than attack behavior.
2. **No Zero-Day Detection Guarantee**: The dataset contains 9 known attack families generated between 2015 and 2016. The model cannot be assumed to generalize to zero-day vulnerabilities without explicit out-of-distribution evaluation.
3. **NetFlow Aggregations vs. Raw Packets**: The benchmark CSVs represent pre-computed flow summary statistics from Argus and Bro-IDS, not raw packet byte streams.

---

## 8. Specific Recommendations for Phase 2 (Preprocessing & Training)

1. **Immediate Column Exclusions**:
   - `id`: Drop completely to eliminate the +0.7272 sequential leakage.
2. **Collinearity Remediation**:
   - Prune redundant mirror pairs (`dloss` / `dbytes`, `sloss` / `sbytes`, `is_ftp_login` / `ct_ftp_cmd`).
3. **Categorical Encoding**:
   - Use `OneHotEncoder(handle_unknown='ignore')` to safely ingest novel protocols (`icmp`, `rtp`, `PAR`, `URN`).
4. **Imbalance Mitigation**:
   - Formulate cost-sensitive loss with high penalties for false negatives ($C_{\text{FN}} = 20 \times C_{\text{FP}}$).
   - Evaluate training-only SMOTE specifically for extreme minority classes (`Worms`, `Shellcode`, `Backdoor`).
   - Strictly prohibit synthetic resampling on validation or test sets.
5. **Evaluation Metric Governance**:
   - Ban accuracy as an evaluation metric.
   - Use PR-AUC (Precision-Recall Area Under Curve), F1-macro, and expected cost loss.
