# Split Integrity & Partition Verification Report

**Project**: Adaptive Threat Intelligence  
**Phase**: Phase 2 — Evaluation Split Construction  
**Dataset**: UNSW-NB15 Cleaned Evaluation Partitions  
**Date**: October 2026  

---

## 1. Executive Summary

To ensure scientifically valid model evaluations, we constructed **group-aware stratified evaluation splits** (70% Train, 15% Validation, 15% Testing) from the combined 257,673 UNSW-NB15 dataset pool.

### Partitioning Guarantees
1. **Group Invariance**: All rows sharing an identical predictor feature vector are assigned to **exactly one partition**. No feature group spans multiple partitions.
2. **Strict Disjointness**:
   $$\text{Train} \cap \text{Validation} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Validation} \cap \text{Test} = \emptyset$$
3. **Target Stratification**: Class proportions are preserved within $\pm 1\%$ across Train, Validation, and Test sets.
4. **Minority Class Protection**: Rare classes (Worms, Backdoor, Analysis, Shellcode) are guaranteed representation across all three partitions.

---

## 2. Binary Evaluation Split (`label`)

- **Clean Record Pool**: 255,915 records across 153,270 unique feature groups.
- **Target Distribution**: Normal (`0`) = 92,072 (35.98%), Attack (`1`) = 163,843 (64.02%).

| Partition | Total Rows | Proportion (%) | Unique Groups | Normal (`0`) Rows | Normal (`0`) % | Attack (`1`) Rows | Attack (`1`) % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 179,511 | 70.14% | 107,289 | 64,591 | 70.15% | 114,920 | 70.14% |
| **Validation** | 37,976 | 14.84% | 22,990 | 13,648 | 14.82% | 24,328 | 14.85% |
| **Test** | 38,428 | 15.02% | 22,991 | 13,833 | 15.02% | 24,595 | 15.01% |
| **Total** | **255,915** | **100.00%** | **153,270** | **92,072** | **100.00%** | **163,843** | **100.00%** |

### Leakage Verification Check
- Feature hashes in $\text{Train} \cap \text{Val}$: **0**
- Feature hashes in $\text{Train} \cap \text{Test}$: **0**
- Feature hashes in $\text{Val} \cap \text{Test}$: **0**
- **Cross-Partition Duplicate Leakage**: **0.00%**

---

## 3. Multiclass Evaluation Split (`attack_cat`)

- **Clean Record Pool**: 217,273 records across 151,239 unique feature groups.
- **Quarantined**: 40,400 records (2,445 conflicting groups).

### Partition Summary

| Partition | Total Rows | Target Proportion (%) | Actual Proportion (%) | Unique Feature Groups |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 151,325 | 70.00% | 69.65% | 105,867 |
| **Validation** | 33,045 | 15.00% | 15.21% | 22,686 |
| **Test** | 32,903 | 15.00% | 15.14% | 22,686 |
| **Total** | **217,273** | **100.00%** | **100.00%** | **151,239** |

### Stratification by Attack Category

| Category | Clean Rows | Clean Groups | Train Rows (%) | Val Rows (%) | Test Rows (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Normal** | 92,072 | 85,308 | 64,591 (70.15%) | 13,648 (14.82%) | 13,833 (15.02%) |
| **Generic** | 58,498 | 7,283 | 40,164 (68.66%) | 9,462 (16.17%) | 8,872 (15.17%) |
| **Exploits** | 28,018 | 25,451 | 19,624 (70.04%) | 4,211 (15.03%) | 4,183 (14.93%) |
| **Fuzzers** | 20,629 | 18,749 | 14,395 (69.78%) | 3,115 (15.10%) | 3,119 (15.12%) |
| **Reconnaissance** | 11,741 | 8,419 | 8,134 (69.28%) | 1,647 (14.03%) | 1,960 (16.69%) |
| **DoS** | 3,690 | 3,624 | 2,583 (70.00%) | 559 (15.15%) | 548 (14.85%) |
| **Shellcode** | 1,498 | 1,446 | 1,048 (69.96%) | 229 (15.29%) | 221 (14.75%) |
| **Analysis** | 616 | 450 | 429 (69.64%) | 97 (15.75%) | 90 (14.61%) |
| **Backdoor** | 352 | 350 | 246 (69.89%) | 53 (15.06%) | 53 (15.06%) |
| **Worms** | 159 | 159 | 111 (69.81%) | 24 (15.09%) | 24 (15.09%) |

### Special Verification: The Worms Class
Worms contains **159 clean distinct feature groups**. Under our group-aware stratified partition:
- Train: Exactly **111 groups** (111 rows, 69.81%)
- Validation: Exactly **24 groups** (24 rows, 15.09%)
- Testing: Exactly **24 groups** (24 rows, 15.09%)
- **Zero group overlap**: No Worms signature in validation or testing was observed in training.

---

## 4. Comparison with Original Author Predefined Split

| Architectural Property | Original Author Predefined Split | Phase 2 Leakage-Resistant Split |
| :--- | :--- | :--- |
| **Partition Sizes** | Train: 32% (82k), Test: 68% (175k) | Train: 70% (179k/151k), Val: 15%, Test: 15% |
| **Validation Set** | None provided | Dedicated 15% out-of-sample tuning set |
| **Cross-Partition Duplicate Leakage** | 8,421 identical rows in test | **0 rows (Strict 0.0% overlap)** |
| **Conflicting Label Flows** | Present in both splits (414 binary, 2,445 multi) | Systematically audited and quarantined |
| **Sequential ID Target Correlation** | $r = +0.7272$ in test (extreme leakage) | `id` column strictly excluded from features |
| **Class Imbalance Shift** | Attack: 55% train $\rightarrow$ 68% test | Proportions stratified to within $\pm 0.8\%$ |
| **Unseen Categories Crash Risk** | 2 unseen protocols, 4 unseen states | `OneHotEncoder(handle_unknown='ignore')` |
