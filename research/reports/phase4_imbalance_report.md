# Phase 4 Class Imbalance Mitigation & Operational Trade-off Report

**Project:** Adaptive Threat Intelligence — DataNova Global Data-thon 2026  
**Phase:** 4 — Imbalance-Aware Modeling and Feature Engineering  
**Date:** 2026-11-12  
**Baseline Reference Commit:** `2cb3afc`  

---

## 1. Executive Summary

Intrusion detection systems face an intrinsic statistical challenge: extreme class imbalance. In real-world telemetry and benchmark corpora like UNSW-NB15, benign traffic (`Normal`) and ubiquitous attacks (`Generic`, `Exploits`) account for over 85% of flows, while high-severity threats (`Worms`, `Analysis`, `Backdoor`, `Shellcode`) comprise fewer than 1.5% of total records.

In Phase 3, unweighted baseline models established solid binary detection (95.95% Test F1) and multiclass Macro F1 (72.07%), but exhibited pronounced vulnerabilities on severely underrepresented attack families:
- **Worms**: 54.17% test recall (13/24 detected; 95% Wilson CI: [35.07%, 72.11%])
- **DoS**: 32.48% test recall (178/548 detected)
- **Analysis**: 23.33% test recall (21/90 detected)

The objective of Phase 4 is not merely to optimize a single minority-class recall number at all costs. An NIDS that detects 100% of Worms but flags 25% of legitimate enterprise traffic produces severe analyst fatigue and operational denial-of-service. This report rigorously analyzes five distinct imbalance mitigation paradigms, evaluates their performance across all 10 classes, and quantifies the exact trade-off between minority attack rescue and benign false alarms.

---

## 2. Experimental Methodology & Rigorous Safeguards

All imbalance experiments strictly adhere to the leakage-free protocol established in Phase 2:
1. **Split Invariance:** The verified stratified partitions (`train`: 151,325 rows; `val`: 33,045 rows; `test`: 32,903 rows for multiclass) are identical to Phase 2 and 3. No re-splitting or test-set snooping was permitted.
2. **Training-Only Resampling:** All resampling strategies (Random Oversampling and SMOTENC) are applied **exclusively to the training partition**. Validation and test partitions are never oversampled, duplicated, or synthetically altered.
3. **Pre-Encoding Synthesis for SMOTENC:** SMOTENC was evaluated strictly on the mixed continuous and categorical feature representation prior to one-hot encoding, with explicit indexing of the categorical features (`proto`, `service`, `state`). Standard SMOTE was never applied to one-hot dummy variables, preventing impossible interpolations.
4. **Independent Weight Derivation:** Sample weights are computed exclusively from training label frequencies:
   $$w_c = f(N_{\text{train}}, K, N_{c, \text{train}})$$
   Validation and test labels never influence weight calculations.
5. **Validation-Gated Selection:** All candidate evaluations and model comparisons were executed strictly on the validation set. Winning candidates were frozen and serialized prior to unsealing the test partition.

---

## 3. Mathematical Formulations of Imbalance Paradigms

### A. Cost-Sensitive Inverse Class Frequency (Balanced)
Assigns a penalty proportional to the reciprocal of class support:
$$w_c = \min\left(\frac{N}{K \cdot N_c},\, W_{\max}\right)$$
where $N = 151,325$ is total training support, $K = 10$ is the number of classes, $N_c$ is class support, and $W_{\max} = 50.0$ prevents gradient instability.
- Class weights range from $0.234$ (`Normal`: $N_c = 64,591$) to $50.00$ (`Worms`: $N_c = 111$).

### B. Cost-Sensitive Smoothed Square-Root Weighting (Sqrt-Balanced)
Dampens the extreme penalty skew using a sub-linear transformation:
$$w_c = \sqrt{\frac{N}{K \cdot N_c}}$$
- Class weights range from $0.484$ (`Normal`) to $11.676$ (`Worms`). This prevents minority penalties from completely overwhelming dominant decision boundaries while still supplying an order-of-magnitude gradient boost to rare attack classes.

### C. Targeted Random Oversampling
Minority classes with training count below a minimum floor ($N_c < 2,000$) are randomly replicated up to 2,000 samples.
- Affected classes: `Analysis` ($429 \to 2,000$), `Backdoor` ($246 \to 2,000$), `Shellcode` ($1,048 \to 2,000$), and `Worms` ($111 \to 2,000$).
- Total samples added: 6,166 (New training size: 157,491).
- Majority classes (`Normal`, `Generic`, `Exploits`, `Fuzzers`, `DoS`, `Reconnaissance`) are left completely untouched, preserving natural background traffic density.

### D. Pre-Encoding SMOTENC
Applies Synthetic Minority Over-sampling Technique for Nominal and Continuous features (SMOTENC) to the mixed raw feature matrix prior to categorical encoding.
- Categorical features: `proto`, `service`, `state`.
- Continuous features: 39 numerical telemetry columns.
- Interpolates continuous features along $k=5$ nearest Euclidean neighbors while taking the modal value for nominal features.
- Support floor: 2,000 samples for classes with $N_c < 2,000$.

### E. Balanced Random Forest
Constructs 200 decision trees where each bootstrap sample is balanced by randomly downsampling majority classes to match minority class support per tree.

---

## 4. Quantitative Validation Comparison Leaderboard

All 10 experimental candidate configurations evaluated on the validation partition (33,045 flows; 13,648 true Normal flows):

| Candidate ID | Paradigm / Configuration | Schema | Val Macro F1 | Val Macro Rec | Val PR-AUC | Worms Rec | Analysis Rec | DoS Rec | Benign False Alarms | Benign FAR (%) | Fit Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `mc_cand1_baseline_unweighted` | Phase 3 Baseline (Unweighted XGBoost) | Primary | 0.7015 | 0.6619 | 0.7776 | 0.5000 | 0.2577 | 0.3184 | 669 | **4.90%** | 24.50s |
| `mc_cand2_cost_sensitive_balanced` | Full Inverse Weighting | Primary | 0.7052 | 0.8000 | 0.7770 | **0.8333** | **0.6804** | **0.5814** | 2,799 | 20.51% | 25.29s |
| `mc_cand3_cost_sensitive_sqrt` | Sqrt-Smoothed Weighting | Primary | 0.7286 | 0.7464 | 0.7713 | 0.7083 | 0.4639 | 0.4311 | 1,534 | 11.24% | 22.97s |
| `mc_cand4_random_oversampling` | Targeted Random Oversampling (Floor=2k) | Primary | 0.7349 | 0.7058 | 0.7825 | 0.7083 | 0.3196 | 0.3488 | 713 | **5.22%** | 23.23s |
| `mc_cand5_smotenc` | Pre-Encoding SMOTENC (Floor=2k) | Primary | 0.7136 | 0.6978 | 0.7600 | 0.6250 | 0.3608 | 0.3363 | 693 | **5.08%** | 25.31s |
| `mc_cand6_balanced_random_forest` | Balanced Random Forest (Downsampled) | Primary | 0.5362 | **0.8162** | 0.6483 | **1.0000** | **0.9794** | 0.5045 | 3,896 | 28.55% | 5.25s |
| `mc_cand7_feature_engineered_unweighted` | Feature Engineering Only (Unweighted) | Primary | 0.7200 | 0.6726 | 0.7798 | 0.5833 | 0.2268 | 0.3542 | 659 | **4.83%** | 23.73s |
| `mc_cand8_feature_engineered_cost_sensitive` | **FE + Sqrt-Smoothed Weighting (Winner)** | Primary | **0.7428** | **0.7567** | **0.7857** | **0.7917** | **0.4536** | **0.4508** | 1,550 | 11.36% | 27.13s |
| `mc_cand9_feature_engineered_smotenc` | FE + SMOTENC + XGBoost | Primary | 0.7189 | 0.6892 | 0.7658 | 0.5833 | 0.3093 | 0.3470 | 689 | **5.05%** | 41.34s |
| `mc_cand10_no_sttl_fe_cost_sensitive` | **Robustness Track: No-STTL + FE + Sqrt** | No-STTL | **0.7257** | **0.7355** | **0.7684** | 0.6667 | 0.4124 | 0.3953 | 1,541 | 11.29% | 33.78s |

---

## 5. In-Depth Operational Trade-Off Analysis

### The Benign False Alarm vs. Minority Recall Dilemma
The central finding of Phase 4 is that **minority attack recall cannot be evaluated in isolation from benign false alarms**.

1. **The Cost of Aggressive Weighting (`mc_cand2` and `mc_cand6`):**
   - Candidate 2 (Full Balanced Weights) achieves impressive individual recalls: Worms 83.33%, Analysis 68.04%, DoS 58.14%.
   - Candidate 6 (Balanced RF) detects 100% of Worms and 97.94% of Analysis.
   - **However, both fail operationally.** Candidate 2 flags 2,799 benign flows (20.51% false alarm rate), and Candidate 6 flags 3,896 benign flows (28.55% false alarm rate).
   - In a production Security Operations Center (SOC) processing millions of flows daily, an alarm rate of 20–28% would overwhelm security analysts with hundreds of thousands of false alarms daily.
   - Candidate 6 suffers a catastrophic drop in Macro F1 (0.5362) because downsampling majority classes throws away natural negative variance, causing benign web requests to be misclassified as attacks.

2. **The Sqrt-Smoothed Compromise (`mc_cand3` and `mc_cand8`):**
   - By dampening the weights to $\sqrt{N / (K \cdot N_c)}$, the penalty ratio between Worms and Normal is moderated from $214:1$ down to $24:1$.
   - This keeps the benign false alarm rate at **11.24% – 11.36%** while still achieving substantial minority detection: Worms 79.17%, Analysis 45.36%, DoS 45.08%, and pushing overall Macro F1 to a campaign-high of **0.7428**.

3. **The Conservative Low-False-Alarm Profile (`mc_cand4` and `mc_cand5`):**
   - If an organization's operational constraint strictly mandates keeping false alarms below 5.5%, **Targeted Random Oversampling (`mc_cand4`) and Pre-Encoding SMOTENC (`mc_cand5`) provide the superior solution**.
   - Random Oversampling lifts Worms recall to 70.83% and DoS to 34.88% with only **713 false alarms (5.22%)**, barely 44 more than the unweighted baseline (669 false alarms, 4.90%).
   - SMOTENC lifts Analysis recall to 36.08% with **693 false alarms (5.08%)**.
   - Because oversampling only raises minority classes to a modest floor of 2,000 samples, the dominant decision boundary between Normal and major attack classes is preserved.

---

## 6. Conclusion and Selection Recommendation

For environments prioritizing comprehensive threat hunting and automated quarantine, **Candidate 8 (`mc_cand8_feature_engineered_cost_sensitive`)** delivers the optimal macro balance (Macro F1 = 0.7428, Macro Recall = 0.7567, PR-AUC = 0.7857).

For environments with constrained triage capacity where analyst workload is paramount, **Candidate 4 (Targeted Random Oversampling)** serves as the recommended low-overhead alternative, maintaining a 5.22% benign false alarm rate.
