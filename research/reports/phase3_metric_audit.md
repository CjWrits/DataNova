# Phase 3 Metric and Reporting Audit Report

**Audit Date:** 2026-10-30  
**Target Baseline Commit:** `2cb3afc`  
**Purpose:** Mandatory pre-Phase 4 audit addressing ten specific review items regarding metric interpretations, confusion matrix tracing, label alignments, latency benchmarking conditions, and confidence intervals.

---

## 1. Tracing the 1,249 Normal-Class False Positives

### Audit Finding
In the initial Phase 3 report summary, a sentence stated:
> *"Out of 1,249 false positives on benign traffic, 1,035 flows were misclassified as Fuzzers."*

**This statement contained a row/column transposition error in narrative reporting.**

### Confusion Matrix Dissection (`multiclass_primary_xgboost` on Test Partition)
In the 10x10 multiclass confusion matrix $C$, where $C_{i, j}$ represents the count of instances whose ground truth is class $i$ and predicted class is $j$:
- **Class Index 0 = `Normal`**
- **Row 0 ($C_{0, :}$)** represents **True Normal flows (Total Support = 13,833)**.
- **Column 0 ($C_{:, 0}$)** represents **Flows predicted by the model as `Normal` (Total = 14,385)**.

```
Row 0 (True Normal Flows):
  C[0, 0] (Predicted Normal - True Negative for Attack): 13,136
  C[0, 1] (Predicted Generic):                                2
  C[0, 2] (Predicted Exploits):                              80
  C[0, 3] (Predicted Fuzzers):                              580
  C[0, 4] (Predicted DoS):                                    6
  C[0, 5] (Predicted Reconnaissance):                         5
  C[0, 6] (Predicted Analysis):                              11
  C[0, 7] (Predicted Backdoor):                               0
  C[0, 8] (Predicted Shellcode):                             13
  C[0, 9] (Predicted Worms):                                  0
  Total Benign Flows Flagged as Attacks (False Alarms):     697 (5.04% false alarm rate)

Column 0 (Flows Predicted as Normal by Model):
  C[0, 0] (True Normal correctly predicted):              13,136
  C[1, 0] (Generic attack predicted as Normal):               4
  C[2, 0] (Exploits attack predicted as Normal):            108
  C[3, 0] (Fuzzers attack predicted as Normal):           1,035
  C[4, 0] (DoS attack predicted as Normal):                  16
  C[5, 0] (Reconnaissance attack predicted as Normal):        1
  C[6, 0] (Analysis attack predicted as Normal):              64
  C[7, 0] (Backdoor attack predicted as Normal):              3
  C[8, 0] (Shellcode attack predicted as Normal):            18
  C[9, 0] (Worms attack predicted as Normal):                 0
  Total Non-Normal Attacks Predicted as Normal (Misses):   1,249
```

### Exact Correction
1. **The 1,249 flows are ATTACKS falsely predicted as Normal.** In one-vs-rest metrics for the `Normal` class, these are classified as "False Positives for Normal" ($FP_{\text{Normal}} = \sum_{i \ne 0} C_{i, 0} = 1,249$). In operational network security terminology, these are **1,249 missed malicious flows** (False Negatives of the NIDS).
2. **The 1,035 flows are true FUZZERS attacks that escaped detection** by being predicted as benign `Normal` traffic ($C_{3, 0} = 1,035$). Fuzzers represent $1,035 / 1,249 = 82.87\%$ of all attacks that masqueraded as benign traffic.
3. **The actual false alarms on benign traffic total 697 flows**, of which **580 benign flows were incorrectly classified as Fuzzers** ($C_{0, 3} = 580$).

---

## 2. Verification of the 1,035 Fuzzers Cell

Confirmed:
- $C_{3, 0} = 1,035$ represents true `Fuzzers` attacks classified as `Normal`.
- True `Fuzzers` support in the test set is 3,119 flows. Of these, 2,029 were correctly detected, 1,035 were misclassified as `Normal`, 43 as `Exploits`, 5 as `Shellcode`, 3 as `DoS`, 2 as `Reconnaissance`, 1 as `Generic`, and 1 as `Worms`.
- **Fuzzers attack recall was 65.05%**, dragged down almost exclusively by the 1,035 instances predicted as benign `Normal`.

---

## 3. Verification of Binary False Positive Rate (FPR)

On the binary test set (38,428 flows):
- True Normal ($y = 0$): 13,833
- True Attack ($y = 1$): 24,595
- Confusion Matrix:
  - $\text{True Negatives } (TN) = 12,954$
  - $\text{False Positives } (FP) = 879$
  - $\text{False Negatives } (FN) = 1,102$
  - $\text{True Positives } (TP) = 23,493$

$$\text{FPR} = \frac{FP}{FP + TN} = \frac{879}{879 + 12954} = \frac{879}{13833} = 0.063544 \quad (\mathbf{6.35\%})$$
$$\text{Specificity} = \frac{TN}{TN + FP} = \frac{12954}{13833} = 0.936456 \quad (\mathbf{93.65\%})$$

The binary FPR calculation of 6.35% is mathematically exact and verified.

---

## 4. Multiclass Macro Average Precision (PR-AUC) Verification

We independently verified the implementation of `macro_pr_auc` against Scikit-Learn's canonical `average_precision_score(y_bin, probs, average="macro")`:
- Scikit-Learn Canonical Macro AP: `0.7878708690858621`
- Phase 3 Reported Macro PR-AUC: `0.787871` (rounded to 0.7879)
- Match: **Exact to 6 decimal places**.

Each column $c \in \{0, \dots, 9\}$ of the probability matrix strictly aligns with class $c$ in the binarized one-hot ground truth matrix.

---

## 5. Verification of Model Classes, Probabilities, and Metric Labels

- Model `classes_` attribute: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
- Target mapping:
  - 0: Normal
  - 1: Generic
  - 2: Exploits
  - 3: Fuzzers
  - 4: DoS
  - 5: Reconnaissance
  - 6: Analysis
  - 7: Backdoor
  - 8: Shellcode
  - 9: Worms
- Metric calculations utilize this exact mapping across all functions. No permutation or misalignment exists.

---

## 6. Test Set Isolation Verification

Inspection of [`scripts/run_baseline_experiments.py`](file:///C:/Users/Chirag/Desktop/DataNova/scripts/run_baseline_experiments.py) lines 333–420 confirms:
1. All 12 candidate models were trained strictly on `train` and evaluated strictly on `val`.
2. Model selection (`select_best_model_for_configuration`) operated exclusively on validation F1, validation PR-AUC, and validation minority recall.
3. The 4 winning models were frozen and serialized to disk (`joblib.dump`).
4. Test evaluation (`evaluate_frozen_model_on_test`) was executed only after freezing.
5. No test metrics were inspected or used during any decision gate.

---

## 7. Inference Latency Benchmark Methodology Audit

The Phase 3 report reported an inference latency of 0.0037 ms/sample for binary and 0.0340 ms/sample for multiclass. 

### Methodological Limitations Identified
1. **Batch vs. Single-Sample Execution:** The measurement was conducted in a single vectorized batch across 38,428 samples (`model.predict(X_test_trans)`). It does not reflect streaming, single-flow per-sample arrival latency.
2. **Preprocessing Excluded from Validation Timing:** In the validation benchmark, preprocessing time was recorded separately from model evaluation. In the test evaluation, preprocessing was included in batch transformation.
3. **Hardware & System State:** Measurements were taken on a single Windows 11 workstation with standard background processes running. No multi-run averaging or warm-up passes were performed.
4. **Scope Clarification:** Tabular NIDS flow inference operates on **summarized network flows**, not raw packets. A flow classifier is not an inline packet-inspection engine.

### Operational Reclassification
These latency numbers must be characterized as **batch-mode algorithmic throughput estimates under benchmark conditions**, rather than proof of real-time line-rate packet inspection.

---

## 8. 95% Wilson Confidence Interval for Worms Recall

Given:
- Total Test Set Worms Support ($n$): 24
- True Positives Detected ($x$): 13
- Observed Test Recall ($\hat{p} = x / n$): 54.17%

Applying the Wilson score interval formula with continuity parameter $z = 1.95996$:

$$\hat{p}_{\text{Wilson}} = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

- **Calculated 95% Wilson Confidence Interval:** **[35.07%, 72.11%]**

### Statistical Context
With only 24 test instances, a single misclassification swings the recall estimate by **4.17 percentage points**. While detecting 13/24 is promising for an unweighted baseline, the true underlying population recall may lie anywhere between 35.1% and 72.1%. Claims regarding Worms detection must reflect this statistical uncertainty.

---

## 9. Re-Framing the `sttl` Ablation Study

- **Original Claim:** Recommended No-STTL as the "primary candidate for production deployment."
- **Audit Correction:** We have not evaluated the model on external real-world network captures (e.g., CIC-IDS2017, CTU-13) to empirically prove that removing `sttl` enhances out-of-distribution transferability.
- **Appropriate Classification:** The No-STTL schema must be described as a **robustness-oriented experimental alternative designed to mitigate synthetic testbed artifacts**, not as a proven production-hardened system.

---

## 10. Correction of Real-Time Packet Inspection Claims

All statements equating tabular flow classification with "real-time inline packet inspection" are retracted. Flow-level intrusion detection requires stateful flow reconstruction, timeout management, and bi-directional flow feature extraction before any ML model receives an input vector. We report model inference throughput at the flow level only.

---

## Conclusion of Audit
No data leakage, label misalignment, or calculation errors exist in the underlying code. The historical baseline records in commit `2cb3afc` remain valid. Phase 4 development proceeds on this verified foundation.
