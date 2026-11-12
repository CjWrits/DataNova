# Phase 4 Final Metric & Implementation Audit Report

**Project:** Adaptive Threat Intelligence — DataNova Global Data-thon 2026  
**Audit Target Commit:** `67068b056599f19cbac9ebe2bbf6a0d6a406ab0b`  
**Date:** 2026-11-12  
**Purpose:** Pre-Phase 5 audit addressing four reviewer-identified items regarding validation vs. test provenance, analyst fatigue claims, threshold discretization, cost-function derivation, pytest warnings, and bundle serialization integrity.

---

## 1. Provenance of Candidate 4 and Candidate 9 False-Alarm Rates

### Audit Finding
In the narrative discussion comparing operational profiles, Candidate 4 (Targeted Random Oversampling) and Candidate 9 (FE + SMOTENC) were described as having benign false alarm rates of 5.22% and 5.05%.

### Provenance Verification
1. **Strict Validation-Only Status:** Neither Candidate 4 nor Candidate 9 was selected for frozen evaluation on the unsealed test partition.
2. The selection decision gate strictly selected **Candidate 8** as the single primary multiclass winner based on highest validation Macro F1 (0.7428) and balanced recall.
3. Therefore, the figures:
   - **Candidate 4:** 713 false alarms / 13,648 benign flows = **5.22%**
   - **Candidate 9:** 689 false alarms / 13,648 benign flows = **5.05%**
   are **validation partition metrics only**.
4. **Correction:** All documentation and executive summaries must clearly designate C4 and C9 as *validation-explored candidate profiles*, not confirmed test operating points. The only primary test-confirmed profile is Candidate 8 (11.79% test false alarm rate; 1,631 / 13,833 benign test flows).

---

## 2. Correction of the "0 Analyst Fatigue Penalty" Claim

### Audit Finding
A summary statement claimed that conservative resampling profiles (C4 and C9) incurred "0 analyst fatigue penalty" relative to the unweighted baseline.

### Quantitative Audit
- Baseline validation false alarms: **669 flows (4.90%)**
- Candidate 5 (Pre-Encoding SMOTENC): **693 flows (5.08%)** $\to$ $+24$ false alarms ($+0.18\%$)
- Candidate 9 (FE + SMOTENC): **689 flows (5.05%)** $\to$ $+20$ false alarms ($+0.15\%$)
- Candidate 4 (Random Oversampling): **713 flows (5.22%)** $\to$ $+44$ false alarms ($+0.32\%$)

### Correction
The claim of "0 penalty" is retracted as mathematically inaccurate. While C4, C5, and C9 avoid the large $+6.46\%$ validation false alarm surge of Candidate 8 (1,550 false alarms; 11.36%), they still produce 20 to 44 additional false alarms across 13,648 validation flows. The penalty is modest ($+0.15\%$ to $+0.32\%$), but non-zero.

---

## 3. Investigation of the Binary Default Threshold (0.4951 vs. 0.5000)

### Audit Finding
In the binary threshold calibration table, the default operating point was listed at $\theta = 0.4951$ rather than the conventional theoretical default of $\theta = 0.5000$.

### Root Cause Analysis in `ml/evaluation/thresholding.py`
Inspection of `sweep_binary_thresholds`:
```python
thresholds = np.linspace(0.01, 0.99, steps)  # steps = 100
...
pt_default = min(points, key=lambda p: abs(p.threshold - 0.50))
```
- For `np.linspace(0.01, 0.99, 100)`, the step size is:
  $$\Delta = \frac{0.99 - 0.01}{99} = \frac{0.98}{99} \approx 0.009898989899$$
- The 49th evaluation node is:
  $$\theta_{49} = 0.01 + 49 \cdot \frac{0.98}{99} \approx 0.4950505 \implies \mathbf{0.4951}$$
- The 50th evaluation node is:
  $$\theta_{50} = 0.01 + 50 \cdot \frac{0.98}{99} \approx 0.5049495 \implies \mathbf{0.5049}$$
- The distance to 0.5000 is identical:
  $$|0.49505 - 0.50000| = 0.00495 \quad \text{vs.} \quad |0.50495 - 0.50000| = 0.00495$$
- Python's `min` selected the first minimum encountered in traversal order ($\theta_{49} = 0.4951$).

### Resolution
This is a **pure grid discretization artifact** resulting from evaluating an even number of steps (100) across $[0.01, 0.99]$. It is not a calibrated parameter shift. We have updated `sweep_binary_thresholds` to ensure exact $\theta = 0.5000$ is explicitly included in the evaluation grid.

---

## 4. Documentation of the 5:1 Validation Cost Function

### Objective Function
For security environments where missing an active intrusion is significantly more detrimental than investigating a false alarm, the binary threshold was optimized using an empirical cost function strictly on validation predictions:
$$\text{Cost}(\theta) = \text{FP}_{\text{val}}(\theta) + C_{\text{ratio}} \cdot \text{FN}_{\text{val}}(\theta)$$
where $C_{\text{ratio}} = 5.0$.

### Optimization & Provenance Protocol
1. **Validation Sweep:** Evaluated across all grid thresholds on the validation set ($N_{\text{val}} = 37,976$).
2. **Minimum Cost Location:** The minimum cost occurred at **$\theta^* = 0.1783$**:
   $$\text{FP}_{\text{val}} = 3,152, \quad \text{FN}_{\text{val}} = 79 \implies \text{Cost} = 3,152 + 5.0 \cdot (79) = \mathbf{3,547}$$
   (At default $\theta = 0.50$, Validation Cost $= 798 + 5.0 \cdot 1,091 = 6,253$).
3. **Freezing:** The threshold $\theta^* = 0.1783$ was frozen prior to unsealing the test partition.
4. **Test Partition Application:** Applying $\theta^* = 0.1783$ to test probabilities ($N_{\text{test}} = 38,428$) yielded:
   $$\text{FP}_{\text{test}} = 3,205, \quad \text{FN}_{\text{test}} = 100 \implies \mathbf{99.59\% \text{ Test Recall}}$$
   Missed attacks on test were reduced from 1,092 down to 100 (a $90.8\%$ reduction in missed intrusions).

---

## 5. Investigation of Pytest Warning

### Warning Identified
```text
UndefinedMetricWarning: Only one class is present in y_true. ROC AUC score is not defined in that case.
```
- **Triggering Test:** `tests/unit/test_imbalance_and_features.py::test_multiclass_prior_calibration`
- **Root Cause:** The test created a synthetic 2-sample ground truth `y_true = np.array([0, 9])` to test prior multiplier shifting. `compute_multiclass_metrics` invoked `roc_auc_score(..., multi_class='ovr')` across all 10 classes. Because classes 1 through 8 had zero samples in `y_true`, scikit-learn issued a standard metric warning for the missing classes.
- **Assessment:** The warning was completely harmless to model logic. However, the test has been updated to provide a 10-class synthetic fixture so that the test suite runs with **zero warnings**.

---

## 6. Verification of Serialized Model Bundles

Inspection of all three Phase 4 bundles in `artifacts/models/phase4/`:
1. `multiclass_primary_winner.joblib` (2.8 MB)
2. `multiclass_nosttl_winner.joblib` (2.9 MB)
3. `binary_calibrated_winner.joblib` (270 KB)

Confirmed:
- Each bundle stores the fitted `XGBClassifier` alongside a 3-element preprocessor tuple:
  `preprocessor = (base_preprocessing_pipeline, NetworkFlowFeatureEngineer, StandardScaler)`
- The bundle executes end-to-end inference directly from raw tabular DataFrames, automatically generating the 10 domain features and scaling them identically to training conditions.

---

## Conclusion
The Phase 4 metrics, thresholds, and serialization bundles are verified. Phase 5 explainability implementation proceeds with full confidence.
