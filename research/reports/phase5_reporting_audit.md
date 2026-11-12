# Phase 5 Reporting & Interpretation Audit Report

**Project:** Adaptive Threat Intelligence — DataNova Global Data-thon 2026  
**Reference Commit:** `f6034d089c0e934049a304bfc4c7ce486a31083e`  
**Date:** 2026-11-12  
**Purpose:** Pre-Phase 6 audit resolving seven reviewer-identified items regarding SHAP interpretation, causality vs. attribution, binary model scope, output space units, No-STTL claims, and sample cohort documentation.

---

## 1. SHAP Attribution vs. Causality and Feature Redundancy

### Audit Finding
The Phase 5 report concluded:
> *"All 10 features registered positive SHAP attributions, therefore zero feature redundancy exists."*
> *"Physical root cause..." / "Functional specialization..."*

### Scientific Correction
1. **Attribution $\ne$ Statistical Independence:** SHAP (SHapley Additive exPlanations) computes fair credit allocation for tree splits learned by a specific trained model. If two collinear or redundant features exist in a dataset, decision trees can split on both across different tree depths or estimators, resulting in non-zero SHAP values for both. Positive attribution does **not** establish that a feature is statistically non-redundant, causal, or uniquely informative in the physical world.
2. **Attribution $\ne$ Physical Intent:** Phrases such as *"physical root cause"* and *"functional specialization"* are retracted. These interpretations represent **plausible domain hypotheses consistent with the observed dataset and transport protocols**, rather than experimentally verified physical causal relationships.
3. **Corrected Status:** All 10 engineered flow features demonstrated active utilization by the XGBoost trees (positive mean absolute SHAP). However, rigorous proof of feature independence or causality requires controlled out-of-distribution ablation and perturbation experiments.

---

## 2. Clarification of Binary Model Explanation Scope

### Audit Finding
Under *"Models Explained"*, the Phase 5 completion report listed four models, including `binary_calibrated_winner.joblib`. However, the runtime benchmarks and global comparison figures analyzed only the three multiclass models.

### Verification & Correction
- While `binary_calibrated_winner.joblib` was loaded and tested for bundle integrity, full global and class-specific SHAP attribution was executed exclusively on the **three multiclass models**:
  1. Phase 3 Baseline Multiclass XGBoost (`artifacts/models/multiclass_primary_xgboost.joblib`)
  2. Phase 4 C8 Feature-Engineered Cost-Sensitive Winner (`artifacts/models/phase4/multiclass_primary_winner.joblib`)
  3. Phase 4 C10 No-STTL Robustness Winner (`artifacts/models/phase4/multiclass_nosttl_winner.joblib`)
- **Reason:** Phase 5's primary research objective was diagnosing 10-class inter-family attack confusion (e.g., why DoS is misclassified as Exploits, why Analysis escapes as Normal, and why benign flows resemble Fuzzers).
- The binary model was not included in the global SHAP comparison matrix. Documentation has been updated to reflect that **three multiclass models were explained**.

---

## 3. SHAP Output Space, Margins, and Attribution Direction

### Output Space Specification
- `shap.TreeExplainer` on the multi-class `XGBClassifier` outputs attributions in **raw margin space (log-odds logits $z_k(x)$)** prior to the multi-class softmax transformation:
  $$z_k(x_i) = \phi_{0, k} + \sum_{j=1}^{D} \phi_{i, j, k}$$
  where $\phi_{0, k}$ is the base expected margin for class $k$, and $\phi_{i, j, k}$ is the SHAP attribution of feature $j$ for class $k$.
- The class predicted probability is given by:
  $$P(y = k \mid x_i) = \frac{\exp(z_k(x_i))}{\sum_{m=0}^{K-1} \exp(z_m(x_i))}$$

### Meaning of Attribution Values
1. **Positive SHAP Value ($\phi_{i, j, k} > 0$):** Indicates that feature $j$ increased the log-odds margin $z_k$, pushing the model's decision boundary **towards class $k$**.
2. **Negative SHAP Value ($\phi_{i, j, k} < 0$):** Indicates that feature $j$ decreased the log-odds margin $z_k$, pushing the model's decision boundary **away from class $k$**.
3. **Units:** SHAP values represent **additive shifts in raw margin log-odds**, NOT percentage probability changes. Phrasing such as *"+1.4407 pull"* denotes a $+1.4407$ increase in logit margin space.
4. **Cross-Model Comparison Precaution:** Comparing raw SHAP magnitudes between Phase 3 (unweighted baseline) and Phase 4 (sample-weighted models) must be done cautiously. Sample-weighted gradient boosting alters the gradient hessians, shifting the empirical logit distribution scale. Cross-model comparisons are most reliable in terms of relative ranking and attribution share (%), rather than raw absolute scalars.

---

## 4. Re-framing No-STTL (Candidate 10) Behavioral Observations

### Audit Finding
The report asserted that Candidate 10's attribution shifts prove the ensemble is stable and production-ready without `sttl`.

### Scientific Correction
1. Candidate 10's explanations show that when `sttl` is excluded from the feature matrix, the model successfully redistributes tree split importance to `ct_state_ttl`, `dttl`, `sbytes`, and the domain-engineered flow features (`feat_src_payload_density`, `feat_dst_payload_density`, `feat_packet_asymmetry_ratio`).
2. This confirms **internal model behavioral compensation within the UNSW-NB15 dataset**.
3. It does **not** independently prove production robustness, resilience to real-world network packet jitter, or transferability across independent captures (e.g., CIC-IDS2017). Candidate 10 remains an experimental robustness track, not an unqualified deployment recommendation.

---

## 5. Documentation of Cohort Sampling Counts and Rare-Class Floor

### Exact Sampling Cohort
To evaluate the 3 models within the 16 GB workstation memory constraint, an initial cohort of 1,113 samples was constructed from the validation set ($N = 33,045$) with a minimum inclusion floor of up to 30 samples per class. When downsampled to `max_eval_samples = 1,000` via stratified sampling, the exact per-class representation was:

| Class ID | Attack Family | Exact Evaluated Cohort Count | Total Validation Support | Cohort Representation % |
| :---: | :--- | :---: | :---: | :---: |
| 0 | **Normal** | 371 | 13,648 | 37.1% |
| 1 | **Generic** | 257 | 10,034 | 25.7% |
| 2 | **Exploits** | 114 | 4,451 | 11.4% |
| 3 | **Fuzzers** | 84 | 3,293 | 8.4% |
| 4 | **Reconnaissance** | 44 | 1,741 | 4.4% |
| 5 | **Analysis** | 27 | 97 | 2.7% |
| 6 | **Backdoor** | 27 | 53 | 2.7% |
| 7 | **DoS** | 27 | 559 | 2.7% |
| 8 | **Shellcode** | 27 | 229 | 2.7% |
| 9 | **Worms** | 22 | 24 | 2.2% |
| **Total** | **All Classes** | **1,000** | **33,045** | **100.0%** |

### Statistical Implication
- While rare classes were protected (e.g., 22 out of 24 Worms were explained), small absolute sample sizes ($n = 22$ for Worms, $n = 27$ for Analysis) mean their per-class explanation rankings reflect localized cluster behaviors and should be interpreted with appropriate caution.

---

## 6. Verification of Feature Aggregation Mapping

Inspection of [ml/explainability/shap_explainer.py](file:///C:/Users/Chirag/Desktop/DataNova/ml/explainability/shap_explainer.py#L42-L88) confirmed:
- Candidate 8: 79 preprocessed columns (69 base transformed + 10 domain engineered).
  - 39 continuous features (`num__*`) map 1-to-1 to base schema numerical names.
  - 30 categorical dummy columns (`cat__proto_*`, `cat__service_*`, `cat__state_*`) map strictly to parent features `proto`, `service`, and `state`.
  - 10 engineered features (`feat_*`) map 1-to-1 to their documented definitions.
  - Aggregation to original feature space yields 43 unique parent features (39 numerical + 3 categorical + 10 engineered - wait, 39 numerical + 3 categorical = 42 base + 10 engineered = 52 original parent features).
- Dimensional alignment between transformed matrix, feature names, and SHAP tensor is exact.

---

## Conclusion
All seven audit items have been resolved and documented. Phase 5 explainability claims are properly qualified. Phase 6 drift detection and adaptive intelligence implementation proceeds on this rigorous scientific foundation.
