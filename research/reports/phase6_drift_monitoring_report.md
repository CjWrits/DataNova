# Phase 6 Drift Detection, Monitoring & Adaptive Response Simulation Report

**Project:** Adaptive Threat Intelligence — DataNova Global Data-thon 2026  
**Date:** 2026-11-12  
**Reference Training Distribution:** `data/processed/unsw_nb15/multiclass/train.csv` (151,325 rows)  
**Sealed Test Partition Status:** Strictly unvisited (0% leakage or calibration influence)  
**Execution Runtime:** 11.80 seconds  

## 1. Executive Summary

Phase 6 establishes a robust, non-destructive drift monitoring and adaptive response simulation framework for the NIDS architecture. Recognizing the foundational dataset constraint—that UNSW-NB15 lacks a chronological traffic stream for natural longitudinal evaluation—this investigation utilizes **seven explicitly labeled, controlled simulation scenarios** on validation data to stress-test statistical divergence metrics (PSI, KS test, novel category detection, missingness rates, and prediction entropy).

Furthermore, Phase 6 implements a **rule-based adaptive decision engine** that delivers actionable operational guidance to Security Operations Centers (SOCs) without ever triggering automated model replacement or unverified retraining. Finally, a direct resilience comparison between Candidate 8 (Primary with STTL) and Candidate 10 (No-STTL) under synthetic STTL perturbation confirms that C10 is invariant to changes in the excluded `sttl` feature under this controlled perturbation. Broader robustness requires additional evaluation.

## 2. Statistical Monitoring Architecture & Metric Policies

### A. Population Stability Index (PSI)
Calculated across 10 reference quantile bins derived from the training partition:
$$\text{PSI} = \sum_{k=1}^{K} (q_k - p_k) \cdot \ln\left(\frac{q_k}{p_k}\right)$$
- $\text{PSI} < 0.10$: **STABLE** (Normal operational variance; no intervention).
- $0.10 \le \text{PSI} < 0.25$: **MODERATE_DRIFT** (Noticeable distribution shift; increase monitoring frequency).
- $\text{PSI} \ge 0.25$: **CRITICAL_DRIFT** (Significant departure; flag features, trigger analyst review).

### B. Kolmogorov-Smirnov (KS) Statistic
Two-sample non-parametric divergence test ($D \in [0, 1]$, $p$-value threshold $\alpha = 0.01$) sensitive to localized shape deformations that quantile-binned PSI might smooth over.

### C. Categorical Divergence & Novel Category Rate
Tracks the emergence of uncataloged protocols or services: $\text{NovelRate} = \frac{\sum \mathbb{I}(x \notin \text{Known})}{|B|}$. Triggers immediate data collection alerts if novel protocols exceed 2%.

### D. Model Prediction Drift & Entropy
Monitors output prediction distribution divergence and confidence margins:
- **Prediction PSI (`Pred PSI`):** Population Stability Index computed over the 10-class predicted probability distribution comparing batch predictions against reference predictions:
  $$\text{Pred PSI} = \sum_{c=1}^{10} (q_c - p_c) \cdot \ln\left(\frac{q_c}{p_c}\right)$$
  *(Note: Pred PSI is a non-negative divergence index with no upper bound of 1.0; a value such as 1.145 indicates substantial redistribution across attack classes, not a probability or proportion).*
- **Mean Confidence:** $\frac{1}{N}\sum \max_c P(c \mid x)$.
- **Low-Confidence Rate:** Proportion of flows where $\max_c P(c \mid x) < 0.60$.
- **Prediction Entropy:** $H(P) = -\sum P(c) \ln P(c)$.

### E. Architectural Separation: Data-Quality vs. Feature Drift
Data quality (missingness, infinities) and feature drift (PSI, KS) are tracked via independent analytical pathways:
- **Feature Distribution Drift:** Evaluated on finite, populated observations to measure genuine density shifts.
- **Data Quality Monitoring:** Tracks null rates ($\text{NaN}$) and non-finite rates ($\pm\infty$). Missing or invalid values represent data-quality defects and pipeline failures; they are never discarded silently or interpreted as distributional stability.

## 3. Controlled Simulation Scenarios & Quantitative Results

| Scenario ID | Scenario Name | Target Features | Feature Drift Status | Drifting Features Count | Pred PSI (Prediction Divergence) | Mean Conf | Low-Conf Rate (<0.60) | Primary Recommended Action |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `scenario_0_stable_control` | Stable Reference Control Batch | `None` | **STABLE** | 0 | 0.009 | 0.900 | 10.1% | **`CONTINUE_MONITORING`** |
| `scenario_1_numerical_shift` | Numerical Volume & Duration Shift | `sbytes,dur,sload` | **CRITICAL_DRIFT** | 3 | 1.145 | 0.842 | 17.4% | **`RECOMMEND_OFFLINE_EVALUATION`** |
| `scenario_2_categorical_shift` | Protocol & Service Frequency Skew | `proto,service` | **MODERATE_DRIFT** | 2 | 0.036 | 0.872 | 16.4% | **`INCREASE_MONITORING_FREQUENCY`** |
| `scenario_3_novel_categories` | Novel Unseen Category Injection | `proto,service` | **MODERATE_DRIFT** | 2 | 0.012 | 0.895 | 10.7% | **`RECOMMEND_DATA_COLLECTION`** |
| `scenario_4_data_corruption` | Missing & Non-Finite Pipeline Corruption | `sbytes,dur,sload,dload` | **STABLE** *(Data Quality Alert)* | 0 | 0.007 | 0.898 | 10.7% | **`FLAG_AFFECTED_FEATURES`** |
| `scenario_5_sttl_artifact_perturbation` | STTL Synthetic Artifact Perturbation | `sttl` | **MODERATE_DRIFT** | 1 | 0.816 | 0.857 | 14.6% | **`RECOMMEND_NO_STTL_FALLBACK`** |
| `scenario_6_multivariate_shift` | Combined Multivariate Network Shift | `sbytes,spkts,ct_srv_dst,service` | **CRITICAL_DRIFT** | 4 | 0.318 | 0.803 | 20.8% | **`RECOMMEND_OFFLINE_EVALUATION`** |

## 4. Scenario-Specific Deep-Dives

### Stable Reference Control Batch (`scenario_0_stable_control`)

- **Description:** Unperturbed validation batch representing standard in-distribution network traffic.
- **Perturbation:** 0.0
- **Observed Drift Status:** `STABLE` (0 drifting features)
- **Primary Adaptive Action:** `CONTINUE_MONITORING`
- **Decisions Logged:**
  - `[LOW] CONTINUE_MONITORING`: Telemetry and model inference metrics remain within baseline bounds. -> *Maintain standard passive telemetry observation cadence.*

### Numerical Volume & Duration Shift (`scenario_1_numerical_shift`)

- **Description:** Simulated 3.5x-5x scaling on transport volume variables (`sbytes`, `dur`, `sload`).
- **Perturbation:** 3.5x sbytes, 4x dur, 5x sload
- **Observed Drift Status:** `CRITICAL_DRIFT` (3 drifting features)
- **Primary Adaptive Action:** `RECOMMEND_OFFLINE_EVALUATION`
- **Decisions Logged:**
  - `[CRITICAL] RECOMMEND_OFFLINE_EVALUATION`: Systemic multivariate drift across 3 critical network features. -> *Initiate controlled offline evaluation against recent labeled captures; do not alter live models.*

### Protocol & Service Frequency Skew (`scenario_2_categorical_shift`)

- **Description:** Simulated protocol shift forcing 60% of batch flows to UDP / DNS traffic.
- **Perturbation:** 60% UDP/DNS surge
- **Observed Drift Status:** `MODERATE_DRIFT` (2 drifting features)
- **Primary Adaptive Action:** `INCREASE_MONITORING_FREQUENCY`
- **Decisions Logged:**
  - `[MEDIUM] INCREASE_MONITORING_FREQUENCY`: Moderate drift identified on 0 features. -> *Shorten monitoring window from hourly to 15-minute intervals to observe trajectory.*

### Novel Unseen Category Injection (`scenario_3_novel_categories`)

- **Description:** Introduction of unseen IoT and modern protocols (`wireguard`, `quic`, `coap`, `mqtt`).
- **Perturbation:** 12% novel proto, 15% novel service
- **Observed Drift Status:** `MODERATE_DRIFT` (2 drifting features)
- **Primary Adaptive Action:** `RECOMMEND_DATA_COLLECTION`
- **Decisions Logged:**
  - `[HIGH] RECOMMEND_DATA_COLLECTION`: New uncataloged protocols or services detected in network traffic. -> *Collect and quarantine sample flows for protocol labeling and offline schema extension.*

### Missing & Non-Finite Pipeline Corruption (`scenario_4_data_corruption`)

- **Description:** Sensor transmission failure injecting 8% NaNs and 5% Infs into telemetry columns.
- **Perturbation:** 8% NaN, 5% Inf
- **Observed Drift Status:** `STABLE` on finite observations (0 drifting features) alongside active **`DATA_QUALITY_ALERT`**.
- **Data-Quality vs Feature Drift Clarification:** 
  In this scenario, the non-corrupted observations retained their original distribution shape, resulting in PSI and KS values below threshold (`STABLE`). However, the data quality monitor detected 8% missingness and 5% non-finites. Missing and non-finite values are data-quality pipeline failures; they are never silently ignored or conflated with distributional stability. This distinction is what triggered the `FLAG_AFFECTED_FEATURES` action.
- **Primary Adaptive Action:** `FLAG_AFFECTED_FEATURES`
- **Decisions Logged:**
  - `[HIGH] FLAG_AFFECTED_FEATURES`: Data pipeline corruption detected with elevated missing or non-finite rates. -> *Inspect upstream flow capture appliances and sensor telemetry parsers.*

### STTL Synthetic Artifact Perturbation (`scenario_5_sttl_artifact_perturbation`)

- **Description:** Synthetic STTL perturbation randomizing `sttl` uniformly across $[1, 255]$ on 1,000 validation flows.
- **Perturbation:** Uniform synthetic randomization of STTL across $[1, 255]$.
- **Severity Classification Breakdown:**
  - **Feature Drift Severity:** `MODERATE_DRIFT` (Among all 43 monitored flow features, only 1 feature—`sttl`—drifted, while the remaining 42 transport features stayed strictly within normal baseline bounds).
  - **Feature-Specific Divergence:** `CRITICAL` for `sttl` in isolation ($\text{PSI} = 3.996, \text{KS} = 0.584, p < 10^{-15}$).
  - **Model Disagreement Severity:** `HIGH` (C8 altered 36.40% of its predictions under this perturbation, while C10 altered 0.00%).
  - **Adaptive Response Priority:** `HIGH` (`RECOMMEND_NO_STTL_FALLBACK`).
- **Primary Adaptive Action:** `RECOMMEND_NO_STTL_FALLBACK`
- **Decisions Logged:**
  - `[HIGH] RECOMMEND_NO_STTL_FALLBACK`: Severe divergence detected on synthetic testbed artifact `sttl` (PSI=3.996, KS=0.584, p=0.00e+00). -> *Evaluate parallel inference against the No-STTL model (Candidate 10) to mitigate synthetic TTL sensitivity.*

### Combined Multivariate Network Shift (`scenario_6_multivariate_shift`)

- **Description:** Concurrent volume scaling, connection fan-out increase, and service concentration.
- **Perturbation:** 2.2x bytes, 1.8x pkts, 2.0x fanout, 35% HTTP skew
- **Observed Drift Status:** `CRITICAL_DRIFT` (4 drifting features)
- **Primary Adaptive Action:** `RECOMMEND_OFFLINE_EVALUATION`
- **Decisions Logged:**
  - `[CRITICAL] RECOMMEND_OFFLINE_EVALUATION`: Systemic multivariate drift across 3 critical network features. -> *Initiate controlled offline evaluation against recent labeled captures; do not alter live models.*
  - `[HIGH] RECOMMEND_ANALYST_REVIEW`: Prediction confidence degraded below operational reliability threshold. -> *Escalate low-confidence flow predictions to SOC tier-2 analysts for manual verification.*

## 5. Candidate 8 vs Candidate 10 Invariance Under Synthetic STTL Perturbation

To evaluate the impact of the synthetic testbed artifact `sttl` identified during exploratory analysis, Scenario 5 applied a controlled synthetic perturbation randomizing `sttl` uniformly across $[1, 255]$ for all 1,000 flows:

- **Candidate 8 (Primary Model with STTL):** **36.40% of predictions were altered** (364 out of 1,000 flows) solely by modifying `sttl`, without altering any underlying packet volumes, durations, connection fan-outs, or protocol flags.
- **Candidate 10 (Robustness Model without STTL):** **0.00% of predictions were altered** (1,000 out of 1,000 flows identical).
- **Interpretation & Scientific Scope:** The experiment confirms that Candidate 10 is invariant to changes in the excluded `sttl` feature under this controlled perturbation. This verifies that C10 does not depend on `sttl`. However, this invariance does not establish general production robustness against other forms of distribution shift, evasion techniques, or real network attacks. Broader robustness requires additional evaluation on independent network traffic captures.

### Adaptive Engine Action
The adaptive engine recognized that `sttl` exhibited critical isolated divergence while core flow features remained stable, triggering the specialized recommendation: **`RECOMMEND_NO_STTL_FALLBACK`**. In an operational SOC, this provides decision support to operators to evaluate parallel inference against Candidate 10, mitigating synthetic TTL sensitivity without altering or retraining deployed models.

## 6. Strict Implementation Boundaries Enforced

All Phase 6 components strictly adhere to non-destructive governance principles:
1. **Zero Automatic Retraining:** Drift alerts never initiate autonomous retraining daemons.
2. **Zero Model Replacement:** Models C8 and C10 remain frozen in their serialized artifacts.
3. **Zero Threshold Mutation:** Binary and multiclass decision thresholds were not altered.
4. **No Synthetic Drift Misrepresentation:** All simulated shifts are explicitly documented as synthetic experiments.

## 7. Artifact Manifest

- Core Detector: [`ml/monitoring/drift_detector.py`](file:///C:/Users/Chirag/Desktop/DataNova/ml/monitoring/drift_detector.py)
- Simulation Suite: [`ml/monitoring/simulation.py`](file:///C:/Users/Chirag/Desktop/DataNova/ml/monitoring/simulation.py)
- Adaptive Engine: [`ml/monitoring/adaptive_engine.py`](file:///C:/Users/Chirag/Desktop/DataNova/ml/monitoring/adaptive_engine.py)
- Visualizer: [`ml/monitoring/visualizer.py`](file:///C:/Users/Chirag/Desktop/DataNova/ml/monitoring/visualizer.py)
- Execution Script: [`scripts/run_phase6_drift_monitoring.py`](file:///C:/Users/Chirag/Desktop/DataNova/scripts/run_phase6_drift_monitoring.py)
- Visualizations Directory: [`research/visualizations/phase6/`](file:///C:/Users/Chirag/Desktop/DataNova/research/visualizations/phase6/)
