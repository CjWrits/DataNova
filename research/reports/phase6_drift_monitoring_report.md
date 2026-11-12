# Phase 6 Drift Detection, Monitoring & Adaptive Response Simulation Report

**Project:** Adaptive Threat Intelligence — DataNova Global Data-thon 2026  
**Date:** 2026-11-12  
**Reference Training Distribution:** `data/processed/unsw_nb15/multiclass/train.csv` (151,325 rows)  
**Sealed Test Partition Status:** Strictly unvisited (0% leakage or calibration influence)  
**Execution Runtime:** 11.80 seconds  

## 1. Executive Summary

Phase 6 establishes a robust, non-destructive drift monitoring and adaptive response simulation framework for the NIDS architecture. Recognizing the foundational dataset constraint—that UNSW-NB15 lacks a chronological traffic stream for natural longitudinal evaluation—this investigation utilizes **seven explicitly labeled, controlled simulation scenarios** on validation data to stress-test statistical divergence metrics (PSI, KS test, novel category detection, missingness rates, and prediction entropy).

Furthermore, Phase 6 implements a **rule-based adaptive decision engine** that delivers actionable operational guidance to Security Operations Centers (SOCs) without ever triggering automated model replacement or unverified retraining. Finally, a direct resilience comparison between Candidate 8 (Primary with STTL) and Candidate 10 (No-STTL) demonstrates the concrete operational value of the No-STTL architecture under adversarial TTL perturbation.

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
Monitors output probability margins: mean confidence $\frac{1}{N}\sum \max_c P(c \mid x)$, low-confidence rate ($P < 0.60$), and prediction entropy $H(P) = -\sum P(c) \ln P(c)$.

## 3. Controlled Simulation Scenarios & Quantitative Results

| Scenario ID | Scenario Name | Target Features | Feature Status | Drifting Features | Pred PSI | Mean Conf | Low-Conf Rate | Primary Recommended Action |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `scenario_0_stable_control` | Stable Reference Control Batch | `None` | **STABLE** | 0 | 0.009 | 0.900 | 10.1% | **`CONTINUE_MONITORING`** |
| `scenario_1_numerical_shift` | Numerical Volume & Duration Shift | `sbytes,dur,sload` | **CRITICAL_DRIFT** | 3 | 1.145 | 0.842 | 17.4% | **`RECOMMEND_OFFLINE_EVALUATION`** |
| `scenario_2_categorical_shift` | Protocol & Service Frequency Skew | `proto,service` | **MODERATE_DRIFT** | 2 | 0.036 | 0.872 | 16.4% | **`INCREASE_MONITORING_FREQUENCY`** |
| `scenario_3_novel_categories` | Novel Unseen Category Injection | `proto,service` | **MODERATE_DRIFT** | 2 | 0.012 | 0.895 | 10.7% | **`RECOMMEND_DATA_COLLECTION`** |
| `scenario_4_data_corruption` | Missing & Non-Finite Pipeline Corruption | `sbytes,dur,sload,dload` | **STABLE** | 0 | 0.007 | 0.898 | 10.7% | **`FLAG_AFFECTED_FEATURES`** |
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
- **Observed Drift Status:** `STABLE` (0 drifting features)
- **Primary Adaptive Action:** `FLAG_AFFECTED_FEATURES`
- **Decisions Logged:**
  - `[HIGH] FLAG_AFFECTED_FEATURES`: Data pipeline corruption detected with elevated missing or non-finite rates. -> *Inspect upstream flow capture appliances and sensor telemetry parsers.*

### STTL Synthetic Artifact Perturbation (`scenario_5_sttl_artifact_perturbation`)

- **Description:** Adversarial TTL spoofing / OS heterogeneity randomizing `sttl` across [1, 255].
- **Perturbation:** Complete randomization of STTL across [1, 255]
- **Observed Drift Status:** `MODERATE_DRIFT` (1 drifting features)
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

## 5. Candidate 8 vs Candidate 10 Resilience Under STTL Perturbation

To evaluate the real-world operational value of the No-STTL architecture developed in Phase 4, Scenario 5 simulated adversarial TTL spoofing by randomizing `sttl` across $[1, 255]$ for all 1,000 flows:

- **Candidate 8 (Primary Model with STTL):** **36.40% of predictions were altered** solely by changing `sttl`, without modifying any underlying packet volumes, durations, or TCP flags.
- **Candidate 10 (Robustness Model without STTL):** **0.00% of predictions were altered**. Candidate 10 demonstrated **complete mathematical invariance** to TTL spoofing.

### Adaptive Engine Action
The adaptive engine recognized that `sttl` exhibited critical drift while core flow features remained stable, triggering the specialized recommendation: **`RECOMMEND_NO_STTL_FALLBACK`**. In a production SOC, this instructs operators to route traffic through Candidate 10, neutralizing adversarial TTL spoofing without retraining.

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
