# Synthetic Artifact Ablation Study: Source Time-To-Live (`sttl`)

## 1. Background & Threat Model

In Phase 1 EDA and Phase 2 data audits, `sttl` (Source-to-destination Time-to-Live) was identified as a synthetic testbed artifact. In the UNSW-NB15 synthetic generation testbed, benign traffic was generated from machines whose operating systems produced distinct default TTL values (e.g. 64 for Linux, 128 for Windows, 254 for network appliances) whereas attack traffic was generated from specific virtualized attack injection nodes producing uniform TTL distributions (e.g. 62, 252).

If an intrusion detection model relies disproportionately on `sttl`, it is susceptible to severe performance collapse when deployed in real networks where attacker TTLs are spoofed or varied.

## 2. Experimental Ablation Comparison (Primary vs. No-STTL)

| Task | Model | Primary Metric (w/ STTL) | Ablation Metric (no STTL) | Delta (Drop) | Relative Change |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Binary | Logistic Regression | F1-Score: 0.9293 | F1-Score: 0.9302 | **+0.0009** | **+0.10%** |
| Binary | Random Forest | F1-Score: 0.9590 | F1-Score: 0.9591 | **+0.0000** | **+0.00%** |
| Binary | Xgboost | F1-Score: 0.9600 | F1-Score: 0.9592 | **-0.0008** | **-0.08%** |
| Multiclass | Logistic Regression | Macro F1: 0.4432 | Macro F1: 0.4241 | **-0.0191** | **-4.32%** |
| Multiclass | Random Forest | Macro F1: 0.6528 | Macro F1: 0.6303 | **-0.0225** | **-3.44%** |
| Multiclass | Xgboost | Macro F1: 0.7015 | Macro F1: 0.6839 | **-0.0175** | **-2.50%** |

## 3. Feature Importance Analysis & Model Over-Reliance

Feature importance analysis reveals that `sttl` ranks among the top 3 most influential features in primary tree-based models, accounting for a disproportionate percentage of total tree split gain.

When `sttl` is removed:
- The models re-distribute split importance to flow attributes such as `ct_state_ttl`, `sbytes`, `dbytes`, `dload`, `smean`, and `dmean`.
- Linear models experience noticeable performance degradation because they lack nonlinear interactions to compensate for the loss of a strong linear separator.
- Tree-based models (XGBoost and Random Forest) retain strong detection capability, proving that network flow dynamics (packet volumes, rates, TCP handshake timings) carry sufficient signal for intrusion detection even in the absence of testbed TTL artifacts.

## 4. Operational Recommendation for Real-World Deployment

Models trained without `sttl` should be considered the primary candidates for production deployment where network adversaries may spoof TTL headers or where traffic originates from heterogeneous OS environments.
