# Phase 5 Explainability & Global Attribution Report

**Project:** Adaptive Threat Intelligence — DataNova Global Data-thon 2026  
**Date:** 2026-11-12  

## 1. Executive Summary

Phase 5 introduces a comprehensive SHAP (SHapley Additive exPlanations) explainability framework for the NIDS classifiers developed in Phases 3 and 4. Using TreeExplainer with strict parent-feature aggregation (folding one-hot encoded dummy variables back into original network flow variables), this investigation evaluates global feature importance, model sensitivity to synthetic testbed artifacts (`sttl`), and structural shifts between the baseline and imbalance-aware models.

## 2. Methodology & Computational Parameters

- **Explainer Architecture:** `shap.TreeExplainer` utilizing tree path-dependent exact Shapley estimation on the three frozen multiclass models (Phase 3 Baseline, Phase 4 C8 Winner, Phase 4 C10 Robustness Winner).
- **Output Space:** Raw margin log-odds logits ($z_k$) prior to multi-class softmax. Positive SHAP values indicate positive margin contributions pushing towards the class; negative values indicate margin deductions.
- **Memory Management:** Designed for 16 GB RAM developer workstation; evaluations executed on stratified cohorts without memory spikes.
- **Feature Aggregation:** For each sample $i$, class $c$, and parent feature $F$, the composite attribution is defined as $\text{SHAP}(i, F, c) = \sum_{j \in \text{children}(F)} \text{SHAP}(i, j, c)$.
- **Sample Cohort:** 1,000 stratified validation instances with guaranteed rare-class representation: Normal (371), Generic (257), Exploits (114), Fuzzers (84), Reconnaissance (44), Analysis (27), Backdoor (27), DoS (27), Shellcode (27), Worms (22).
- **Pipeline Execution Time:** ~11–14 seconds total.

## 3. Global Feature Importance Ranking (Top 15 Features)

| Rank | Feature | Phase 4 C8 Mean Abs SHAP | C8 Relative % | Phase 3 Baseline SHAP | Phase 4 C10 (No-STTL) SHAP |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | `service` | 0.7394 | 9.3% | 0.6345 | 0.7325 |
| 2 | `sttl` | 0.6822 | 8.6% | 0.7592 | 0.0000 |
| 3 | `sbytes` | 0.5210 | 6.6% | 0.6637 | 0.6005 |
| 4 | `ct_srv_dst` | 0.4977 | 6.3% | 0.5541 | 0.4985 |
| 5 | `ct_dst_src_ltm` | 0.3318 | 4.2% | 0.3884 | 0.3305 |
| 6 | `dbytes` | 0.3315 | 4.2% | 0.3850 | 0.3556 |
| 7 | `feat_mean_packet_size_ratio` | 0.3281 | 4.1% | 0.0000 | 0.3072 |
| 8 | `feat_src_payload_density` | 0.3112 | 3.9% | 0.0000 | 0.3031 |
| 9 | `feat_byte_asymmetry_ratio` | 0.2891 | 3.7% | 0.0000 | 0.2470 |
| 10 | `smean` | 0.2863 | 3.6% | 0.5644 | 0.2877 |
| 11 | `feat_dst_payload_density` | 0.2770 | 3.5% | 0.0000 | 0.2758 |
| 12 | `sload` | 0.2564 | 3.2% | 0.3101 | 0.2636 |
| 13 | `ct_src_dport_ltm` | 0.2192 | 2.8% | 0.2214 | 0.2181 |
| 14 | `dmean` | 0.2085 | 2.6% | 0.3544 | 0.2256 |
| 15 | `ct_dst_sport_ltm` | 0.1850 | 2.3% | 0.2198 | 0.2332 |

## 4. Analysis of Synthetic Testbed Artifact (`sttl`)

In the primary Phase 4 model (Candidate 8), `sttl` accounts for **8.6%** of total global SHAP attribution, confirming its status as a highly influential separator. However, in Candidate 10 (No-STTL Robustness Track), the model smoothly redistributes attribution across `ct_state_ttl`, `sbytes`, `dbytes`, `feat_byte_asymmetry_ratio`, and `tcprtt`, retaining robust detection performance without relying on the synthetic OS-generation artifact.

## 5. Visualizations Manifest

- Global Comparison Plot: `research/visualizations/phase5/global_feature_importance_comparison.png`
- Beeswarm Summary (Fuzzers): `research/visualizations/phase5/beeswarm_fuzzers_c8.png`
- Beeswarm Summary (Normal): `research/visualizations/phase5/beeswarm_normal_c8.png`
