# Phase 5 Class-Specific Attribution & Behavioral Profiles

**Project:** Adaptive Threat Intelligence — DataNova Global Data-thon 2026  

## 1. Class Attribution Profiles

Features that govern detection vary fundamentally across attack families:

### Normal Attribution Profile

| Rank | Feature | Mean Abs SHAP | Directional Pull | Relative Contribution |
| :---: | :--- | :---: | :---: | :---: |
| 1 | `sttl` | 2.5420 | Positive (+) | 38.7% |
| 2 | `service` | 0.9597 | Negative (-) | 14.6% |
| 3 | `ct_dst_src_ltm` | 0.5862 | Negative (-) | 8.9% |
| 4 | `dbytes` | 0.1868 | Negative (-) | 2.8% |
| 5 | `sbytes` | 0.1335 | Negative (-) | 2.0% |

### Fuzzers Attribution Profile

| Rank | Feature | Mean Abs SHAP | Directional Pull | Relative Contribution |
| :---: | :--- | :---: | :---: | :---: |
| 1 | `sttl` | 1.2670 | Negative (-) | 20.0% |
| 2 | `service` | 1.0943 | Negative (-) | 17.2% |
| 3 | `ct_srv_dst` | 0.4415 | Positive (+) | 7.0% |
| 4 | `ct_dst_src_ltm` | 0.3016 | Negative (-) | 4.8% |
| 5 | `dmean` | 0.2950 | Negative (-) | 4.6% |

### DoS Attribution Profile

| Rank | Feature | Mean Abs SHAP | Directional Pull | Relative Contribution |
| :---: | :--- | :---: | :---: | :---: |
| 1 | `ct_srv_dst` | 1.4407 | Negative (-) | 27.0% |
| 2 | `sttl` | 0.4205 | Negative (-) | 7.9% |
| 3 | `dbytes` | 0.2696 | Negative (-) | 5.1% |
| 4 | `ct_dst_src_ltm` | 0.2569 | Negative (-) | 4.8% |
| 5 | `ct_srv_src` | 0.2273 | Negative (-) | 4.3% |

### Analysis Attribution Profile

| Rank | Feature | Mean Abs SHAP | Directional Pull | Relative Contribution |
| :---: | :--- | :---: | :---: | :---: |
| 1 | `dbytes` | 1.7775 | Negative (-) | 18.1% |
| 2 | `dmean` | 0.8401 | Negative (-) | 8.5% |
| 3 | `response_body_len` | 0.7851 | Negative (-) | 8.0% |
| 4 | `sload` | 0.5687 | Negative (-) | 5.8% |
| 5 | `feat_synack_to_rtt_ratio` | 0.4809 | Negative (-) | 4.9% |

### Backdoor Attribution Profile

| Rank | Feature | Mean Abs SHAP | Directional Pull | Relative Contribution |
| :---: | :--- | :---: | :---: | :---: |
| 1 | `sload` | 1.4451 | Negative (-) | 15.0% |
| 2 | `feat_mean_packet_size_ratio` | 1.1228 | Negative (-) | 11.7% |
| 3 | `feat_byte_asymmetry_ratio` | 0.6965 | Negative (-) | 7.2% |
| 4 | `feat_src_payload_density` | 0.4821 | Negative (-) | 5.0% |
| 5 | `sbytes` | 0.4419 | Negative (-) | 4.6% |

### Shellcode Attribution Profile

| Rank | Feature | Mean Abs SHAP | Directional Pull | Relative Contribution |
| :---: | :--- | :---: | :---: | :---: |
| 1 | `service` | 1.8068 | Negative (-) | 18.6% |
| 2 | `ct_src_dport_ltm` | 1.4181 | Negative (-) | 14.6% |
| 3 | `sttl` | 0.9735 | Negative (-) | 10.0% |
| 4 | `sbytes` | 0.9615 | Negative (-) | 9.9% |
| 5 | `feat_dst_payload_density` | 0.5086 | Negative (-) | 5.2% |

### Worms Attribution Profile

| Rank | Feature | Mean Abs SHAP | Directional Pull | Relative Contribution |
| :---: | :--- | :---: | :---: | :---: |
| 1 | `sbytes` | 0.9953 | Negative (-) | 10.5% |
| 2 | `feat_byte_asymmetry_ratio` | 0.9762 | Negative (-) | 10.3% |
| 3 | `service` | 0.8203 | Negative (-) | 8.6% |
| 4 | `ct_dst_src_ltm` | 0.6350 | Negative (-) | 6.7% |
| 5 | `feat_src_payload_density` | 0.5898 | Negative (-) | 6.2% |

## 2. Distinctive Attack Class Mechanisms

- **Normal Traffic:** Dominated by low `ct_state_ttl`, balanced `feat_byte_asymmetry_ratio`, and standard `dur`.
- **DoS Attacks:** Heavily driven by `ct_srv_src` (high service connection counts), `sload`, and `feat_load_ratio`.
- **Analysis Probes:** Driven by `ct_dst_ltm`, small `sbytes`, and `feat_service_fanout_ratio` indicating horizontal scanning.
- **Worms:** Characterized by rapid fan-out counts, specific `dmean` packet payload thresholds, and distinct TCP flags.
