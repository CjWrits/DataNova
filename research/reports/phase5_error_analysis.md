# Phase 5 Diagnostic Error Analysis: False Positives & False Negatives

**Project:** Adaptive Threat Intelligence — DataNova Global Data-thon 2026  

## 1. Benign Traffic Flagged as Attacks (False Alarms)

### Root-Cause Analysis: Why Benign Traffic Resembles Fuzzers

In Candidate 8, **43 validation benign flows** were misclassified as `Fuzzers`. SHAP attribution drivers:

| Driver Rank | Feature | Mean SHAP Pull Towards Fuzzers | Explanation |
| :---: | :--- | :---: | :--- |
| 1 | `service` | +0.5937 | Elevated variance in session duration and packet size ratio resembling fuzz testing. |
| 2 | `sttl` | +0.4567 | Elevated variance in session duration and packet size ratio resembling fuzz testing. |
| 3 | `ct_srv_dst` | +0.3270 | Elevated variance in session duration and packet size ratio resembling fuzz testing. |
| 4 | `smean` | +0.0976 | Elevated variance in session duration and packet size ratio resembling fuzz testing. |
| 5 | `feat_dst_payload_density` | +0.0709 | Elevated variance in session duration and packet size ratio resembling fuzz testing. |

**Core Insight:** Fuzzing tools inject randomized payload lengths and malformed headers. Legitimate flows with dynamic content negotiation (e.g., compressed HTTP streams, DNS query retries, TLS handshakes with large certificate chains) exhibit similar asymmetric byte ratios and packet rates, causing the cost-weighted boundary to pull them into Fuzzers.

## 2. Attacks Missed as Normal (Escaped Attacks)

### Why Attacks Masquerade as Benign Flows

#### Fuzzers Missed as Normal (Count: 11)

| Driver Rank | Feature | Mean SHAP Pull Towards Normal |
| :---: | :--- | :---: |
| 1 | `sttl` | -0.6371 |
| 2 | `ct_dst_src_ltm` | +0.5323 |
| 3 | `service` | +0.5151 |
| 4 | `dbytes` | +0.0846 |
| 5 | `dur` | +0.0709 |

#### DoS Missed as Normal (Count: 0)

| Driver Rank | Feature | Mean SHAP Pull Towards Normal |
| :---: | :--- | :---: |

#### Analysis Missed as Normal (Count: 12)

| Driver Rank | Feature | Mean SHAP Pull Towards Normal |
| :---: | :--- | :---: |
| 1 | `sttl` | -0.8291 |
| 2 | `response_body_len` | +0.7186 |
| 3 | `dbytes` | +0.3343 |
| 4 | `feat_mean_packet_size_ratio` | +0.2882 |
| 5 | `ct_dst_src_ltm` | +0.1503 |

**Core Insight for Missed Attacks:** Attacks missed as Normal typically feature low connection counts (`ct_state_ttl`, `ct_srv_src`) and conservative packet sizes that mimic standard interactive sessions, successfully evading volumetric heuristics.

## 3. Attack-to-Attack Confusion: Why DoS and Worms Are Misclassified as Exploits

A critical discovery from confusion matrix dissection and SHAP attribution is that **underperforming minority attacks do not primarily escape as Normal traffic; they are misclassified into the dominant Exploits family**:

- In Candidate 8, **237 of 548 test DoS flows (43.2%)** and **252 validation DoS flows** were misclassified as `Exploits`.
- Similarly, **6 of 24 test Worms flows (25.0%)** were misclassified as `Exploits`.

### Drivers Pushing DoS Towards Exploits

| Driver Rank | Feature | Mean SHAP Pull Towards Exploits | Explanation |
| :---: | :--- | :---: | :--- |
| 1 | `dttl` | +0.6269 | Server response TTL and service destination counts match Exploits response signatures. |
| 2 | `ct_srv_dst` | +0.3143 | Server response TTL and service destination counts match Exploits response signatures. |
| 3 | `sbytes` | +0.1282 | Server response TTL and service destination counts match Exploits response signatures. |
| 4 | `ct_srv_src` | +0.0678 | Server response TTL and service destination counts match Exploits response signatures. |
| 5 | `smean` | +0.0588 | Server response TTL and service destination counts match Exploits response signatures. |

**Root-Cause Synthesis:** Both DoS floods and remote Exploits target server daemons, triggering symmetric server ACK/RST packets with identical Destination TTL values (`dttl = 252` or `64`) and high connection density (`ct_srv_dst`). Because the Exploits family has 8x higher representation in training data than DoS, the model's split thresholds heavily favor the Exploits leaf nodes unless volumetric payload and load ratios are exceptionally extreme.
