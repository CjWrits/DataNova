# Phase 7 Inference API, Backend Integration & Service Validation Report

**Project:** Adaptive Threat Intelligence — DataNova Global Data-thon 2026  
**Date:** 2026-11-12  
**Framework:** FastAPI 0.133.1 (Uvicorn 0.41.0, Pydantic 2.11.7)  
**Reference Commit:** `9e686f6`  
**Test Suite Status:** 79/79 Passed (100% pass rate, 0 failures, 0 warnings)  

---

## 1. Executive Summary

Phase 7 implements a service-layer backend for the Adaptive Threat Intelligence NIDS architecture using FastAPI. It integrates the serialized machine learning models, statistical drift monitoring engines, and SHAP explainability utilities developed in Phases 3–6 into a unified, versioned REST API (`/api/v1`).

### Operational Scope & Boundaries
- **Tabular Ingestion Exclusively:** The API accepts pre-computed tabular flow records (42 features matching the UNSW-NB15 schema). It is not an inline packet-inspection engine and does not perform live network sniffing.
- **Strict Model Immutability:** Models remain frozen in their serialized Joblib artifacts. No retraining, weight mutation, threshold adaptation, or silent model replacement occurs during API operation.
- **Explicit Model Selection:** Multiclass inference defaults to Candidate 8 (Primary with STTL). The alternative Candidate 10 (No-STTL) is strictly selectable via explicit client configuration and is never substituted automatically.
- **Bounded Resources:** Designed to operate deterministically within the 16 GB workstation environment by caching models on startup, bounding batch inputs, and limiting SHAP operations to small cohorts ($N \le 10$).

---

## 2. API Endpoint Inventory & Specifications

The API exposes 9 endpoints grouped under versioned paths:

| Method | Endpoint | Description | Input / Constraints | Primary Response Payload |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` | Root health and readiness | None | Service status, model readiness, reference readiness |
| `GET` | `/api/v1/health` | Versioned health check | None | Service status, model readiness, reference readiness |
| `GET` | `/api/v1/system/status` | Runtime system telemetry | None | Uptime, RSS memory (MB), loaded models, active thresholds |
| `GET` | `/api/v1/models` | Catalog of loaded models | None | Model IDs, schemas, versions, feature counts, threshold presets |
| `POST` | `/api/v1/predict/binary` | Binary attack classification | `flows: List[Flow]` ($1 \le N \le 1000$), `threshold_preset` | Predicted label (0/1), label name, attack probability, threshold applied |
| `POST` | `/api/v1/predict/multiclass` | 10-class attack classification | `flows: List[Flow]` ($1 \le N \le 1000$), `model_id` (`c8` or `c10`) | Predicted attack category, per-class probabilities, confidence score |
| `POST` | `/api/v1/predict/compare` | Dual C8 vs C10 comparison | `flows: List[Flow]` ($1 \le N \le 1000$) | Both predictions, individual confidences, agreement rate, L1 probability distance |
| `POST` | `/api/v1/monitor/drift` | Telemetry drift & data quality | `flows: List[Dict]` ($1 \le N \le 5000$), `batch_id` | PSI, KS test, missing/inf rates, novel categories, adaptive SOC recommendations |
| `POST` | `/api/v1/explain` | Local SHAP tree attributions | `flows: List[Flow]` ($1 \le N \le 10$), `model_id`, `top_k` | Predicted class, top-k feature attributions in raw log-odds margins, direction |

---

## 3. Request & Response Examples

### A. Binary Detection (`POST /api/v1/predict/binary`)

#### Request
```json
{
  "flows": [
    {
      "proto": "tcp",
      "service": "http",
      "state": "FIN",
      "dur": 0.1214,
      "spkts": 10,
      "dpkts": 8,
      "sbytes": 2520,
      "dbytes": 1040,
      "rate": 140.0,
      "sttl": 62.0,
      "dttl": 252.0,
      "sload": 150000.0,
      "dload": 65000.0,
      "sloss": 0.0,
      "dloss": 0.0,
      "sinpkt": 12.14,
      "dinpkt": 14.28,
      "sjit": 1.5,
      "djit": 1.2,
      "swin": 255.0,
      "stcpb": 624100.0,
      "dtcpb": 812400.0,
      "dwin": 255.0,
      "tcprtt": 0.0012,
      "synack": 0.0006,
      "ackdat": 0.0006,
      "smean": 252.0,
      "dmean": 130.0,
      "trans_depth": 1.0,
      "response_body_len": 512.0,
      "ct_srv_src": 2.0,
      "ct_state_ttl": 0.0,
      "ct_dst_ltm": 3.0,
      "ct_src_dport_ltm": 1.0,
      "ct_dst_sport_ltm": 1.0,
      "ct_dst_src_ltm": 2.0,
      "is_ftp_login": 0.0,
      "ct_ftp_cmd": 0.0,
      "ct_flw_http_mthd": 1.0,
      "ct_src_ltm": 2.0,
      "ct_srv_dst": 2.0,
      "is_sm_ips_ports": 0.0
    }
  ],
  "threshold_preset": "default"
}
```

#### Response
```json
{
  "model_id": "binary_winner",
  "model_name": "Binary Calibrated XGBoost Winner",
  "threshold_preset": "default",
  "threshold_applied": 0.4951,
  "predictions": [
    {
      "index": 0,
      "label": 0,
      "label_name": "Normal",
      "attack_probability": 0.0412,
      "threshold_applied": 0.4951
    }
  ],
  "flow_count": 1,
  "processing_time_ms": 18.61
}
```

---

### B. Multiclass Classification (`POST /api/v1/predict/multiclass`)

#### Request
```json
{
  "flows": [ ... ],
  "model_id": "c8"
}
```

#### Response
```json
{
  "model_id": "c8",
  "model_name": "Candidate 8: FE + Cost-Sensitive Sqrt",
  "schema_used": "primary",
  "predictions": [
    {
      "index": 0,
      "predicted_class_id": 0,
      "predicted_class_name": "Normal",
      "confidence": 0.9421,
      "probabilities": {
        "Normal": 0.9421,
        "Generic": 0.0215,
        "Exploits": 0.0142,
        "Fuzzers": 0.0118,
        "DoS": 0.0034,
        "Reconnaissance": 0.0031,
        "Analysis": 0.0019,
        "Backdoor": 0.0011,
        "Shellcode": 0.0006,
        "Worms": 0.0003
      }
    }
  ],
  "flow_count": 1,
  "processing_time_ms": 19.45
}
```

---

### C. Model Comparison (`POST /api/v1/predict/compare`)

#### Request
```json
{
  "flows": [ ... ]
}
```

#### Response
```json
{
  "total_flows": 1,
  "agreement_count": 1,
  "disagreement_count": 0,
  "agreement_rate": 1.0,
  "disagreement_rate": 0.0,
  "comparison_items": [
    {
      "index": 0,
      "c8_predicted_class": "Normal",
      "c8_confidence": 0.9421,
      "c10_predicted_class": "Normal",
      "c10_confidence": 0.9315,
      "agreement": true,
      "l1_distribution_divergence": 0.0212
    }
  ],
  "c8_model_name": "Candidate 8: FE + Cost-Sensitive Sqrt",
  "c10_model_name": "Candidate 10: Robustness Track No-STTL",
  "processing_time_ms": 32.10
}
```

---

### D. SHAP Explainability (`POST /api/v1/explain`)

#### Request
```json
{
  "flows": [ ... ],
  "model_id": "c8",
  "top_k": 5
}
```

#### Response
```json
{
  "model_id": "c8",
  "model_name": "Candidate 8: FE + Cost-Sensitive Sqrt",
  "output_space_description": "Attributions represent additive shifts in raw pre-softmax log-odds margins.",
  "explanations": [
    {
      "index": 0,
      "predicted_class": "Normal",
      "output_space": "raw log-odds margin logit units",
      "base_margin": -0.8412,
      "top_attributions": [
        {
          "feature_name": "sttl",
          "attribution_margin": 1.4215,
          "direction": "push_towards_class"
        },
        {
          "feature_name": "ct_state_ttl",
          "attribution_margin": 0.9851,
          "direction": "push_towards_class"
        },
        {
          "feature_name": "feat_src_payload_density",
          "attribution_margin": 0.6521,
          "direction": "push_towards_class"
        },
        {
          "feature_name": "sbytes",
          "attribution_margin": -0.3412,
          "direction": "push_away_from_class"
        },
        {
          "feature_name": "rate",
          "attribution_margin": 0.2814,
          "direction": "push_towards_class"
        }
      ]
    }
  ],
  "flow_count": 1,
  "processing_time_ms": 677.80
}
```

---

## 4. Operational Safeguards & Architectural Policies

1. **Input Validation & Non-Finite Rejection:**
   Pydantic schema validation enforces finite numeric types. Payloads containing `NaN`, `Infinity`, or string values where floats are expected are rejected with HTTP 422 (`ValidationError`).
2. **Explicit Batch Limits:**
   - Prediction requests: Maximum 1,000 flows per batch.
   - Drift monitoring requests: Maximum 5,000 flows per batch.
   - SHAP explanation requests: Maximum 10 flows per batch (safeguard against excessive memory consumption).
3. **Threshold Preset Resolution:**
   The binary inference endpoint supports calibrated presets stored directly within the model bundle:
   - `default`: Calibrated baseline operating point (`0.4951`)
   - `f1_optimal`: Threshold maximizing test Macro F1 (`0.5247`)
   - `cost_optimal`: Threshold minimizing operational misclassification cost (`0.1783`, Recall = 99.6%)
   - `low_false_alarm`: High-precision operating point (`0.7227`, FPR = 2.0%)
   - `high_recall`: High-sensitivity operating point (`0.3268`, Recall = 98.1%)
4. **Data Quality vs. Statistical Drift Separation:**
   Data pipeline failures (missingness, infinities) are monitored independently from density drift (PSI, KS). Corrupted pipelines trigger `FLAG_AFFECTED_FEATURES` at HIGH priority, regardless of whether clean rows exhibit distribution stability.
5. **Safe Error Masking:**
   Unhandled server exceptions are trapped by a centralized exception handler. Raw file system paths, machine usernames, and internal stack traces are masked, returning a standardized JSON error message.

---

## 5. Local Startup & Execution Procedure

### Requirements
- Python 3.10+ (tested on Python 3.13)
- Virtual environment with project dependencies installed (`fastapi`, `uvicorn`, `pydantic-settings`, `xgboost`, `shap`, `scikit-learn`, `joblib`).

### Startup Command
From the project root (`C:\Users\Chirag\Desktop\DataNova`):
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Interactive Documentation
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`
- **OpenAPI Schema:** `http://127.0.0.1:8000/openapi.json`

---

## 6. Performance Benchmarks

*Measured locally under single-threaded benchmark conditions. Values represent local execution timings, not guaranteed network throughput under live production loads.*

| Operation | Batch Size | Measured Latency | Per-Flow Processing Time |
| :--- | :---: | :---: | :---: |
| Binary Inference | 1 flow | 18.61 ms | 18.61 ms |
| Binary Inference | 100 flows | 24.12 ms | 0.24 ms |
| Multiclass Inference (C8) | 1 flow | 19.45 ms | 19.45 ms |
| Multiclass Inference (C8) | 100 flows | 31.61 ms | 0.31 ms |
| Model Comparison (C8 vs C10) | 10 flows | 40.60 ms | 4.06 ms |
| SHAP TreeExplainer Attribution | 1 flow | 677.80 ms | 677.80 ms |
| SHAP TreeExplainer Attribution | 10 flows | 4,210.50 ms | 421.05 ms |

---

## 7. Known Limitations & Research Boundaries

1. **Tabular Feature Dependency:** The API requires pre-extracted flow statistics (e.g. `dur`, `spkts`, `sbytes`). It cannot directly ingest raw PCAP files or process live network packet streams without an external flow exporter (e.g. Argus / Zeek).
2. **SHAP Latency:** Computing exact TreeExplainer attributions requires tree traversal across all estimators in the ensemble. While single-flow latency (~670 ms) is acceptable for tier-2 SOC analyst inspection, it is not suitable for line-rate inline filtering.
3. **Local Development Configuration:** The current API service runs unauthenticated over HTTP, suitable for research validation and local testing. Public or production deployment would require an API gateway, TLS termination, and token-based authentication.
