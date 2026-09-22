# Project Roadmap & Implementation Phases

## Adaptive Threat Intelligence — DataNova Global Data-thon 2026

This roadmap details the sequential implementation phases for building the explainable, imbalance-aware, and adaptive Network Intrusion Detection System. Progression between phases requires passing all verification gates.

---

### Phase 0: Project Foundation and Architecture Setup [COMPLETED]
- [x] Environment inspection (OS, Python 3.13, Node 24, npm 11, Git, disk space).
- [x] Repository initialization and Git isolation.
- [x] Monorepo directory scaffold (`data/`, `research/`, `ml/`, `artifacts/`, `backend/`, `frontend/`, `scripts/`, `tests/`, `docs/`, `configs/`).
- [x] Comprehensive `.gitignore` protecting data dumps, model binaries, environment secrets, and caches.
- [x] Environment templates (`.env.example`).
- [x] Centralized configuration (`configs/system_config.yaml`, `configs/logging.yaml`).
- [x] Dependency management (`requirements.txt`, `pyproject.toml`, `frontend/package.json`).
- [x] Technical documentation (`docs/architecture.md`, `docs/roadmap.md`, `docs/methodology.md`, `docs/setup.md`).
- **Gate**: Strict phase boundary respected — no data downloaded, no models trained, no APIs built.

---

### Phase 1: Data Acquisition, Ingestion & Zero-Leakage Preprocessing
- [ ] Ingestion script for selected benchmark network intrusion dataset (e.g. CIC-IDS-2017 / UNSW-NB15).
- [ ] Data validation and schema integrity checks (handling infinite values, NaNs, duplicate flows).
- [ ] Exploratory Data Analysis (EDA) notebook documenting class distributions and feature correlation.
- [ ] Deterministic, zero-leakage partition module (Train / Validation / Hold-out Test).
- [ ] Transformer pipeline fitted strictly on training data (encoders, scalers, imputers).
- **Verification Gate**: Automated test verifying split isolation: `assert len(set(train_idx) & set(test_idx)) == 0`, and no data transformation leakage.

---

### Phase 2: Imbalance-Aware ML Model Training & Benchmarking
- [ ] Baseline benchmark implementation (Logistic Regression, Random Forest).
- [ ] Advanced tree-based estimator training (Cost-Sensitive XGBoost).
- [ ] Imbalance remediation experiments (Cost-sensitive loss vs. training-only SMOTE vs. Focal loss).
- [ ] Threshold calibration based on business cost matrix (cost of False Alarm vs. cost of Missed Breach).
- [ ] Model registry serialization (`artifacts/models/`) with complete run metadata and feature schemas.
- **Verification Gate**: Evaluation strictly on untouched test partition reporting PR-AUC, F1-macro, and Cost-weighted Loss. Zero synthetic records in test splits.

---

### Phase 3: Explainability (SHAP) & Statistical Drift Detection
- [ ] SHAP TreeExplainer integration for trained XGBoost models.
- [ ] Local instance feature attribution extraction (top-k contributing features for alert cards).
- [ ] Global feature importance summary extraction and visualization.
- [ ] Drift detection module implementing:
  - Two-sample Kolmogorov-Smirnov (KS) test for numerical flow features.
  - Population Stability Index (PSI) and Chi-square tests for categorical distributions.
- [ ] Sliding-window monitoring simulation over dynamic flow batches.
- **Verification Gate**: Unit tests verifying SHAP additivity property ($\sum \phi_i \approx f(x) - E[f(x)]$) and synthetic drift sensitivity validation.

---

### Phase 4: Backend API & Historical Replay Service
- [ ] FastAPI application setup with async routing and CORS configuration.
- [ ] Pydantic request/response schemas for telemetry inputs, threat classifications, and SHAP attributions.
- [ ] SQLAlchemy database models and migration setup for threat detection audit logging and drift events.
- [ ] Endpoints implementation:
  - `POST /api/v1/predict` — Real-time flow classification with confidence scores.
  - `POST /api/v1/explain` — Instant SHAP feature attribution waterfall.
  - `GET /api/v1/drift/status` — Sliding-window drift metrics and PSI scores.
  - `GET /api/v1/models` — Active model version metadata and registry history.
  - `POST /api/v1/replay/start` — Controlled historical network telemetry streaming simulation.
- **Verification Gate**: Pytest API integration suite with 100% route coverage and schema validation.

---

### Phase 5: Next.js SOC Analyst Dashboard
- [ ] Next.js (App Router) project build and TypeScript configuration.
- [ ] Tailwind CSS security dashboard layout with dark/light SOC themes.
- [ ] Real-time threat feed table with severity badges and flow metadata.
- [ ] Interactive SHAP explanation drawer with horizontal bar charts and impact directions.
- [ ] Statistical drift monitoring dashboard featuring Recharts time-series and PSI gauges.
- [ ] Historical replay control panel with play/pause/speed controls.
- **Verification Gate**: End-to-end integration: frontend accurately consumes backend endpoints without mock data.

---

### Phase 6: System Hardening, Benchmarking & Documentation Polish
- [ ] End-to-end benchmark suite comparing baseline vs. adaptive model under simulated drift.
- [ ] Security audit (input sanitization, rate limiting, dependency vulnerability check).
- [ ] Docker containerization (`docker-compose.yml` for backend, database, and frontend).
- [ ] Final hackathon presentation materials, architecture diagrams, and demonstration video walkthrough.
