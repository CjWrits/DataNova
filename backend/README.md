# Backend API Service (`backend/`)

## Purpose
The `backend/` directory contains the FastAPI-based application server for Adaptive Threat Intelligence. It exposes secure endpoints for real-time inference, batch flow replay, SHAP-based feature attributions, drift monitoring metrics, and model management.

## Architecture
- `app/api/` — API routers and versioned endpoints (`/api/v1/predict`, `/api/v1/explain`, `/api/v1/drift`, `/api/v1/models`, `/api/v1/replay`).
- `app/core/` — Application configuration (Pydantic Settings), security, database connections, and logging.
- `app/models/` — SQLAlchemy ORM database models (Audit logs, detection records, drift events, model registry metadata).
- `app/schemas/` — Pydantic models for request validation and response serialization.
- `app/services/` — Business logic abstractions:
  - Inference service (loading pipeline from `artifacts/models/`, computing predictions and confidence scores).
  - Explanation service (calling TreeExplainer on model inputs and formatting feature attributions).
  - Drift service (calculating Kolmogorov-Smirnov p-values and population stability metrics over sliding inference windows).
  - Historical replay service (replaying dataset records sequentially to simulate streaming telemetry).
- `tests/` — Backend unit and integration tests using Pytest and HTTPX AsyncClient.

## Tech Stack
- **Framework**: FastAPI (async ASGI)
- **Validation**: Pydantic v2
- **Server**: Uvicorn
- **ORM & DB**: SQLAlchemy 2.0 with SQLite (pluggable PostgreSQL support)

## Phase 0 Status
Directory scaffold and packaging are initialized. Endpoints and services will be implemented in Phase 4.
