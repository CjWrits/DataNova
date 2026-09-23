# Configuration Directory (`configs/`)

## Purpose
This directory houses centralized, declarative configuration files for the Adaptive Threat Intelligence system. Storing parameters in structured YAML files guarantees experiment reproducibility and eliminates hardcoded constants across the codebase.

## Files
- `system_config.yaml` — Declarative hyperparameters, data splits, feature configurations, cost matrix parameters, and drift detection thresholds.
- `logging.yaml` — Standardized logging configuration across ML training, CLI utilities, and FastAPI ASGI execution.

## Usage in Code
Configurations are loaded into typed Pydantic models via `backend/app/core/config.py` or pipeline runners in `scripts/`.
