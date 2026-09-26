# Test Suite (`tests/`)

## Purpose
This directory houses unit, integration, and ML validation test suites to ensure system reliability, numerical stability, and algorithmic correctness.

## Test Categorization
- `unit/`:
  - Preprocessing and split isolation checks (ensuring no test indices appear in train).
  - Feature extraction logic tests.
  - Cost matrix and custom metric calculation validations.
  - Statistical drift calculation unit tests (KS test, PSI formulas).
  - Pydantic schema validation tests.
- `integration/`:
  - End-to-end pipeline sanity tests (dummy flow passed into pipeline -> prediction + SHAP explanation).
  - Backend API integration tests (FastAPI test client).
  - Database persistence and audit logging checks.

## Running Tests
```bash
pytest tests
```
