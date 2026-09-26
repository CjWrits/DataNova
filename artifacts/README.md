# Artifacts Repository (`artifacts/`)

## Purpose
This directory stores serialized models, evaluation metrics, drift reports, and benchmark summaries. All serialized binaries are excluded from Git version control via `.gitignore`, while folder structures and metadata templates are retained.

## Subdirectories
- `models/` — Serialized pipeline bundles (`.joblib`, `.json`), including fitted preprocessors, encoders, classifiers, and SHAP explainer objects.
- `metrics/` — JSON-formatted performance evaluations, ROC/PR curve tables, confusion matrices, and drift statistics across versions.
- `reports/` — Exported experiment summaries, PDF/HTML reports, and audit trails.

## Versioning Convention
Models follow semantic versioning:
- Format: `model_<algorithm>_v<major>.<minor>.<patch>.joblib`
- Metadata: Each model artifact is accompanied by a `<model_id>_metadata.json` documenting:
  - Training dataset hash and date
  - Feature names and schema versions
  - Hyperparameters
  - Evaluation metrics on hold-out test split
  - Drift baseline distribution statistics

## Phase 0 Status
No model artifacts or metric files exist. Creation occurs during Phase 2 and Phase 3.
