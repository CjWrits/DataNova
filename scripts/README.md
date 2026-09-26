# Utility & Automation Scripts (`scripts/`)

## Purpose
This directory contains standalone, repeatable command-line scripts for pipeline execution, benchmark data preparation, model evaluation, and server verification.

## Planned Scripts (Future Phases)
- `download_data.py` — Verifies checksums and downloads designated raw intrusion detection datasets.
- `run_preprocessing.py` — Executes zero-leakage split and preprocessing pipeline.
- `train_baseline.py` — Trains benchmark classifier and logs metrics to `artifacts/metrics/`.
- `evaluate_model.py` — Runs full test evaluation suite (PR-AUC, cost matrix, FPR).
- `generate_drift_report.py` — Computes drift statistics comparing reference baseline against target windows.
- `run_historical_replay.py` — Feeds historical flow events through backend API to validate live streaming simulation.

## Usage Guidelines
All scripts must:
- Accept arguments via standard CLI flags (e.g. `--config`, `--data-dir`, `--seed`).
- Load configurations from `configs/system_config.yaml`.
- Log progress with timestamps and structured outputs.
- Never write intermediate data outside the designated `data/` or `artifacts/` directories.
