# Research & Experimentation Workspace

## Purpose
This directory houses exploratory data analysis (EDA), prototype notebooks, ablation studies, baseline comparisons, and statistical visualizations for the **Adaptive Threat Intelligence** project.

## Subdirectories
- `notebooks/` — Jupyter notebooks for exploratory data analysis, class imbalance analysis, feature importance screening, and distribution profiling.
- `experiments/` — Experiment logs, hyperparameter run configurations, and validation tracking.
- `visualizations/` — Saved publication-grade plots (ROC-PR curves, confusion matrices, SHAP summary plots, drift time-series).

## Guidelines
1. **Reproducibility**: Notebooks must fix random seeds and run top-to-bottom without manual variable mutations.
2. **Production Boundary**: Core ML logic must not remain trapped in notebooks. Code proven effective in `research/` must be refactored into modular components within `ml/`.
3. **No Fabricated Benchmarks**: All figures and findings must originate directly from verified evaluation runs.
