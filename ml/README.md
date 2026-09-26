# Machine Learning Architecture (`ml/`)

## Purpose
The `ml/` package encapsulates production-grade, reproducible machine learning modules for the Adaptive Threat Intelligence NIDS. It is structured to ensure complete isolation of concerns, zero data leakage, and rigorous statistical evaluation.

## Subpackage Responsibilities
- `ml/preprocessing/`:
  - Data ingestion, cleaning, schema enforcement, categorical encoding, and zero-leakage splitters (train, validation, test).
  - Scalers and transformers that fit strictly on training splits and transform evaluation/inference flows.
- `ml/features/`:
  - Domain-specific flow feature extraction (duration, packet rates, byte distributions, flow flags, inter-arrival times).
  - Feature selection and collinearity reduction routines.
- `ml/training/`:
  - Imbalance-aware model training pipelines (cost-sensitive learning, focal loss, SMOTE/class weighting on training splits only).
  - Model definitions: Scikit-learn baselines, Random Forest, XGBoost classifiers.
- `ml/evaluation/`:
  - Imbalance-appropriate evaluation metrics: Precision-Recall AUC (PR-AUC), F1-Score (macro/weighted), Expected Cost / Cost-Sensitive Loss, Confusion Matrices, and False Alarm Rate (FAR).
  - Threshold calibration (optimizing F-beta or business-defined operational constraints).
- `ml/explainability/`:
  - SHAP (SHapley Additive exPlanations) TreeExplainer integration for XGBoost and ensemble models.
  - Global feature importance and local instance-level attribution extraction.
- `ml/drift/`:
  - Statistical drift detection engines: Kolmogorov-Smirnov (KS) tests for continuous features, Chi-square for categoricals, and Population Stability Index (PSI).
  - Prediction confidence and concept drift alarms.
- `ml/registry/`:
  - Model versioning, serialization (Joblib/safetensors), metadata tracking, and champion/challenger rollback management.

## Engineering Principles
- **No In-place Overwriting**: Scalers and encoders are serialized alongside model weights into `artifacts/models/`.
- **Strict Separation**: Inference code never triggers training runs; retraining is explicitly governed.
- **Phase Boundary**: No models are trained in Phase 0.
