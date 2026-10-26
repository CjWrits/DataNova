# Baseline Machine Learning Experiment Report (Phase 3)

## 1. Executive Summary

Phase 3 establishes rigorous, unweighted baseline models across two classification tasks (Binary and Multiclass) and two feature representations (Primary with all 42 features vs. Ablation with 41 features excluding the synthetic artifact `sttl`). Exactly 12 baseline experiments were executed with fixed baseline hyperparameters and zero data leakage.

### Model Selection Protocol & Test Partition Integrity

Model selection was conducted strictly using the validation partition. Test partition evaluations were performed exclusively on the frozen selected winning configurations. No test set leakage occurred at any stage.

## 2. Frozen Selected Winning Configurations

| Configuration | Winner Model | Val F1 / Macro F1 | Val PR-AUC | Test F1 / Macro F1 | Test PR-AUC | Test Latency (ms/sample) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **binary_primary** | Xgboost | 0.9600 | 0.9957 | **0.9595** | **0.9956** | 0.0037 |
| **binary_alternative_no_sttl** | Xgboost | 0.9592 | 0.9956 | **0.9594** | **0.9955** | 0.0038 |
| **multiclass_primary** | Xgboost | 0.7015 | 0.7776 | **0.7207** | **0.7879** | 0.0340 |
| **multiclass_alternative_no_sttl** | Xgboost | 0.6839 | 0.7571 | **0.6957** | **0.7623** | 0.0340 |

## 3. Complete 12-Experiment Validation Leaderboard

| ID | Task | Schema | Model | Train Time (s) | Val Acc | Val Bal Acc | Val F1 / Macro F1 | Val PR-AUC | Val FPR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `binary_primary_logistic_regression` | binary | primary | Logistic Regression | 8.2s | 0.9057 | 0.8820 | **0.9293** | 0.9863 | 0.2024 |
| `binary_primary_random_forest` | binary | primary | Random Forest | 9.0s | 0.9477 | 0.9447 | **0.9590** | 0.9951 | 0.0660 |
| `binary_primary_xgboost` | binary | primary | Xgboost | 3.7s | 0.9491 | 0.9473 | **0.9600** | 0.9957 | 0.0593 |
| `binary_no_sttl_logistic_regression` | binary | alternative_no_sttl | Logistic Regression | 5.6s | 0.9065 | 0.8807 | **0.9302** | 0.9853 | 0.2108 |
| `binary_no_sttl_random_forest` | binary | alternative_no_sttl | Random Forest | 8.9s | 0.9477 | 0.9446 | **0.9591** | 0.9950 | 0.0664 |
| `binary_no_sttl_xgboost` | binary | alternative_no_sttl | Xgboost | 3.5s | 0.9482 | 0.9465 | **0.9592** | 0.9956 | 0.0594 |
| `multiclass_primary_logistic_regression` | multiclass | primary | Logistic Regression | 36.5s | 0.8524 | 0.4322 | **0.4432** | 0.5222 | N/A |
| `multiclass_primary_random_forest` | multiclass | primary | Random Forest | 6.7s | 0.9084 | 0.6068 | **0.6528** | 0.7595 | N/A |
| `multiclass_primary_xgboost` | multiclass | primary | Xgboost | 18.7s | 0.9093 | 0.6619 | **0.7015** | 0.7776 | N/A |
| `multiclass_no_sttl_logistic_regression` | multiclass | alternative_no_sttl | Logistic Regression | 37.7s | 0.8427 | 0.4169 | **0.4241** | 0.5039 | N/A |
| `multiclass_no_sttl_random_forest` | multiclass | alternative_no_sttl | Random Forest | 6.8s | 0.9070 | 0.5881 | **0.6303** | 0.7328 | N/A |
| `multiclass_no_sttl_xgboost` | multiclass | alternative_no_sttl | Xgboost | 22.1s | 0.9095 | 0.6411 | **0.6839** | 0.7571 | N/A |

## 4. Multiclass Per-Class Breakdown (Primary Schema Winners)

### Test Set Per-Class Breakdown (Xgboost)

**Summary Metrics:**
- **Accuracy**: 0.9116
- **Balanced Accuracy**: 0.6861
- **Macro F1**: 0.7207
- **Weighted F1**: 0.9077
- **Macro PR-AUC**: 0.787871
- **Macro ROC-AUC**: 0.99185

| Class | Support | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | 13,833 | 0.9132 | 0.9496 | 0.9310 |
| **Generic** | 8,872 | 0.9975 | 0.9872 | 0.9923 |
| **Exploits** | 4,183 | 0.8448 | 0.9211 | 0.8813 |
| **Fuzzers** | 3,119 | 0.7548 | 0.6505 | 0.6988 |
| **DoS** | 548 | 0.7355 | 0.3248 | 0.4506 |
| **Reconnaissance** | 1,960 | 0.9373 | 0.9224 | 0.9298 |
| **Analysis** | 90 | 0.6176 | 0.2333 | 0.3387 |
| **Backdoor** | 53 | 0.7895 | 0.5660 | 0.6593 |
| **Shellcode** | 221 | 0.7545 | 0.7647 | 0.7596 |
| **Worms** | 24 | 0.5909 | 0.5417 | 0.5652 |

## 5. Diagnostic Visualizations Generated

- Precision-Recall Curves: `research/visualizations/pr_curves_binary.png`
- ROC Curves: `research/visualizations/roc_curves_binary.png`
- STTL Impact Comparison: `research/visualizations/sttl_impact_comparison.png`
- Multiclass Per-Class F1: `research/visualizations/multiclass_per_class_f1.png`
- Top 20 Flow Features: `research/visualizations/feature_importances_top20.png`
- Test Partition Confusion Matrices: `research/visualizations/confusion_matrices_baseline.png`
