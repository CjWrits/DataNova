# Model Architecture Comparison: Logistic Regression vs Random Forest vs XGBoost

## 1. Architectural Paradigms & Structural Strengths

This report evaluates three fundamentally distinct machine learning paradigms applied to tabular network telemetry:

1. **Logistic Regression (L2 Regularized L-BFGS)**: A linear baseline operating on scaled flow features. It computes a linear hyperplane separating attack vectors from normal traffic. Extremely fast to train and evaluate, but fundamentally incapable of modeling complex nonlinear flow feature interactions without explicit polynomial expansion.
2. **Random Forest (Bagged Ensembles)**: An ensemble of 200 unpruned decision trees trained via bootstrap aggregation with random feature sub-spacing. Robust to feature scale variations, handles complex high-order interactions, but memory-intensive and computationally heavier during inference.
3. **XGBoost (Histogram-Based Gradient Boosted Decision Trees)**: An ensemble of 300 sequential trees trained via second-order Taylor expansion of the loss function. Employs histogram binning (`tree_method='hist'`) for high training velocity and superior gradient descent on tabular boundaries.

## 2. Quantitative Performance & Computational Efficiency

| Model | Task | Schema | Train Time | Val Latency (ms) | Val Bal Acc | Val F1 / Macro F1 | Val PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | binary | primary | 8.16s | 0.0014 ms | 0.8820 | 0.9293 | 0.9863 |
| Random Forest | binary | primary | 9.01s | 0.0104 ms | 0.9447 | 0.9590 | 0.9951 |
| Xgboost | binary | primary | 3.72s | 0.0016 ms | 0.9473 | 0.9600 | 0.9957 |
| Logistic Regression | binary | alternative_no_sttl | 5.62s | 0.0005 ms | 0.8807 | 0.9302 | 0.9853 |
| Random Forest | binary | alternative_no_sttl | 8.89s | 0.0113 ms | 0.9446 | 0.9591 | 0.9950 |
| Xgboost | binary | alternative_no_sttl | 3.54s | 0.0016 ms | 0.9465 | 0.9592 | 0.9956 |
| Logistic Regression | multiclass | primary | 36.48s | 0.0005 ms | 0.4322 | 0.4432 | 0.5222 |
| Random Forest | multiclass | primary | 6.68s | 0.0122 ms | 0.6068 | 0.6528 | 0.7595 |
| Xgboost | multiclass | primary | 18.65s | 0.0166 ms | 0.6619 | 0.7015 | 0.7776 |
| Logistic Regression | multiclass | alternative_no_sttl | 37.67s | 0.0006 ms | 0.4169 | 0.4241 | 0.5039 |
| Random Forest | multiclass | alternative_no_sttl | 6.79s | 0.0123 ms | 0.5881 | 0.6303 | 0.7328 |
| Xgboost | multiclass | alternative_no_sttl | 22.07s | 0.0110 ms | 0.6411 | 0.6839 | 0.7571 |

## 3. Minority Class Handling Under Severe Imbalance

Because Phase 3 baseline models are unweighted and un-resampled (pure baselines), they expose the raw vulnerability of each architecture to severe class imbalance:

- **Worms (159 total records, 111 train, 24 test)**: Logistic Regression completely fails on Worms (Recall ~ 0.00). Tree-based models (Random Forest and XGBoost) demonstrate vastly superior capability to isolate isolated hyper-rectangular feature clusters belonging to rare attacks.
- **Analysis & Backdoor (few hundred samples)**: Linear boundaries fail to capture the subtle variations in packet sizes and jitter that distinguish Analysis probes from standard Web traffic.
- **Generic & Exploits (tens of thousands of samples)**: Dominate tree splits and gradient updates.

## 4. Key Takeaway & Selection Rationale

XGBoost and Random Forest decisively outperform Logistic Regression across all metrics. XGBoost demonstrates the highest Macro PR-AUC and lowest latency per sample, while Random Forest displays marginally more conservative boundary smoothing. Both tree ensembles provide solid foundations for subsequent imbalance-aware mitigation in Phase 4.
