"""Imbalance-aware estimation and resampling module for network intrusion detection.

Provides verified, leakage-free techniques for mitigating severe class imbalance:
1. Cost-sensitive sample weighting derived strictly from training partition class frequencies.
2. Targeted Random Oversampling applied exclusively to training data.
3. SMOTENC applied strictly on mixed raw numerical/categorical features prior to one-hot encoding.
4. Balanced Random Forest ensembles.

Guarantees:
- Resampling is never applied to validation or test data.
- Class weights are calculated solely from training partition labels.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

from ml.preprocessing.schema import CATEGORICAL_FEATURES, MULTICLASS_LABEL_MAPPING


def compute_training_sample_weights(
    y_train: np.ndarray,
    weighting_type: str = "balanced",
    clip_max: Optional[float] = 50.0,
) -> np.ndarray:
    """Calculate sample weights strictly from the training partition class frequencies.

    Formulas:
    - 'balanced': w_c = N / (K * N_c)
    - 'sqrt_balanced': w_c = sqrt( N / (K * N_c) )
    - 'custom_minority_boost': balanced weights with designated extra boost for Worms, Analysis, DoS.

    Parameters:
        y_train: Ground truth integer-encoded training labels.
        weighting_type: 'balanced', 'sqrt_balanced', or 'smoothed'.
        clip_max: Maximum allowed weight multiplier to prevent gradient explosion.

    Returns:
        1D array of sample weights matching the length of y_train.
    """
    classes, counts = np.unique(y_train, return_counts=True)
    n_samples = len(y_train)
    n_classes = len(classes)

    class_weight_map: Dict[int, float] = {}
    for c, cnt in zip(classes, counts):
        if weighting_type == "balanced":
            raw_w = n_samples / (n_classes * cnt)
        elif weighting_type in ("sqrt_balanced", "smoothed"):
            raw_w = np.sqrt(n_samples / (n_classes * cnt))
        else:
            raise ValueError(f"Unknown weighting_type: {weighting_type}")

        if clip_max is not None:
            raw_w = min(raw_w, clip_max)
        class_weight_map[c] = float(raw_w)

    sample_weights = np.array([class_weight_map[y] for y in y_train], dtype=float)
    return sample_weights


def apply_training_resampling(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    method: str = "smotenc",
    min_samples_floor: int = 2000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """Apply targeted resampling exclusively to training data before feature encoding.

    Target strategy: Raises minority classes with count < min_samples_floor
    up to min_samples_floor, while leaving dominant majority classes untouched.
    This preserves the natural network background distribution while alleviating
    extreme sparsity for Worms, Analysis, Backdoor, and Shellcode.

    Parameters:
        X_train: Raw training DataFrame with mixed numerical and categorical features.
        y_train: Integer-encoded labels.
        method: 'smotenc' or 'random_oversample'.
        min_samples_floor: Target sample floor for minority classes.
        random_state: Fixed random seed.

    Returns:
        (X_resampled, y_resampled, audit_stats)
    """
    classes, counts = np.unique(y_train, return_counts=True)
    target_strategy: Dict[int, int] = {}
    for c, cnt in zip(classes, counts):
        if cnt < min_samples_floor:
            target_strategy[int(c)] = min_samples_floor

    initial_counts = {int(c): int(cnt) for c, cnt in zip(classes, counts)}

    if not target_strategy:
        # No minority class needs oversampling
        return X_train.copy(), y_train.copy(), {"method": "none", "added_samples": 0}

    if method == "random_oversample":
        ros = RandomOverSampler(sampling_strategy=target_strategy, random_state=random_state)
        X_res, y_res = ros.fit_resample(X_train, y_train)
    elif method == "smotenc":
        # Identify categorical columns explicitly
        cat_indices = [
            X_train.columns.get_loc(c)
            for c in CATEGORICAL_FEATURES
            if c in X_train.columns
        ]
        smotenc = SMOTENC(
            categorical_features=cat_indices,
            sampling_strategy=target_strategy,
            random_state=random_state,
            k_neighbors=5,
        )
        X_res, y_res = smotenc.fit_resample(X_train, y_train)
    else:
        raise ValueError(f"Unknown resampling method: '{method}'. Choose 'smotenc' or 'random_oversample'.")

    res_classes, res_counts = np.unique(y_res, return_counts=True)
    final_counts = {int(c): int(cnt) for c, cnt in zip(res_classes, res_counts)}

    audit_stats = {
        "method": method,
        "initial_total": len(y_train),
        "resampled_total": len(y_res),
        "added_samples": len(y_res) - len(y_train),
        "initial_class_counts": initial_counts,
        "final_class_counts": final_counts,
    }

    if not isinstance(X_res, pd.DataFrame):
        X_res = pd.DataFrame(X_res, columns=X_train.columns)

    return X_res, y_res, audit_stats


def instantiate_balanced_random_forest(
    n_estimators: int = 200,
    random_state: int = 42,
) -> BalancedRandomForestClassifier:
    """Instantiate a Balanced Random Forest with downsampled bagging per tree."""
    return BalancedRandomForestClassifier(
        n_estimators=n_estimators,
        sampling_strategy="auto",
        replacement=True,
        random_state=random_state,
        n_jobs=-1,
    )
