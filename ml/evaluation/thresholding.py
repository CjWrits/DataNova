"""Threshold calibration and operational operating-point analysis for intrusion detection.

Provides validation-driven threshold optimization:
- Binary threshold sweeps for F1-maximization, cost-weighted loss minimization,
  and constraint-bounded false positive rates (e.g., FPR <= 1% or <= 5%).
- Multiclass confidence margin adjustments and class-conditional decision thresholds
  to rescue high-severity minority attacks (Worms, Analysis, DoS) while measuring
  the explicit trade-off on benign false alarms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from ml.evaluation.metrics import compute_binary_metrics, compute_multiclass_metrics
from ml.preprocessing.schema import MULTICLASS_LABEL_MAPPING


@dataclass
class ThresholdPoint:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    fpr: float
    specificity: float
    cost: float
    tn: int
    fp: int
    fn: int
    tp: int


def sweep_binary_thresholds(
    y_val_true: np.ndarray,
    y_val_prob: np.ndarray,
    cost_fn_ratio: float = 5.0,
    steps: int = 100,
) -> Tuple[List[ThresholdPoint], Dict[str, ThresholdPoint]]:
    """Sweep decision thresholds across [0.01, 0.99] strictly on validation partition.

    Parameters:
        y_val_true: Ground truth binary labels (0 = Normal, 1 = Attack).
        y_val_prob: Predicted probability for positive class (Attack).
        cost_fn_ratio: Relative cost penalty of a False Negative (missed attack)
                       compared to a False Positive (benign false alarm). Cost = FP + cost_fn_ratio * FN.
        steps: Number of threshold steps to evaluate.

    Returns:
        (all_points, selected_optimal_points)
    """
    thresholds = np.linspace(0.01, 0.99, steps)
    points: List[ThresholdPoint] = []

    y_t = np.asarray(y_val_true).astype(int)
    probs = np.asarray(y_val_prob)
    if probs.ndim == 2:
        probs = probs[:, 1]

    for th in thresholds:
        preds = (probs >= th).astype(int)
        cm = confusion_matrix(y_t, preds, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        acc = (tp + tn) / len(y_t)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
        bal_acc = (rec + spec) / 2.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        cost = fp + cost_fn_ratio * fn

        points.append(
            ThresholdPoint(
                threshold=round(float(th), 4),
                accuracy=round(float(acc), 6),
                precision=round(float(prec), 6),
                recall=round(float(rec), 6),
                f1=round(float(f1), 6),
                balanced_accuracy=round(float(bal_acc), 6),
                fpr=round(float(fpr), 6),
                specificity=round(float(spec), 6),
                cost=round(float(cost), 2),
                tn=tn,
                fp=fp,
                fn=fn,
                tp=tp,
            )
        )

    # Identify optimal operating points
    # 1. Default threshold (closest to 0.50)
    pt_default = min(points, key=lambda p: abs(p.threshold - 0.50))

    # 2. F1-optimal threshold
    pt_f1_opt = max(points, key=lambda p: p.f1)

    # 3. Cost-optimal threshold (minimizing FP + cost_fn_ratio * FN)
    pt_cost_opt = min(points, key=lambda p: p.cost)

    # 4. Low-False-Alarm threshold (FPR <= 2% while maximizing recall)
    low_fpr_candidates = [p for p in points if p.fpr <= 0.02]
    pt_low_fpr = max(low_fpr_candidates, key=lambda p: p.recall) if low_fpr_candidates else pt_default

    # 5. High-Recall threshold (Recall >= 98% while minimizing FPR)
    high_rec_candidates = [p for p in points if p.recall >= 0.98]
    pt_high_rec = min(high_rec_candidates, key=lambda p: p.fpr) if high_rec_candidates else pt_default

    optimal_points = {
        "default": pt_default,
        "f1_optimal": pt_f1_opt,
        "cost_optimal": pt_cost_opt,
        "low_false_alarm": pt_low_fpr,
        "high_recall": pt_high_rec,
    }

    return points, optimal_points


def calibrate_multiclass_prior_multipliers(
    y_val_true: np.ndarray,
    val_probs: np.ndarray,
    class_names: List[str],
    boost_classes: Dict[str, float],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply calibrated prior odds multipliers to designated minority classes.

    Adjusts decision rule:
        y_pred = argmax_c ( P(y=c | x) * multiplier_c )

    Parameters:
        y_val_true: Ground truth encoded labels (0..9).
        val_probs: Raw output probabilities from model.predict_proba.
        class_names: Class names indexed 0..9.
        boost_classes: Dict mapping class names to positive multipliers (e.g., {'Worms': 2.5, 'Analysis': 2.0}).

    Returns:
        (adjusted_predictions, adjusted_metrics)
    """
    multipliers = np.ones(len(class_names), dtype=float)
    for cname, mult in boost_classes.items():
        if cname in MULTICLASS_LABEL_MAPPING:
            idx = MULTICLASS_LABEL_MAPPING[cname]
            multipliers[idx] = mult

    # Scale probabilities by multipliers and re-evaluate argmax
    adjusted_probs = val_probs * multipliers
    adjusted_preds = np.argmax(adjusted_probs, axis=1)

    metrics = compute_multiclass_metrics(
        y_true=y_val_true,
        y_pred=adjusted_preds,
        y_prob=val_probs,  # Keep unadjusted calibrated probabilities for AUCPR
        class_names=class_names,
    )
    return adjusted_preds, metrics
