"""Comprehensive evaluation metrics library for network intrusion detection.

Provides standardized, leakage-free metrics computation for binary and multiclass
intrusion detection tasks, with support for PR-AUC (Average Precision), ROC-AUC,
per-class recall/precision/F1, false positive rate analysis, and confusion matrices.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from ml.preprocessing.schema import (
    BINARY_LABEL_MAPPING,
    MULTICLASS_ID_TO_LABEL,
    MULTICLASS_LABEL_MAPPING,
)


def compute_binary_metrics(
    y_true: Union[np.ndarray, pd.Series, List[int]],
    y_pred: Union[np.ndarray, pd.Series, List[int]],
    y_prob: Optional[Union[np.ndarray, pd.Series, List[float]]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive binary classification evaluation metrics.

    Parameters:
        y_true: Ground truth binary labels (0 = Normal, 1 = Attack).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class (Attack = 1).
                Can be 1D array of positive probabilities or 2D array of shape (N, 2).

    Returns:
        Dict with accuracy, balanced_accuracy, precision, recall, specificity,
        false_positive_rate, f1, roc_auc, pr_auc, and confusion_matrix.
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_pred).astype(int)

    # Extract 1D positive class probability
    prob_pos: Optional[np.ndarray] = None
    if y_prob is not None:
        arr_prob = np.asarray(y_prob)
        if arr_prob.ndim == 2 and arr_prob.shape[1] == 2:
            prob_pos = arr_prob[:, 1]
        elif arr_prob.ndim == 1:
            prob_pos = arr_prob
        else:
            raise ValueError(f"Unexpected shape for binary y_prob: {arr_prob.shape}")

    # Confusion matrix
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Core rates
    acc = float(accuracy_score(y_t, y_p))
    bal_acc = float(balanced_accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, pos_label=1, zero_division=0.0))
    rec = float(recall_score(y_t, y_p, pos_label=1, zero_division=0.0))
    f1 = float(f1_score(y_t, y_p, pos_label=1, zero_division=0.0))

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Probabilistic curves (ROC-AUC, PR-AUC)
    roc_auc = float("nan")
    pr_auc = float("nan")
    if prob_pos is not None:
        try:
            roc_auc = float(roc_auc_score(y_t, prob_pos))
        except Exception:
            roc_auc = float("nan")

        try:
            pr_auc = float(average_precision_score(y_t, prob_pos))
        except Exception:
            pr_auc = float("nan")

    return {
        "accuracy": round(acc, 6),
        "balanced_accuracy": round(bal_acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "specificity": round(specificity, 6),
        "false_positive_rate": round(fpr, 6),
        "f1": round(f1, 6),
        "roc_auc": round(roc_auc, 6) if not np.isnan(roc_auc) else None,
        "pr_auc": round(pr_auc, 6) if not np.isnan(pr_auc) else None,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": [[tn, fp], [fn, tp]],
        },
        "support": {
            "normal": tn + fp,
            "attack": fn + tp,
            "total": len(y_t),
        },
    }


def compute_multiclass_metrics(
    y_true: Union[np.ndarray, pd.Series, List[Union[int, str]]],
    y_pred: Union[np.ndarray, pd.Series, List[Union[int, str]]],
    y_prob: Optional[Union[np.ndarray, pd.Series, List[List[float]]]] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive multiclass classification evaluation metrics.

    Supports integer-encoded or string-category labels. Evaluates macro-averaged,
    weighted, and per-class metrics across all 10 network attack families.

    Parameters:
        y_true: Ground truth labels (encoded 0..9 or strings).
        y_pred: Predicted labels (encoded 0..9 or strings).
        y_prob: 2D array of predicted class probabilities shape (N, n_classes).
        class_names: List of class names ordered by class index.

    Returns:
        Dict with accuracy, balanced_accuracy, macro/weighted precision/recall/F1,
        macro_pr_auc, macro_roc_auc, per_class breakdown, and 10x10 confusion matrix.
    """
    if class_names is None:
        class_names = list(MULTICLASS_LABEL_MAPPING.keys())

    # Map string categories to integers if necessary
    def _to_int(y_arr: Union[np.ndarray, pd.Series, List[Any]]) -> np.ndarray:
        arr = np.asarray(y_arr)
        if arr.dtype.kind in {"U", "O", "S"}:
            return np.array([MULTICLASS_LABEL_MAPPING.get(val, -1) for val in arr], dtype=int)
        return arr.astype(int)

    y_t = _to_int(y_true)
    y_p = _to_int(y_pred)
    n_classes = len(class_names)
    labels = list(range(n_classes))

    # Overall metrics
    acc = float(accuracy_score(y_t, y_p))
    bal_acc = float(balanced_accuracy_score(y_t, y_p))
    macro_prec = float(precision_score(y_t, y_p, labels=labels, average="macro", zero_division=0.0))
    macro_rec = float(recall_score(y_t, y_p, labels=labels, average="macro", zero_division=0.0))
    macro_f1 = float(f1_score(y_t, y_p, labels=labels, average="macro", zero_division=0.0))
    weighted_f1 = float(f1_score(y_t, y_p, labels=labels, average="weighted", zero_division=0.0))

    # Per-class metrics
    per_cls_prec = precision_score(y_t, y_p, labels=labels, average=None, zero_division=0.0)
    per_cls_rec = recall_score(y_t, y_p, labels=labels, average=None, zero_division=0.0)
    per_cls_f1 = f1_score(y_t, y_p, labels=labels, average=None, zero_division=0.0)

    # Confusion matrix
    cm = confusion_matrix(y_t, y_p, labels=labels)
    supports = cm.sum(axis=1)

    per_class_dict: Dict[str, Dict[str, Any]] = {}
    for idx, cname in enumerate(class_names):
        tp = int(cm[idx, idx])
        fn = int(supports[idx] - tp)
        fp = int(cm[:, idx].sum() - tp)
        tn = int(len(y_t) - (tp + fp + fn))

        per_class_dict[cname] = {
            "class_id": idx,
            "precision": round(float(per_cls_prec[idx]), 6),
            "recall": round(float(per_cls_rec[idx]), 6),
            "f1": round(float(per_cls_f1[idx]), 6),
            "support": int(supports[idx]),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    # Macro ROC-AUC and PR-AUC
    macro_roc_auc: Optional[float] = None
    macro_pr_auc: Optional[float] = None

    if y_prob is not None:
        prob_mat = np.asarray(y_prob)
        if prob_mat.shape[1] == n_classes:
            try:
                y_t_bin = label_binarize(y_t, classes=labels)
                macro_roc_auc = float(roc_auc_score(y_t_bin, prob_mat, average="macro", multi_class="ovr"))
            except Exception:
                macro_roc_auc = None

            try:
                y_t_bin = label_binarize(y_t, classes=labels)
                pr_scores = []
                for c in range(n_classes):
                    if y_t_bin[:, c].sum() > 0:
                        pr_scores.append(average_precision_score(y_t_bin[:, c], prob_mat[:, c]))
                if pr_scores:
                    macro_pr_auc = float(np.mean(pr_scores))
            except Exception:
                macro_pr_auc = None

    return {
        "accuracy": round(acc, 6),
        "balanced_accuracy": round(bal_acc, 6),
        "macro_precision": round(macro_prec, 6),
        "macro_recall": round(macro_rec, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "macro_roc_auc": round(macro_roc_auc, 6) if macro_roc_auc is not None else None,
        "macro_pr_auc": round(macro_pr_auc, 6) if macro_pr_auc is not None else None,
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist(),
        "class_labels": class_names,
    }


def format_binary_metrics_markdown(metrics: Dict[str, Any], title: str = "Binary Metrics") -> str:
    """Format binary metrics as a clean Markdown table."""
    cm = metrics.get("confusion_matrix", {})
    return (
        f"### {title}\n\n"
        f"| Metric | Value |\n"
        f"| :--- | :--- |\n"
        f"| Accuracy | {metrics.get('accuracy', 0.0):.4f} |\n"
        f"| Balanced Accuracy | {metrics.get('balanced_accuracy', 0.0):.4f} |\n"
        f"| Precision (Attack) | {metrics.get('precision', 0.0):.4f} |\n"
        f"| Recall / Sensitivity | {metrics.get('recall', 0.0):.4f} |\n"
        f"| Specificity (Normal) | {metrics.get('specificity', 0.0):.4f} |\n"
        f"| False Positive Rate (FPR) | {metrics.get('false_positive_rate', 0.0):.4f} |\n"
        f"| F1-Score | {metrics.get('f1', 0.0):.4f} |\n"
        f"| PR-AUC (Average Precision) | {metrics.get('pr_auc') if metrics.get('pr_auc') is not None else 'N/A'} |\n"
        f"| ROC-AUC | {metrics.get('roc_auc') if metrics.get('roc_auc') is not None else 'N/A'} |\n"
        f"| Confusion Matrix (TN / FP / FN / TP) | {cm.get('tn', 0):,d} / {cm.get('fp', 0):,d} / {cm.get('fn', 0):,d} / {cm.get('tp', 0):,d} |\n"
    )


def format_multiclass_metrics_markdown(metrics: Dict[str, Any], title: str = "Multiclass Metrics") -> str:
    """Format multiclass metrics and per-class breakdown as a clean Markdown table."""
    lines = [
        f"### {title}\n",
        f"**Summary Metrics:**",
        f"- **Accuracy**: {metrics.get('accuracy', 0.0):.4f}",
        f"- **Balanced Accuracy**: {metrics.get('balanced_accuracy', 0.0):.4f}",
        f"- **Macro F1**: {metrics.get('macro_f1', 0.0):.4f}",
        f"- **Weighted F1**: {metrics.get('weighted_f1', 0.0):.4f}",
        f"- **Macro PR-AUC**: {metrics.get('macro_pr_auc') if metrics.get('macro_pr_auc') is not None else 'N/A'}",
        f"- **Macro ROC-AUC**: {metrics.get('macro_roc_auc') if metrics.get('macro_roc_auc') is not None else 'N/A'}\n",
        f"| Class | Support | Precision | Recall | F1-Score |",
        f"| :--- | :--- | :--- | :--- | :--- |",
    ]
    per_class = metrics.get("per_class", {})
    for cname, vals in per_class.items():
        lines.append(
            f"| **{cname}** | {vals.get('support', 0):,d} | {vals.get('precision', 0.0):.4f} | "
            f"{vals.get('recall', 0.0):.4f} | {vals.get('f1', 0.0):.4f} |"
        )
    return "\n".join(lines) + "\n"
