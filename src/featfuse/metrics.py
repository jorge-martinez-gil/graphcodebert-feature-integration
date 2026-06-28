"""Evaluation metrics for code-pair classification.

A single :func:`classification_metrics` call returns the full benchmark metric
suite so that every method is reported on the same footing:

accuracy, precision, recall, F1, Matthews correlation (MCC), balanced accuracy,
ROC-AUC, PR-AUC (average precision), Brier score and Expected Calibration Error
(ECE). Threshold-free metrics (ROC/PR-AUC, Brier, ECE) are computed only when
probability scores are supplied.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        matthews_corrcoef,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    _SK = True
except Exception:  # pragma: no cover
    _SK = False


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width probability bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def _prf(y_true, y_pred, positive_label):
    if _SK:
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label=positive_label, zero_division=0
        )
        return float(p), float(r), float(f)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tp = int(((y_pred == positive_label) & (y_true == positive_label)).sum())
    fp = int(((y_pred == positive_label) & (y_true != positive_label)).sum())
    fn = int(((y_pred != positive_label) & (y_true == positive_label)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Optional[Sequence[float]] = None,
    positive_label: int = 1,
) -> Dict[str, float]:
    """Return the full metric suite as a flat dict of floats."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p, r, f = _prf(y_true, y_pred, positive_label)

    if _SK:
        acc = float(accuracy_score(y_true, y_pred))
        bal = float(balanced_accuracy_score(y_true, y_pred))
        mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0
    else:
        acc = float((y_true == y_pred).mean())
        bal = acc
        mcc = 0.0

    out: Dict[str, float] = {
        "accuracy": acc,
        "balanced_accuracy": bal,
        "precision": p,
        "recall": r,
        "f1": f,
        "mcc": mcc,
        "n": int(len(y_true)),
    }

    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        if len(np.unique(y_true)) > 1 and _SK:
            try:
                out["roc_auc"] = float(roc_auc_score(y_true, y_score))
                out["pr_auc"] = float(average_precision_score(y_true, y_score))
            except Exception:
                pass
        if _SK:
            try:
                out["brier"] = float(brier_score_loss(y_true, y_score))
            except Exception:
                pass
        out["ece"] = expected_calibration_error(y_true, y_score)
    return out


METRIC_KEYS = [
    "accuracy", "balanced_accuracy", "precision", "recall", "f1",
    "mcc", "roc_auc", "pr_auc", "brier", "ece",
]
