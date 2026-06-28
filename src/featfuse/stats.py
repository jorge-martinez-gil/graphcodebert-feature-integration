"""Statistical significance utilities for honest benchmarking.

A benchmark is only credible if differences between methods come with uncertainty
estimates. This module provides:

* :func:`bootstrap_metric_ci` — bootstrap confidence interval for any metric.
* :func:`mcnemar_test` — paired test for whether two classifiers differ on the same
  test set (the appropriate test for comparing two models on identical examples).
* :func:`paired_bootstrap_diff` — bootstrap CI and p-value for the *difference*
  between two methods on the same test set.
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import numpy as np


def bootstrap_metric_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: Sequence,
    y_pred_or_score: Sequence,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Percentile bootstrap CI for ``metric_fn(y_true, y_pred_or_score)``."""
    y_true = np.asarray(y_true)
    y = np.asarray(y_pred_or_score)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = float(metric_fn(y_true, y))
    if n == 0:
        return {"point": point, "low": point, "high": point, "std": 0.0}
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            stats[b] = metric_fn(y_true[idx], y[idx])
        except Exception:
            stats[b] = np.nan
    stats = stats[~np.isnan(stats)]
    alpha = (1.0 - confidence) / 2.0
    low, high = np.percentile(stats, [100 * alpha, 100 * (1 - alpha)])
    return {"point": point, "low": float(low), "high": float(high), "std": float(stats.std())}


def mcnemar_test(y_true: Sequence, pred_a: Sequence, pred_b: Sequence) -> Dict[str, float]:
    """McNemar's paired test comparing two classifiers on the same examples.

    Uses the exact binomial test on discordant pairs (robust for small samples);
    returns the discordant counts, statistic and two-sided p-value.
    """
    y_true = np.asarray(y_true)
    a = np.asarray(pred_a) == y_true
    b = np.asarray(pred_b) == y_true
    n01 = int(np.sum(a & ~b))   # A right, B wrong
    n10 = int(np.sum(~a & b))   # A wrong, B right
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "statistic": 0.0, "p_value": 1.0}
    try:
        from scipy.stats import binomtest

        p = binomtest(min(n01, n10), n, 0.5, alternative="two-sided").pvalue
    except Exception:
        from math import comb

        k = min(n01, n10)
        p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    stat = (abs(n01 - n10) - 1) ** 2 / n  # continuity-corrected chi-square
    return {"n01": n01, "n10": n10, "statistic": float(stat), "p_value": float(p)}


def paired_bootstrap_diff(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: Sequence,
    pred_a: Sequence,
    pred_b: Sequence,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap CI and p-value for ``metric(A) - metric(B)`` on the same test set."""
    y_true = np.asarray(y_true)
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = float(metric_fn(y_true, a) - metric_fn(y_true, b))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = metric_fn(y_true[idx], a[idx]) - metric_fn(y_true[idx], b[idx])
    alpha = (1.0 - confidence) / 2.0
    low, high = np.percentile(diffs, [100 * alpha, 100 * (1 - alpha)])
    # two-sided bootstrap p-value for H0: diff == 0
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": point, "low": float(low), "high": float(high), "p_value": float(min(1.0, p))}
