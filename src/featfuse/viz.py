"""Publication-quality visualizations.

All functions use a non-interactive backend, a consistent serif/colourblind-safe
style, and save high-DPI figures suitable for papers and slides. Each returns the
path it wrote so the experiment runner can collect them into a report.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Colourblind-safe qualitative palette (Wong, 2011).
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.axisbelow": True,
    })


def plot_feature_importance(names: Sequence[str], importances: Sequence[float], path: str,
                            title: str = "Feature importance") -> str:
    set_style()
    order = np.argsort(importances)
    names = [names[i] for i in order]
    vals = np.asarray(importances)[order]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(names))))
    ax.barh(range(len(names)), vals, color=PALETTE[0])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("importance")
    ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_metric_bars(method_metrics: Dict[str, Dict[str, float]], metrics: Sequence[str], path: str,
                     title: str = "Method comparison") -> str:
    set_style()
    methods = list(method_metrics)
    x = np.arange(len(metrics))
    w = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(1.6 * len(metrics) + 2, 4))
    for i, m in enumerate(methods):
        vals = [method_metrics[m].get(k, 0.0) for k in metrics]
        ax.bar(x + i * w, vals, w, label=m, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x + w * (len(methods) - 1) / 2)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9, ncol=min(3, len(methods)))
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_calibration(y_true: Sequence[int], y_prob: Sequence[float], path: str, n_bins: int = 10,
                     title: str = "Reliability diagram") -> str:
    set_style()
    y_true = np.asarray(y_true, float); y_prob = np.asarray(y_prob, float)
    bins = np.linspace(0, 1, n_bins + 1)
    centers, accs = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob > lo) & (y_prob <= hi)
        if m.any():
            centers.append(y_prob[m].mean()); accs.append(y_true[m].mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.plot(centers, accs, "o-", color=PALETTE[1], label="model")
    ax.set_xlabel("predicted probability"); ax.set_ylabel("empirical accuracy")
    ax.set_title(title); ax.legend(frameon=False)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_feature_correlation(matrix: np.ndarray, names: Sequence[str], path: str,
                             title: str = "Feature correlation") -> str:
    set_style()
    matrix = np.asarray(matrix, float)
    corr = np.corrcoef(matrix, rowvar=False)
    corr = np.nan_to_num(corr)
    fig, ax = plt.subplots(figsize=(0.6 * len(names) + 2, 0.6 * len(names) + 2))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_cost_vs_performance(points: Dict[str, Dict[str, float]], path: str,
                             x_key: str = "runtime_s", y_key: str = "f1",
                             title: str = "Cost vs. performance") -> str:
    set_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, (name, d) in enumerate(points.items()):
        ax.scatter(d.get(x_key, 0), d.get(y_key, 0), s=80, color=PALETTE[i % len(PALETTE)])
        ax.annotate(name, (d.get(x_key, 0), d.get(y_key, 0)), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel(x_key); ax.set_ylabel(y_key); ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)
    return path
