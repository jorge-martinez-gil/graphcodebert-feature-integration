import numpy as np

from featfuse.metrics import classification_metrics, expected_calibration_error
from featfuse.stats import bootstrap_metric_ci, mcnemar_test, paired_bootstrap_diff


def test_perfect_predictions():
    y = [0, 1, 0, 1, 1]
    m = classification_metrics(y, y, [0.0, 1.0, 0.0, 1.0, 1.0])
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["mcc"] == 1.0
    assert m["roc_auc"] == 1.0


def test_metric_keys_present_with_scores():
    rng = np.random.default_rng(0)
    yt = rng.integers(0, 2, 100)
    ys = np.clip(yt * 0.6 + rng.normal(0, 0.3, 100) + 0.2, 0, 1)
    m = classification_metrics(yt, (ys >= 0.5).astype(int), ys)
    for k in ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc", "brier", "ece"]:
        assert k in m


def test_ece_zero_for_perfect_calibration():
    # probabilities exactly equal to empirical accuracy in each bin
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, p) == 0.0


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(1)
    yt = rng.integers(0, 2, 120)
    pred = np.where(rng.random(120) < 0.8, yt, 1 - yt)
    f1 = lambda t, p: classification_metrics(t, p)["f1"]
    ci = bootstrap_metric_ci(f1, yt, pred, n_boot=300, seed=0)
    assert ci["low"] <= ci["point"] <= ci["high"]


def test_mcnemar_detects_difference():
    rng = np.random.default_rng(2)
    yt = rng.integers(0, 2, 200)
    good = np.where(rng.random(200) < 0.95, yt, 1 - yt)
    bad = np.where(rng.random(200) < 0.55, yt, 1 - yt)
    res = mcnemar_test(yt, good, bad)
    assert res["p_value"] < 0.05


def test_paired_bootstrap_diff_sign():
    rng = np.random.default_rng(3)
    yt = rng.integers(0, 2, 200)
    good = np.where(rng.random(200) < 0.9, yt, 1 - yt)
    bad = np.where(rng.random(200) < 0.6, yt, 1 - yt)
    f1 = lambda t, p: classification_metrics(t, p)["f1"]
    d = paired_bootstrap_diff(f1, yt, good, bad, n_boot=300)
    assert d["diff"] > 0
