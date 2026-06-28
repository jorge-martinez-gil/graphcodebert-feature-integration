# The benchmark

FeatFuse evaluates **how engineered features complement pretrained code models**.
The first supported task is **code clone / similarity detection** on IR-Plag;
additional tasks and datasets are on the roadmap.

## Metrics

Reported for every method on the same footing:

| Group | Metrics |
|---|---|
| Threshold | accuracy, balanced accuracy, precision, recall, F1, MCC |
| Ranking | ROC-AUC, PR-AUC (average precision) |
| Calibration | Brier score, Expected Calibration Error (ECE) |
| Cost | training time, inference latency (per example) |

## Significance

Differences come with uncertainty: bootstrap confidence intervals on the primary
metric, McNemar's test and paired bootstrap difference tests for comparing two
methods on the same test set.

## Commands

```bash
featfuse run        -c configs/smoke.yaml        # full pipeline, CPU, seconds
featfuse run        -c configs/classical_features_irplag.yaml --ablate
featfuse ablate     -c configs/classical_features_irplag.yaml
featfuse importance -c configs/classical_features_irplag.yaml
featfuse list features      # discover available plugins
```
