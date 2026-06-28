# Reproducibility

Every run is fully traceable. `featfuse run -c <config>` writes:

```
runs/<name>/
  manifest.json      seed, config hash, git commit, platform, library versions
  results.json       metrics (+ bootstrap CIs) for each split
  results_table.md   auto-generated Markdown table
  results_table.tex  auto-generated LaTeX (booktabs) table
  REPORT.md          human-readable report linking the figures
  figures/*.png      calibration, metric comparison, feature correlation
```

Principles:
- **One command, one config.** A YAML file is a complete description of an experiment.
- **Deterministic seeds** across Python, NumPy and (when present) PyTorch.
- **No fabricated numbers.** The repository ships the *machinery*; reported results
  are produced by running it. The neural GraphCodeBERT result is reproduced with
  `configs/graphcodebert_irplag.yaml` (needs `featfuse[neural]` + a GPU); the
  GPU-free engineered-feature baseline with `configs/classical_features_irplag.yaml`.
- **Manifests pin provenance.** Compare two `manifest.json` files to see exactly what
  differed between runs (seed, code commit, configuration).

To reproduce the engineered-feature baseline locally:

```bash
pip install -e ".[dev]"
featfuse run -c configs/classical_features_irplag.yaml --ablate
cat runs/classical_features_irplag/REPORT.md
```
