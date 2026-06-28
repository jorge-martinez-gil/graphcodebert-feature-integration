# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/) and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [0.2.0] — Research infrastructure release

Turns the single-paper script into a reusable, extensible benchmark platform.

### Added
- **Installable package** `featfuse` with a `featfuse` command-line interface
  (`run`, `ablate`, `importance`, `list`, `info`) and a plugin **registry** for
  features, fusion strategies, models and datasets.
- **11 engineered features** across lexical, structural and execution families
  (generalising the paper's single execution-similarity feature). Features are
  pairwise, deterministic and reproducible without a JVM.
- **5 neural fusion strategies**: `concat` (the paper's design), `gated`,
  `attention`, `residual`, `film`. The classification head now accepts an
  arbitrary feature vector + a swappable fusion module.
- **Encoders**: HuggingFace wrapper (GraphCodeBERT/CodeBERT/UniXcoder/…) plus a
  dependency-free `ToyEncoder` so the pipeline runs without GPUs or downloads.
- **Full metric suite**: accuracy, precision, recall, F1, MCC, balanced accuracy,
  ROC-AUC, PR-AUC, Brier score and Expected Calibration Error.
- **Statistics**: bootstrap confidence intervals, McNemar's test and paired
  bootstrap difference tests.
- **Automatic ablation studies** (leave-one-out + single-feature) and
  **feature importance** (permutation + native).
- **Auto-generated outputs**: Markdown + LaTeX (booktabs) tables, publication-quality
  figures (calibration, metric comparison, feature correlation) and a `REPORT.md`.
- **Reproducibility**: deterministic seeding and a `manifest.json` per run recording
  seed, config hash, git commit, platform and library versions.
- **Tooling**: tests (pytest), GitHub Actions CI (Py 3.9–3.12), Dockerfile, Makefile,
  issue/PR templates, `CITATION.cff`/`CITATION.bib`, and how-to documentation.

### Fixed
- The runner no longer ignores the published fixed splits and re-splits randomly;
  splits are loaded deterministically and *enriched* with the precomputed execution
  feature that the split files omitted.
- Removed the hard-coded Windows dataset path; data location is configuration-driven.

### Changed
- The original training script and notebook moved to `legacy/` for provenance.
