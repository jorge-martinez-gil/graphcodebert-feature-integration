<div align="center">

# FeatFuse

### Feature integration for GraphCodeBERT and other code language models

An open, reproducible benchmark for measuring whether engineered lexical, structural, and execution-based features improve code similarity detection when fused with pretrained code transformers.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/arXiv-2408.08903-b31b1b.svg)](https://arxiv.org/abs/2408.08903)
[![Cite](https://img.shields.io/badge/Cite-BibTeX-blue.svg)](#citation)

[![GitHub stars](https://img.shields.io/github/stars/jorge-martinez-gil/graphcodebert-feature-integration)](https://github.com/jorge-martinez-gil/graphcodebert-feature-integration/stargazers)
[![Open issues](https://img.shields.io/github/issues/jorge-martinez-gil/graphcodebert-feature-integration)](https://github.com/jorge-martinez-gil/graphcodebert-feature-integration/issues)
[![Last commit](https://img.shields.io/github/last-commit/jorge-martinez-gil/graphcodebert-feature-integration)](https://github.com/jorge-martinez-gil/graphcodebert-feature-integration/commits/main)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

> Keywords: GraphCodeBERT feature integration · code transformer feature fusion · clone detection benchmark · source code similarity · feature engineering for code models

---

## Why FeatFuse?

Pretrained code models such as GraphCodeBERT, CodeBERT, and UniXcoder learn strong token- and data-flow-based representations, but they can still miss signals that are cheap to compute explicitly and highly informative for code clone and similarity tasks.

FeatFuse is a benchmark and toolkit for testing that idea systematically.

It lets you evaluate whether handcrafted features such as lexical overlap, structural statistics, and execution-aware signals improve a pretrained encoder on clone detection tasks, while keeping the setup fully reproducible and easy to ablate.

This repo turns the research from the paper "Improving Source Code Similarity Detection Through GraphCodeBERT and Integration of Additional Features" into a reusable experimental platform.

---

## What FeatFuse includes

- A plug-in registry for features, fusion strategies, models, and datasets
- CPU-friendly classical baselines and neural fusion experiments
- Reproducible runs with manifest tracking and publication-ready reports
- Built-in metrics, bootstrap confidence intervals, and significance tests
- A CLI for running experiments, ablations, and feature importance analysis

### Feature categories

- Lexical features: token similarity, identifier overlap, edit-based measures
- Structural features: nesting depth, branch density, code-size statistics
- Execution-based features: behavioral signals that capture semantic equivalence

### Fusion strategies

- `concat`
- `gated`
- `attention`
- `residual`
- `film`

### Supported backends

- GraphCodeBERT
- CodeBERT
- UniXcoder
- Hugging Face encoders via the same registry pattern

---

## Quickstart

### Install

```bash
git clone https://github.com/jorge-martinez-gil/graphcodebert-feature-integration
cd graphcodebert-feature-integration
pip install -e ".[dev]"
```

### Run a smoke test

```bash
featfuse info
featfuse run -c configs/smoke.yaml
```

This creates a traceable run directory under `runs/` with metrics, tables, figures, and a report.

### Reproduce a classical baseline

```bash
featfuse run -c configs/classical_features_irplag.yaml --ablate
```

### Reproduce the neural paper setup

```bash
pip install "featfuse[neural]"
featfuse run -c configs/graphcodebert_irplag.yaml
```

> GPU support is only required for the neural fine-tuning experiments. The smoke and classical feature pipelines run on CPU.

---

## Repository structure

```text
.
├── configs/                  # YAML experiment configs
├── data/                     # datasets and generated artifacts
├── docs/                     # extension and reproducibility guides
├── legacy/                   # original paper scripts and notebook
├── notebooks/                # exploratory notebooks
├── src/featfuse/             # core library
│   ├── data/
│   ├── features/
│   ├── fusion/
│   ├── models/
│   ├── cli.py
│   ├── config.py
│   ├── experiment.py
│   ├── metrics.py
│   ├── stats.py
│   ├── train.py
│   ├── registry.py
│   ├── reproducibility.py
│   └── viz.py
├── tests/                    # unit and smoke tests
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── README.md
├── pyproject.toml
├── requirements.txt
├── CITATION.cff
├── CITATION.bib
└── CHANGELOG.md
```

---

## Research question FeatFuse answers

The key question is not only whether a pretrained code model is strong, but whether a specific engineered signal helps that model in a controlled and measurable way.

FeatFuse is designed to answer questions such as:

- Does a lexical feature improve clone detection beyond text-only embeddings?
- Does an execution-based signal help on semantically equivalent but syntactically different pairs?
- Which fusion strategy is best for this model-feature combination?
- Are the observed gains statistically significant?

---

## Results

### Published paper benchmark

On the IR-Plag benchmark, the fused GraphCodeBERT + feature approach reports the best result in the paper:

| Approach | Precision | Recall | F-Measure |
|---|---:|---:|---:|
| CodeBERT | 0.72 | 1.00 | 0.84 |
| Output Analysis | 0.88 | 0.93 | 0.90 |
| Boosting (XGBoost) | 0.88 | 0.99 | 0.93 |
| Bagging (Random Forest) | 0.95 | 0.97 | 0.96 |
| GraphCodeBERT | 0.98 | 0.95 | 0.96 |
| GraphCodeBERT + feature (this work) | 0.98 | 1.00 | 0.99 |

### Classical feature baseline

```bash
featfuse run -c configs/classical_features_irplag.yaml --ablate
```

This benchmark gives a CPU-friendly reference point without a transformer backend, and it is useful for fair ablation studies.

---

## Documentation and extension points

See the docs directory for deeper guidance:

- [docs/reproducibility.md](docs/reproducibility.md)
- [docs/benchmark.md](docs/benchmark.md)
- [docs/adding_a_feature.md](docs/adding_a_feature.md)
- [docs/adding_a_fusion.md](docs/adding_a_fusion.md)
- [docs/adding_a_model.md](docs/adding_a_model.md)
- [docs/roadmap.md](docs/roadmap.md)

---

## Contributing

New features, fusion strategies, models, and datasets are all designed as small plugins. Contributions are welcome.

- See [CONTRIBUTING.md](CONTRIBUTING.md)
- Open an issue for a bug, dataset, or feature request
- Propose a benchmark extension or new fusion architecture

---

## Citation

If FeatFuse or its benchmark results contribute to your research, please cite the paper:

```bibtex
@article{martinezgil2024graphcodebert,
  title   = {Improving Source Code Similarity Detection Through GraphCodeBERT and Integration of Additional Features},
  author  = {Martinez-Gil, Jorge},
  journal = {arXiv preprint arXiv:2408.08903},
  year    = {2024},
  url     = {https://arxiv.org/abs/2408.08903}
}
```

You can also generate the software citation with:

```bash
featfuse cite --software
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Project status

FeatFuse is intended as an open, extensible platform for feature-aware code representation learning. It is especially useful for benchmarking engineered signals against pretrained code models while keeping every experiment reproducible from a committed config and manifest.

If you want to compare handcrafted features and transformer-based encoders in a controlled setting, this repository is built for that workflow.
