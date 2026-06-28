<div align="center">

# FeatFuse — Feature Integration for Code Language Models

### An open, reproducible benchmark for augmenting GraphCodeBERT (and CodeBERT, UniXcoder, …) with engineered features

*Does adding hand-crafted lexical, structural, and execution-based features to a pretrained code transformer actually help? FeatFuse lets you measure it — reproducibly, with one command.*

[![CI](https://github.com/jorge-martinez-gil/graphcodebert-feature-integration/actions/workflows/ci.yml/badge.svg)](https://github.com/jorge-martinez-gil/graphcodebert-feature-integration/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/arXiv-2408.08903-b31b1b.svg)](https://arxiv.org/abs/2408.08903)

</div>

> **Keywords:** GraphCodeBERT feature integration · code transformer feature fusion · GraphCodeBERT benchmark · code similarity transformer · code representation learning · feature engineering for code models · code embeddings · code intelligence benchmark · software engineering transformers.

---

## What problem does this solve?

Pretrained code language models such as **GraphCodeBERT** produce powerful general-purpose representations of source code, but for a specific task (e.g. **code clone / similarity detection**) they are used as black boxes: you cannot easily ask *"would my hand-crafted feature improve this model, and by how much — significantly?"*

**FeatFuse** turns the research code from the paper *["Improving Source Code Similarity Detection Through GraphCodeBERT and Integration of Additional Features"](https://arxiv.org/abs/2408.08903)* (Martinez-Gil, 2024) into reusable **research infrastructure**: a plugin-based platform where an **engineered feature**, a **fusion architecture**, a **code model**, and a **dataset** are all swappable, and where every comparison is benchmarked with the full metric suite, statistical significance tests, ablations, and auto-generated tables and figures.

It is designed so that future work on GraphCodeBERT, CodeBERT, feature fusion, and code intelligence can evaluate against a common, reproducible baseline — and cite the associated paper.

## Why integrate engineered features? Why are pretrained code models insufficient on their own?

Transformer code encoders are trained on token and data-flow context. They can miss signals that are cheap to compute explicitly and orthogonal to what attention captures, for example:

- **Behavioural / execution signal** — two fragments that produce the *same output* are likely equivalent even if their tokens differ. (This is the paper's original additional feature.)
- **Structural signal** — nesting depth, a cyclomatic-complexity proxy, keyword usage.
- **Lexical signal** — token / character-n-gram overlap, edit similarity, identifier overlap.

These features are especially helpful on **semantically equivalent but syntactically dissimilar** pairs, which are exactly the hard cases for a purely text-driven encoder. FeatFuse makes the contribution of each feature *measurable and interpretable* rather than assumed.

## How does feature integration improve performance?

Engineered features are fused with the transformer's pooled embedding before classification. FeatFuse ships **five fusion strategies** so the architecture is a free variable, not a fixed choice:

| Strategy | Idea | Output width |
|---|---|---|
| `concat` | project features → `E`, concatenate (the paper's design) | `2E` |
| `gated` | a learned gate mixes embedding and features | `E` |
| `attention` | soft attention over {embedding, features} | `E` |
| `residual` | inject features additively via an MLP residual | `E` |
| `film` | feature-wise linear modulation (scale & shift) of the embedding | `E` |

Whether a given feature + fusion combination helps is an **empirical question** — so FeatFuse answers it with ablations, feature-importance analysis, and significance testing instead of claims.

---

## Quickstart (CPU, no GPU, no downloads)

```bash
git clone https://github.com/jorge-martinez-gil/graphcodebert-feature-integration
cd graphcodebert-feature-integration
pip install -e ".[dev]"

featfuse info                         # list registered features / fusion / models / datasets
featfuse run -c configs/smoke.yaml    # full pipeline end-to-end in seconds
```

A run writes a fully traceable directory:

```
runs/smoke/
  manifest.json   results.json   results_table.md   results_table.tex   REPORT.md   figures/
```

---

## Results

### Reported in the paper (neural backend: GraphCodeBERT + feature, fused in the head)

These are the results from [arXiv:2408.08903](https://arxiv.org/abs/2408.08903) on the **IR-Plag** benchmark. Reproduce them with `configs/graphcodebert_irplag.yaml` (requires `featfuse[neural]` and, realistically, a GPU).

| Approach | Precision | Recall | F-Measure |
|---|:---:|:---:|:---:|
| CodeBERT | 0.72 | 1.00 | 0.84 |
| Output Analysis | 0.88 | 0.93 | 0.90 |
| Boosting (XGBoost) | 0.88 | 0.99 | 0.93 |
| Bagging (Random Forest) | 0.95 | 0.97 | 0.96 |
| GraphCodeBERT | 0.98 | 0.95 | 0.96 |
| **GraphCodeBERT + feature (this work)** | **0.98** | **1.00** | **0.99** |

### Reproducible here (classical backend: engineered features → a classifier, GPU-free)

To give an honest, runs-anywhere reference point, FeatFuse also benchmarks the **engineered features on their own** (no transformer) with `configs/classical_features_irplag.yaml`. These numbers are produced by the command below and live in `runs/` — they characterise how far the features go alone and are **not** the neural result above.

```bash
featfuse run -c configs/classical_features_irplag.yaml --ablate
```

| Method (IR-Plag test split) | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | ECE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gradient Boosting · 11 engineered features | 0.913 | 0.906 | 0.980 | 0.941 | 0.784 | 0.975 | 0.056 |

> Every number in this repository is reproducible from a committed config and its `manifest.json`. We never fabricate or hand-edit benchmark results.

---

## How do I … ?

### …reproduce the published experiments
`featfuse run -c configs/graphcodebert_irplag.yaml` (neural, needs `pip install "featfuse[neural]"` + GPU). The GPU-free baseline is `configs/classical_features_irplag.yaml`. See [docs/reproducibility.md](docs/reproducibility.md).

### …evaluate a new feature
Subclass `PairFeature`, add `@FEATURES.register("my_feature")`, list it in a config, and run. ~15 lines — see [docs/adding_a_feature.md](docs/adding_a_feature.md). Then inspect its contribution:

```bash
featfuse importance -c configs/classical_features_irplag.yaml   # permutation + native importance
featfuse ablate     -c configs/classical_features_irplag.yaml   # leave-one-out + single-feature
```

### …implement a new fusion architecture
Write an `nn.Module` with an `out_dim` attribute and `@FUSIONS.register("my_fusion")`, then set `model.fusion: my_fusion`. See [docs/adding_a_fusion.md](docs/adding_a_fusion.md).

### …benchmark a new code model
Register a HuggingFace encoder (`@MODELS.register`) and set `model.encoder`. The wrapper already supports GraphCodeBERT / CodeBERT / UniXcoder; CodeT5, StarCoder, Qwen-Coder and DeepSeek-Coder follow the same pattern. See [docs/adding_a_model.md](docs/adding_a_model.md).

### …contribute
New features, fusion strategies, models and datasets are all one-line plugins. See [CONTRIBUTING.md](CONTRIBUTING.md) and the [roadmap](docs/roadmap.md).

---

## What's inside

```
src/featfuse/
  registry.py        plugin registry (features / fusion / models / datasets)
  features/          engineered features: lexical, structural, execution-based
  fusion/            neural fusion strategies (concat, gated, attention, residual, film)
  models/            encoders (HuggingFace wrapper + dependency-free ToyEncoder) + fusion head
  data/              dataset loaders (IR-Plag; BigCloneBench / POJ-104 planned)
  metrics.py         accuracy, precision, recall, F1, MCC, ROC/PR-AUC, Brier, ECE
  stats.py           bootstrap CIs, McNemar, paired bootstrap difference tests
  experiment.py      config-driven runner + ablation + feature importance
  report.py / viz.py auto-generated Markdown/LaTeX tables and publication figures
  cli.py             the `featfuse` command-line interface
configs/             smoke (CPU) · classical baseline · neural reproduction
docs/                how-to guides + reproducibility + roadmap
tests/               unit + end-to-end CPU smoke tests (run in CI)
legacy/              the original paper script and notebook, kept for provenance
```

## Evaluation & metrics

Threshold metrics (accuracy, balanced accuracy, precision, recall, F1, MCC), ranking metrics (ROC-AUC, PR-AUC), calibration (Brier, ECE), and cost (training time, inference latency) — all reported together, with bootstrap confidence intervals and McNemar / paired-bootstrap significance tests. Details in [docs/benchmark.md](docs/benchmark.md).

---

## Citation

If you use FeatFuse or this benchmark, please cite the paper:

```bibtex
@article{martinezgil2024graphcodebert,
  title   = {Improving Source Code Similarity Detection Through GraphCodeBERT and Integration of Additional Features},
  author  = {Martinez-Gil, Jorge},
  journal = {arXiv preprint arXiv:2408.08903},
  year    = {2024},
  url     = {https://arxiv.org/abs/2408.08903}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is included (GitHub shows a "Cite this repository" button). A related follow-up on interpretability is *Augmenting the Interpretability of GraphCodeBERT for Code Similarity Tasks* ([arXiv:2410.05275](https://arxiv.org/abs/2410.05275)).

## License

Released under the **MIT License** — see [LICENSE](LICENSE).
