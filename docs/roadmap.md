# Roadmap

FeatFuse aims to be the default platform for evaluating feature-fusion architectures
on top of code language models. Contributions toward any item below are welcome
(see [CONTRIBUTING.md](../CONTRIBUTING.md)).

## Models (neural backend)
- [x] GraphCodeBERT, CodeBERT, UniXcoder (via the HuggingFace wrapper)
- [ ] CodeT5, PLBART (encoder-decoder pooling)
- [ ] StarCoder, Qwen-Coder, DeepSeek-Coder, Llama-based code models (decoder pooling)

## Datasets / tasks
- [x] IR-Plag (clone / similarity detection)
- [ ] BigCloneBench, POJ-104, PoolC, CodeJam (automatic downloaders)
- [ ] code search / retrieval, defect & vulnerability detection, summarization

## Features
- [x] lexical, structural/metric, execution-based
- [ ] AST / control-flow / data-flow graph features, readability & static-analysis features
- [ ] retrieval-based features

## Fusion & efficiency
- [x] concat, gated, attention, residual, FiLM
- [ ] adapter layers, LoRA-based feature injection
- [ ] uncertainty estimation & post-hoc calibration

## Platform
- [ ] leaderboard generation from `runs/` directories
- [ ] interactive (HTML) visualizations
- [ ] carbon-footprint proxy for the cost axis
