# Contributing to FeatFuse

Thanks for helping build the open benchmark for **feature integration in code
language models**. The fastest way to contribute is to add a *plugin* — a new
engineered feature, fusion strategy, encoder, or dataset. Every plugin is a small
class/function registered with a one-line decorator, so the benchmark picks it up
automatically.

## Quick start

```bash
git clone https://github.com/jorge-martinez-gil/graphcodebert-feature-integration
cd graphcodebert-feature-integration
pip install -e ".[dev]"
pytest -q                       # all tests should pass
featfuse run -c configs/smoke.yaml
```

## Adding a plugin (the common case)

| You want to add | Subclass / register | Guide |
|---|---|---|
| an engineered feature | `PairFeature` → `@FEATURES.register` | [docs/adding_a_feature.md](docs/adding_a_feature.md) |
| a fusion strategy | `nn.Module` → `@FUSIONS.register` | [docs/adding_a_fusion.md](docs/adding_a_fusion.md) |
| a code encoder | encoder class → `@MODELS.register` | [docs/adding_a_model.md](docs/adding_a_model.md) |
| a dataset / task | loader → `@DATASETS.register` | [docs/adding_a_model.md](docs/adding_a_model.md) |

## Ground rules (scientific integrity)

1. **No fabricated results.** Any number in code, docs, or a PR description must be
   reproducible from a committed config via `featfuse run`.
2. **Tests required.** New plugins need at least one test; `pytest -q` must pass.
3. **Docs required.** Update the relevant `docs/` page and, if user-facing, the README.
4. **Keep the core light.** The classical pipeline must keep running without
   `torch`/`transformers` (those live behind the `neural` extra).

## Pull requests

- Branch from `main`, keep PRs focused, and fill in the PR template checklist.
- CI runs the test suite on Python 3.9–3.12 plus the CPU smoke benchmark.
- By contributing you agree your work is released under the project's MIT License.
