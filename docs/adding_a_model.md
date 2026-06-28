# Adding a code model or dataset

## A new encoder

Wrap any HuggingFace checkpoint by reusing `HFEncoder`, or register your own. An
encoder used by the **neural** backend needs `embed_dim` and a
`pooled(input_ids, attention_mask) -> (B, hidden)` method; one used by the
**classical** backend needs `encode_pairs(code1, code2) -> (N, D)`.

```python
from featfuse.registry import MODELS
from featfuse.models.encoders import HFEncoder

@MODELS.register("codebert", kind="encoder", needs_torch=True)
def build_codebert(**kw):
    return HFEncoder("microsoft/codebert-base")
```

Then: `model: { backend: neural, encoder: microsoft/codebert-base }`. The same
pattern covers UniXcoder, CodeT5, PLBART, StarCoder, Qwen-Coder and DeepSeek-Coder
checkpoints.

## A new dataset / task

Register a loader returning a `Dataset` with named splits of `CodePair`s:

```python
from featfuse.registry import DATASETS
from featfuse.data.base import Dataset
from featfuse.types import CodePair

@DATASETS.register("bigclonebench", family="clone_detection", language="java")
def load_bcb(root="data", **kw):
    splits = {"train": [...], "validation": [...], "test": [...]}  # lists of CodePair
    ds = Dataset(splits); ds.name = "bigclonebench"; return ds
```

A loader should be **reproducible**: download deterministically (cache under `root`)
and document the source. See `src/featfuse/data/irplag.py` for a worked example,
including how to *enrich* split records with precomputed features.
