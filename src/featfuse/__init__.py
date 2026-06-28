"""FeatFuse — An open benchmark for feature integration in code language models.

FeatFuse turns the research code accompanying *"Improving Source Code Similarity
Detection with GraphCodeBERT and Additional Feature Integration"*
(Martinez-Gil, 2024, arXiv:2408.08903) into reusable research infrastructure for
studying how **engineered features** can complement **pretrained code language
models** (GraphCodeBERT, CodeBERT, UniXcoder, ...).

The package exposes four extension points, each backed by a plugin registry:

* ``featfuse.features``  — engineered feature extractors (lexical, structural, execution-based, ...)
* ``featfuse.fusion``    — neural fusion strategies (concat, gated, attention, residual, ...)
* ``featfuse.models``    — code encoders (HuggingFace transformers + a dependency-free toy encoder)
* ``featfuse.data``      — datasets / loaders (IR-Plag today; BigCloneBench / POJ-104 next)

A researcher can register a new feature or fusion module in <20 lines and run the
full, reproducible benchmark with a single command (``featfuse run -c config.yaml``).

See ``docs/`` and the project README for tutorials and reproduction instructions.
"""

from __future__ import annotations

__version__ = "0.2.0"
__author__ = "Jorge Martinez-Gil and contributors"
__license__ = "MIT"
__paper__ = "arXiv:2408.08903"

from .registry import FEATURES, FUSIONS, MODELS, DATASETS, Registry
from .types import CodePair

__all__ = [
    "__version__",
    "FEATURES",
    "FUSIONS",
    "MODELS",
    "DATASETS",
    "Registry",
    "CodePair",
]
