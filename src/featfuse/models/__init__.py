"""Code encoders and the fusion classification head.

Importing this package registers the encoders (``toy``, ``hf``) in
:data:`featfuse.registry.MODELS`.
"""

from __future__ import annotations

from .encoders import HFEncoder, ToyEncoder  # noqa: F401  (registers encoders)
from .head import build_fusion_classifier  # noqa: F401

__all__ = ["HFEncoder", "ToyEncoder", "build_fusion_classifier"]
