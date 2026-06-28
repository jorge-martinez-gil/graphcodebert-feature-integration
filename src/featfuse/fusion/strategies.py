"""Neural fusion strategies for combining a code embedding with engineered features.

Every strategy is an ``nn.Module`` that maps a transformer pooled embedding
``h`` of shape ``(B, E)`` and an engineered-feature vector ``f`` of shape
``(B, F)`` to a fused representation of shape ``(B, out_dim)``. A downstream linear
head turns the fused representation into class logits.

PyTorch is imported lazily, so importing this module (and listing the available
strategies) works even in environments without torch installed. Construction of a
strategy requires the optional ``featfuse[neural]`` dependencies.

Implemented strategies
----------------------
``concat``    : project features to ``E`` and concatenate  (the paper's design; ``out_dim = 2E``)
``gated``     : learned per-dimension gate mixes ``h`` and projected features  (``out_dim = E``)
``attention`` : 2-way soft attention over ``{h, projected features}``          (``out_dim = E``)
``residual``  : inject features additively via an MLP residual branch          (``out_dim = E``)
``film``      : feature-wise linear modulation (FiLM) of ``h`` by features     (``out_dim = E``)
"""

from __future__ import annotations

from ..registry import FUSIONS

try:  # PyTorch is an optional dependency (featfuse[neural]).
    import torch
    import torch.nn as nn

    _TORCH = True
except Exception:  # pragma: no cover - exercised only when torch is absent
    _TORCH = False


def _require_torch() -> None:
    if not _TORCH:
        raise RuntimeError(
            "PyTorch is required for neural fusion strategies. "
            "Install the neural extra:  pip install 'featfuse[neural]'"
        )


if _TORCH:

    class _FusionBase(nn.Module):
        out_dim: int

    class ConcatFusion(_FusionBase):
        def __init__(self, embed_dim: int, feature_dim: int, **_: object):
            super().__init__()
            self.proj = nn.Linear(feature_dim, embed_dim)
            self.act = nn.GELU()
            self.out_dim = 2 * embed_dim

        def forward(self, h, f):
            return torch.cat([h, self.act(self.proj(f))], dim=-1)

    class GatedFusion(_FusionBase):
        def __init__(self, embed_dim: int, feature_dim: int, **_: object):
            super().__init__()
            self.proj = nn.Linear(feature_dim, embed_dim)
            self.gate = nn.Linear(2 * embed_dim, embed_dim)
            self.out_dim = embed_dim

        def forward(self, h, f):
            pf = self.proj(f)
            g = torch.sigmoid(self.gate(torch.cat([h, pf], dim=-1)))
            return g * h + (1.0 - g) * pf

    class AttentionFusion(_FusionBase):
        def __init__(self, embed_dim: int, feature_dim: int, **_: object):
            super().__init__()
            self.proj = nn.Linear(feature_dim, embed_dim)
            self.score = nn.Linear(embed_dim, 1)
            self.out_dim = embed_dim

        def forward(self, h, f):
            pf = self.proj(f)
            stack = torch.stack([h, pf], dim=1)            # (B, 2, E)
            w = torch.softmax(self.score(stack), dim=1)     # (B, 2, 1)
            return (w * stack).sum(dim=1)                    # (B, E)

    class ResidualFusion(_FusionBase):
        def __init__(self, embed_dim: int, feature_dim: int, hidden: int = None, **_: object):
            super().__init__()
            hidden = hidden or embed_dim
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, hidden), nn.GELU(), nn.Linear(hidden, embed_dim)
            )
            self.norm = nn.LayerNorm(embed_dim)
            self.out_dim = embed_dim

        def forward(self, h, f):
            return self.norm(h + self.mlp(f))

    class FiLMFusion(_FusionBase):
        """Feature-wise Linear Modulation: features predict a scale & shift for ``h``."""

        def __init__(self, embed_dim: int, feature_dim: int, **_: object):
            super().__init__()
            self.to_scale = nn.Linear(feature_dim, embed_dim)
            self.to_shift = nn.Linear(feature_dim, embed_dim)
            self.out_dim = embed_dim

        def forward(self, h, f):
            return (1.0 + self.to_scale(f)) * h + self.to_shift(f)


@FUSIONS.register("concat", family="fusion", out="2E", description="Project features to E and concatenate (paper design).")
def build_concat(embed_dim, feature_dim, **kw):
    _require_torch()
    return ConcatFusion(embed_dim, feature_dim, **kw)


@FUSIONS.register("gated", family="fusion", out="E", description="Learned gate mixes embedding and projected features.")
def build_gated(embed_dim, feature_dim, **kw):
    _require_torch()
    return GatedFusion(embed_dim, feature_dim, **kw)


@FUSIONS.register("attention", family="fusion", out="E", description="2-way soft attention over embedding and features.")
def build_attention(embed_dim, feature_dim, **kw):
    _require_torch()
    return AttentionFusion(embed_dim, feature_dim, **kw)


@FUSIONS.register("residual", family="fusion", out="E", description="Additive MLP residual injection of features.")
def build_residual(embed_dim, feature_dim, **kw):
    _require_torch()
    return ResidualFusion(embed_dim, feature_dim, **kw)


@FUSIONS.register("film", family="fusion", out="E", description="Feature-wise linear modulation (scale & shift).")
def build_film(embed_dim, feature_dim, **kw):
    _require_torch()
    return FiLMFusion(embed_dim, feature_dim, **kw)
