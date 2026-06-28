"""Fusion strategy tests — run only when the optional torch extra is installed."""

import pytest

torch = pytest.importorskip("torch")

from featfuse.registry import FUSIONS
import featfuse.fusion  # noqa: F401


@pytest.mark.parametrize("name", ["concat", "gated", "attention", "residual", "film"])
def test_fusion_output_shapes(name):
    B, E, F = 4, 16, 5
    fusion = FUSIONS.create(name, E, F)
    h = torch.randn(B, E)
    f = torch.randn(B, F)
    out = fusion(h, f)
    assert out.shape[0] == B
    assert out.shape[1] == fusion.out_dim
    expected = 2 * E if name == "concat" else E
    assert fusion.out_dim == expected
