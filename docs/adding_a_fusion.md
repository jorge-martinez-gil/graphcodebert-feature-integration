# Adding a fusion strategy

A fusion strategy is an `nn.Module` that combines a transformer pooled embedding
`h` of shape `(B, E)` with an engineered-feature vector `f` of shape `(B, F)` and
returns a fused representation `(B, out_dim)`.

```python
import torch, torch.nn as nn
from featfuse.registry import FUSIONS

class BilinearFusion(nn.Module):
    def __init__(self, embed_dim, feature_dim, **kw):
        super().__init__()
        self.bilinear = nn.Bilinear(embed_dim, feature_dim, embed_dim)
        self.out_dim = embed_dim          # REQUIRED: declare the output width
    def forward(self, h, f):
        return torch.relu(self.bilinear(h, f))

@FUSIONS.register("bilinear", family="fusion", description="Bilinear interaction of embedding and features.")
def build_bilinear(embed_dim, feature_dim, **kw):
    return BilinearFusion(embed_dim, feature_dim, **kw)
```

Then select it in a neural config:

```yaml
model:
  backend: neural
  encoder: microsoft/graphcodebert-base
  fusion: bilinear
```

Every fusion module **must** expose an `out_dim` attribute — the classification head
reads it to size the final linear layer. Built-in strategies (`concat`, `gated`,
`attention`, `residual`, `film`) are good references in
`src/featfuse/fusion/strategies.py`.
