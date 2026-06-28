# Adding an engineered feature

A feature maps a pair of code fragments to one or more scalar values. Implement
`PairFeature` and register it — that's it.

```python
from featfuse.features.base import PairFeature
from featfuse.registry import FEATURES

@FEATURES.register("comment_density_ratio", family="structural")
class CommentDensityRatio(PairFeature):
    name = "comment_density_ratio"
    family = "structural"
    dim = 1  # number of scalar columns produced

    def extract(self, code1, code2, meta=None):
        def density(code):
            lines = code.splitlines() or [""]
            comments = sum(1 for ln in lines if ln.strip().startswith("//"))
            return comments / len(lines)
        from featfuse.features._textutil import ratio_sim
        return [ratio_sim(density(code1), density(code2))]
```

Use it from any config:

```yaml
features:
  - comment_density_ratio
  - token_jaccard
```

Guidelines:
- Return exactly `self.dim` values; keep them in a sensible range (similarities in
  `[0, 1]` compose best).
- Features must be **deterministic** and depend only on the two fragments (and
  optional precomputed `meta`).
- Add a test (see `tests/test_features.py`) and you're done — `featfuse list features`
  and every config can now use it.
