"""The generalized fusion classification head.

:class:`FusionClassifier` is the direct, generalized successor of the paper's
``RobertaForSequenceClassificationWithOutput``. The original concatenated a single
scalar feature with GraphCodeBERT's pooled output; this version accepts an arbitrary
**feature vector** and an arbitrary **fusion strategy** (concat / gated / attention
/ residual / film), making the architecture a free variable that researchers can
swap from a config file.

PyTorch is imported lazily; constructing a :class:`FusionClassifier` requires the
``featfuse[neural]`` extra.
"""

from __future__ import annotations

from ..registry import FUSIONS

try:
    import torch
    import torch.nn as nn
    from transformers.modeling_outputs import SequenceClassifierOutput

    _TORCH = True
except Exception:  # pragma: no cover - neural extra
    _TORCH = False


if _TORCH:

    class FusionClassifier(nn.Module):
        def __init__(
            self,
            encoder,
            feature_dim: int,
            fusion: str = "concat",
            num_labels: int = 2,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.encoder = encoder
            self.num_labels = num_labels
            embed_dim = encoder.embed_dim
            self.fusion_name = fusion
            self.fusion = FUSIONS.create(fusion, embed_dim, feature_dim)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(self.fusion.out_dim, num_labels)

        def forward(self, input_ids, attention_mask=None, features=None, labels=None):
            h = self.encoder.pooled(input_ids, attention_mask)
            fused = self.fusion(h, features)
            logits = self.classifier(self.dropout(fused))
            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))
            return SequenceClassifierOutput(loss=loss, logits=logits)


def build_fusion_classifier(encoder, feature_dim, fusion="concat", num_labels=2, dropout=0.1):
    """Construct a :class:`FusionClassifier` (requires torch)."""
    if not _TORCH:
        raise RuntimeError(
            "FusionClassifier requires the neural extra:  pip install 'featfuse[neural]'"
        )
    return FusionClassifier(encoder, feature_dim, fusion, num_labels, dropout)
