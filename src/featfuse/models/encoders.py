"""Code encoders.

Two implementations are provided:

* :class:`HFEncoder` wraps any HuggingFace encoder checkpoint
  (``microsoft/graphcodebert-base``, ``microsoft/codebert-base``,
  ``microsoft/unixcoder-base``, ...). It is the encoder used to reproduce the
  paper. Requires the optional ``featfuse[neural]`` dependencies.

* :class:`ToyEncoder` is a deterministic, dependency-free hashing encoder. It needs
  no network access, no model download and no torch, which makes the full benchmark
  pipeline runnable and testable on any machine (CI, laptops, classrooms).
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Sequence

import numpy as np

from ..registry import MODELS

_WORD = re.compile(r"[A-Za-z_]\w*|\d+|\S")


@MODELS.register("toy", kind="encoder", needs_torch=False)
class ToyEncoder:
    """Deterministic bag-of-hashed-tokens encoder (no dependencies, no download).

    Each token is hashed into a fixed-size vector; the document embedding is the
    L2-normalised, sub-linear-weighted sum of its token vectors. This is *not* a
    pretrained model — it exists so the pipeline, fusion plumbing and metrics can be
    exercised end-to-end without GPUs or network access. Swap in :class:`HFEncoder`
    for real experiments.
    """

    def __init__(self, embed_dim: int = 64, seed: int = 42):
        self.embed_dim = int(embed_dim)
        self.seed = seed

    def _hash(self, token: str) -> int:
        h = hashlib.md5(f"{self.seed}:{token}".encode()).hexdigest()
        return int(h, 16)

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.embed_dim, dtype=np.float64)
        toks = _WORD.findall(text or "")
        for t in toks:
            hv = self._hash(t)
            idx = hv % self.embed_dim
            sign = 1.0 if (hv >> 8) % 2 else -1.0
            vec[idx] += sign
        # sub-linear scaling + L2 normalise for stable magnitudes
        vec = np.sign(vec) * np.log1p(np.abs(vec))
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._embed_one(t) for t in texts]) if texts else np.zeros((0, self.embed_dim))

    def encode_pairs(self, code1: Sequence[str], code2: Sequence[str]) -> np.ndarray:
        """Encode a batch of pairs into a single interaction vector.

        Concatenates ``[e1, e2, |e1-e2|, e1*e2]`` — a standard sentence-pair
        interaction representation — giving a ``4 * embed_dim`` feature block.
        """
        e1 = self.encode(code1)
        e2 = self.encode(code2)
        return np.hstack([e1, e2, np.abs(e1 - e2), e1 * e2])


class HFEncoder:
    """HuggingFace transformer encoder (GraphCodeBERT / CodeBERT / UniXcoder / ...).

    Lazily imports ``torch`` and ``transformers`` so the rest of FeatFuse works
    without them installed.
    """

    def __init__(self, name: str = "microsoft/graphcodebert-base"):
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover - neural extra
            raise RuntimeError(
                "HFEncoder requires the neural extra:  pip install 'featfuse[neural]'"
            ) from exc
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)
        self.embed_dim = self.model.config.hidden_size

    def pooled(self, input_ids, attention_mask=None):
        """Return a ``(B, hidden)`` pooled representation (pooler or mean-pool)."""
        import torch

        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if getattr(out, "pooler_output", None) is not None:
            return out.pooler_output
        last = out.last_hidden_state
        if attention_mask is None:
            return last.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).type_as(last)
        return (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


@MODELS.register("hf", kind="encoder", needs_torch=True)
def build_hf(name: str = "microsoft/graphcodebert-base", **_: object) -> HFEncoder:
    return HFEncoder(name)
