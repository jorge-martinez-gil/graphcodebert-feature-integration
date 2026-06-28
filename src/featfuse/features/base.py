"""Base classes for engineered feature plugins.

A :class:`PairFeature` maps a pair of code fragments to one or more scalar values.
Features are deliberately *pairwise and pure*: they depend only on the two
fragments (and optional precomputed metadata), so they are deterministic and can be
recomputed on any dataset split without external state. A :class:`FeatureSet`
composes several features into a single fixed-width vector with named columns —
the engineered-feature half of every fusion experiment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..types import CodePair


class PairFeature(ABC):
    """Abstract base for a feature extractor over a pair of code fragments."""

    #: short, unique, registry-friendly identifier
    name: str = "feature"
    #: human-readable grouping used in reports (lexical / structural / execution / ...)
    family: str = "misc"
    #: number of scalar columns produced
    dim: int = 1

    @abstractmethod
    def extract(self, code1: str, code2: str, meta: Optional[Dict[str, Any]] = None) -> List[float]:
        """Return ``self.dim`` scalar values describing the pair."""

    def columns(self) -> List[str]:
        """Names for the produced columns (length ``self.dim``)."""
        if self.dim == 1:
            return [self.name]
        return [f"{self.name}[{i}]" for i in range(self.dim)]

    def __call__(self, pair: CodePair) -> List[float]:
        return self.extract(pair.code1, pair.code2, pair.meta)


class FeatureSet:
    """An ordered collection of features producing a single named vector."""

    def __init__(self, features: Sequence[PairFeature]):
        self.features: List[PairFeature] = list(features)

    @property
    def columns(self) -> List[str]:
        cols: List[str] = []
        for f in self.features:
            cols.extend(f.columns())
        return cols

    @property
    def dim(self) -> int:
        return sum(f.dim for f in self.features)

    def vector(self, pair: CodePair) -> np.ndarray:
        vals: List[float] = []
        for f in self.features:
            out = f.extract(pair.code1, pair.code2, pair.meta)
            if len(out) != f.dim:
                raise ValueError(f"Feature '{f.name}' returned {len(out)} values, expected {f.dim}")
            vals.extend(float(v) for v in out)
        return np.asarray(vals, dtype=np.float64)

    def transform(self, pairs: Sequence[CodePair]) -> np.ndarray:
        """Return an ``(n_pairs, dim)`` matrix of features."""
        if not pairs:
            return np.zeros((0, self.dim), dtype=np.float64)
        return np.vstack([self.vector(p) for p in pairs])

    @classmethod
    def from_names(cls, names: Sequence[str]) -> "FeatureSet":
        from ..registry import FEATURES  # local import to avoid cycles

        return cls([FEATURES.create(n) for n in names])
