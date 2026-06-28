"""Dataset abstractions.

A :class:`Dataset` exposes named splits (``train`` / ``validation`` / ``test``),
each a list of :class:`~featfuse.types.CodePair`. New benchmarks (BigCloneBench,
POJ-104, PoolC, ...) implement :class:`Dataset` and register themselves in
:data:`featfuse.registry.DATASETS`.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Sequence

from ..types import CodePair


class Dataset:
    """A code-pair dataset with named splits."""

    name: str = "dataset"

    def __init__(self, splits: Dict[str, List[CodePair]]):
        self.splits = splits

    def __getitem__(self, split: str) -> List[CodePair]:
        return self.splits[split]

    def available(self) -> List[str]:
        return list(self.splits)

    def summary(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for name, pairs in self.splits.items():
            pos = sum(1 for p in pairs if p.label == 1)
            out[name] = {"n": len(pairs), "positive": pos, "negative": len(pairs) - pos}
        return out


def load_records(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def records_to_pairs(records: Sequence[dict]) -> List[CodePair]:
    return [CodePair.from_record(r) for r in records]


def seeded_split(
    pairs: Sequence[CodePair], fractions: Sequence[float], seed: int = 42
) -> Dict[str, List[CodePair]]:
    """Deterministically split into train/validation/test by ``fractions``."""
    if not abs(sum(fractions) - 1.0) < 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {fractions}")
    items = list(pairs)
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(fractions[0] * n)
    n_val = int(fractions[1] * n)
    return {
        "train": items[:n_train],
        "validation": items[n_train : n_train + n_val],
        "test": items[n_train + n_val :],
    }
