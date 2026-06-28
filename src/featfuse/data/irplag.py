"""IR-Plag dataset loader.

IR-Plag is the academic source-code plagiarism benchmark used by the original
paper. This loader supports two reproducible modes:

1. **Explicit splits** — load the published ``training/validation/test.json`` files.
2. **Seeded split** — deterministically split ``data2.json`` by fractions.

In both modes, if the precomputed execution-similarity values live only in
``data2.json`` (the published split files omit them), the loader *enriches* every
pair's metadata by matching on ``(code1, code2)`` — so the paper's execution feature
remains available on the fixed splits without re-running a JVM.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from ..registry import DATASETS
from ..types import CodePair
from .base import Dataset, load_records, records_to_pairs, seeded_split


def _enrichment_map(root: str, single_file: Optional[str]) -> Dict[tuple, dict]:
    if not single_file:
        return {}
    path = os.path.join(root, single_file)
    if not os.path.exists(path):
        return {}
    out: Dict[tuple, dict] = {}
    for r in load_records(path):
        key = (r.get("code1"), r.get("code2"))
        meta = {k: v for k, v in r.items() if k not in {"code1", "code2", "score", "label", "index"}}
        out[key] = meta
    return out


def _enrich(pairs: List[CodePair], emap: Dict[tuple, dict]) -> List[CodePair]:
    if not emap:
        return pairs
    for p in pairs:
        extra = emap.get((p.code1, p.code2))
        if extra:
            for k, v in extra.items():
                p.meta.setdefault(k, v)
    return pairs


@DATASETS.register("irplag", family="clone_detection", language="java")
def load_irplag(
    root: str = "data",
    train: Optional[str] = None,
    validation: Optional[str] = None,
    test: Optional[str] = None,
    single_file: Optional[str] = "data2.json",
    split_fractions=(0.7, 0.15, 0.15),
    seed: int = 42,
    **_: object,
) -> Dataset:
    emap = _enrichment_map(root, single_file)

    if train and validation and test:
        splits = {
            "train": _enrich(records_to_pairs(load_records(os.path.join(root, train))), emap),
            "validation": _enrich(records_to_pairs(load_records(os.path.join(root, validation))), emap),
            "test": _enrich(records_to_pairs(load_records(os.path.join(root, test))), emap),
        }
    else:
        if not single_file:
            raise ValueError("Provide explicit splits or a single_file to split.")
        pairs = records_to_pairs(load_records(os.path.join(root, single_file)))
        splits = seeded_split(pairs, list(split_fractions), seed=seed)

    ds = Dataset(splits)
    ds.name = "irplag"
    return ds
