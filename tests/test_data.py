import json
import os

from featfuse.data import seeded_split
from featfuse.registry import DATASETS
from featfuse.types import CodePair

import featfuse.data  # noqa: F401

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def test_seeded_split_is_deterministic_and_partitions():
    pairs = [CodePair(code1=str(i), code2=str(i), label=i % 2) for i in range(100)]
    a = seeded_split(pairs, [0.7, 0.15, 0.15], seed=1)
    b = seeded_split(pairs, [0.7, 0.15, 0.15], seed=1)
    assert [p.code1 for p in a["train"]] == [p.code1 for p in b["train"]]
    total = sum(len(a[s]) for s in ("train", "validation", "test"))
    assert total == 100


def test_irplag_loads_and_enriches_fixed_splits():
    if not os.path.exists(os.path.join(DATA, "training.json")):
        import pytest

        pytest.skip("dataset files not present")
    ds = DATASETS.create("irplag", root=DATA, train="training.json",
                          validation="validation.json", test="test.json",
                          single_file="data2.json")
    assert set(ds.available()) == {"train", "validation", "test"}
    # every fixed-split pair should be enriched with the precomputed exec feature
    assert all("output" in p.meta for p in ds["train"])
