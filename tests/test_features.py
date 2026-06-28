import numpy as np

from featfuse.features import FeatureSet
from featfuse.registry import FEATURES
from featfuse.types import CodePair

import featfuse.features  # noqa: F401  (registration)


def test_identical_code_is_maximally_similar():
    code = "public int f(int x){ return x+1; }"
    p = CodePair(code1=code, code2=code, label=1)
    for name in FEATURES.names():
        if name == "exec_output_similarity":
            continue  # depends on precomputed/JVM
        feat = FEATURES.create(name)
        val = feat.extract(p.code1, p.code2)[0]
        assert 0.0 <= val <= 1.0
        assert val > 0.99, f"{name} should be ~1 for identical code, got {val}"


def test_feature_values_in_range_for_different_code():
    p = CodePair(code1="int a = 1;", code2="while(true){ doSomething(); }")
    for name in FEATURES.names():
        if name == "exec_output_similarity":
            continue
        val = FEATURES.create(name).extract(p.code1, p.code2)[0]
        assert 0.0 <= val <= 1.0


def test_exec_feature_prefers_precomputed_meta():
    feat = FEATURES.create("exec_output_similarity")
    out = feat.extract("System.out.println(1);", "System.out.println(2);", {"output": 0.42})
    assert out == [0.42]


def test_exec_feature_graceful_without_jdk_or_meta():
    feat = FEATURES.create("exec_output_similarity")
    out = feat.extract("x", "y", None)  # no meta, no JDK -> 0.0, no crash
    assert out == [0.0]


def test_feature_set_vector_shape_and_names():
    names = ["token_jaccard", "edit_ratio", "cyclomatic_ratio"]
    fs = FeatureSet.from_names(names)
    p = CodePair(code1="if(a){b();}", code2="if(c){d();}")
    v = fs.vector(p)
    assert v.shape == (3,)
    assert fs.columns == names
    X = fs.transform([p, p])
    assert X.shape == (2, 3)
