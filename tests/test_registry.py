import pytest

from featfuse.registry import Registry


def test_register_and_create():
    reg = Registry("thing")

    @reg.register("foo", family="x")
    class Foo:
        def __init__(self, a=1):
            self.a = a

    assert "foo" in reg
    assert reg.names() == ["foo"]
    assert reg.meta("foo") == {"family": "x"}
    assert reg.create("foo", a=5).a == 5


def test_duplicate_raises():
    reg = Registry("thing")
    reg.register("a")(lambda: 1)
    with pytest.raises(KeyError):
        reg.register("a")(lambda: 2)


def test_unknown_raises():
    reg = Registry("thing")
    with pytest.raises(KeyError):
        reg.get("nope")


def test_builtin_registries_populated():
    import featfuse.features  # noqa: F401
    import featfuse.fusion  # noqa: F401
    import featfuse.models  # noqa: F401
    import featfuse.data  # noqa: F401
    from featfuse.registry import FEATURES, FUSIONS, MODELS, DATASETS

    assert len(FEATURES) >= 10
    assert {"concat", "gated", "attention", "residual", "film"} <= set(FUSIONS.names())
    assert "toy" in MODELS
    assert "irplag" in DATASETS
