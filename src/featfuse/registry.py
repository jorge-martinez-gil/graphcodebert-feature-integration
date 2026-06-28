"""A tiny, dependency-free plugin registry.

Every FeatFuse extension point (features, fusion strategies, models, datasets) is a
:class:`Registry`. Registering a plugin is a one-line decorator; discovering and
instantiating plugins by name is what lets the YAML-driven benchmark stay generic.

Example
-------
>>> from featfuse.registry import FEATURES
>>> @FEATURES.register("my_feature")
... class MyFeature:
...     ...
>>> FEATURES.create("my_feature")          # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Tuple, TypeVar

T = TypeVar("T")


class Registry:
    """Maps string names to factories (classes or callables)."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: Dict[str, Callable[..., Any]] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, **meta: Any) -> Callable[[T], T]:
        """Decorator that registers ``obj`` under ``name``.

        ``meta`` may carry human-readable attributes (e.g. ``family="lexical"``)
        that are surfaced by ``featfuse list``.
        """

        name = name.lower()

        def _wrap(obj: T) -> T:
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' is already registered")
            self._items[name] = obj  # type: ignore[assignment]
            self._meta[name] = dict(meta)
            return obj

        return _wrap

    def get(self, name: str) -> Callable[..., Any]:
        key = name.lower()
        if key not in self._items:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Available: {', '.join(self.names()) or '(none)'}"
            )
        return self._items[key]

    def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Instantiate the registered factory ``name``."""
        return self.get(name)(*args, **kwargs)

    def names(self) -> List[str]:
        return sorted(self._items)

    def meta(self, name: str) -> Dict[str, Any]:
        return self._meta.get(name.lower(), {})

    def items(self) -> Iterable[Tuple[str, Callable[..., Any]]]:
        return self._items.items()

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Registry({self.kind}, {self.names()})"


# The four canonical extension points.
FEATURES = Registry("feature")
FUSIONS = Registry("fusion")
MODELS = Registry("model")
DATASETS = Registry("dataset")
