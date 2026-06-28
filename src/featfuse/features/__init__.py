"""Engineered feature plugins.

Importing this package registers every built-in feature in
:data:`featfuse.registry.FEATURES`. Add a new feature by subclassing
:class:`featfuse.features.base.PairFeature` and decorating it with
``@FEATURES.register("my_name", family="...")`` (see ``docs/adding_a_feature.md``).
"""

from __future__ import annotations

from .base import FeatureSet, PairFeature
from . import lexical, structural, execution  # noqa: F401  (side-effect: registration)

__all__ = ["FeatureSet", "PairFeature"]
