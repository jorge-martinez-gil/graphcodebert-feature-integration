"""Neural fusion strategy plugins.

Importing this package registers every built-in fusion strategy in
:data:`featfuse.registry.FUSIONS`. Build one with
``FUSIONS.create("gated", embed_dim=768, feature_dim=11)`` (requires torch).
"""

from __future__ import annotations

from . import strategies  # noqa: F401  (side-effect: registration)

__all__ = ["strategies"]
