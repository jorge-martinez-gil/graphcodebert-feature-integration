"""Dataset loaders.

Importing this package registers built-in datasets in
:data:`featfuse.registry.DATASETS`. ``irplag`` ships today; BigCloneBench, POJ-104
and PoolC loaders are the next planned additions (see ``docs/roadmap.md``).
"""

from __future__ import annotations

from .base import Dataset, seeded_split  # noqa: F401
from . import irplag  # noqa: F401  (side-effect: registration)

__all__ = ["Dataset", "seeded_split"]
