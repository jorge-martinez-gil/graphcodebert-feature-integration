"""Reproducibility helpers: deterministic seeding and run manifests.

Every benchmark run emits a ``manifest.json`` recording the seed, configuration,
software versions, platform and (when available) the git commit. This makes every
number traceable back to the exact code and configuration that produced it — a
prerequisite for the benchmark to be trusted and cited.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Seed Python, NumPy and (if installed) PyTorch.

    Returns the seed so callers can log it. ``deterministic`` additionally requests
    deterministic cuDNN behaviour when torch is present.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is an optional dependency
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:  # pragma: no cover - torch optional
        pass
    return seed


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.decode().strip()
    except Exception:  # pragma: no cover
        pass
    return None


def _version(mod: str) -> Optional[str]:
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def config_hash(config: Dict[str, Any]) -> str:
    """Stable SHA-256 over a configuration dict (order-independent)."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def make_manifest(config: Dict[str, Any], seed: int, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a reproducibility manifest for a run."""
    manifest: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "config_hash": config_hash(config),
        "config": config,
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": {
            m: _version(m)
            for m in ("featfuse", "numpy", "scipy", "sklearn", "torch", "transformers", "pandas")
        },
    }
    if extra:
        manifest.update(extra)
    return manifest
