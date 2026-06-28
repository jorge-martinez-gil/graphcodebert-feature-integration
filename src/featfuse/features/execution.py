"""Execution-based engineered feature (the paper's original signal).

The original work compiles and runs each Java fragment and compares their *standard
output* with a sequence-matching ratio. We preserve that behaviour, but make it
robust and reproducible:

* If a precomputed value is available in ``meta`` (the dataset's ``output`` field),
  it is used directly — no JVM required, so the benchmark reproduces anywhere.
* Otherwise, if a JDK (``javac``/``java``) is available, the value is computed.
* Otherwise the feature degrades gracefully to ``0.0`` and records that the JDK was
  missing, instead of crashing the whole run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from ..registry import FEATURES
from .base import PairFeature


def _have_jdk() -> bool:
    return shutil.which("javac") is not None and shutil.which("java") is not None


def _run_java(code: str, workdir: str) -> Optional[str]:
    class_name = "Test"
    src = os.path.join(workdir, f"{class_name}.java")
    with open(src, "w", encoding="utf-8") as fh:
        fh.write(f"public class {class_name} {{\n{code}\n}}")
    try:
        subprocess.run(["javac", src], check=True, cwd=workdir, stderr=subprocess.PIPE, timeout=30)
        res = subprocess.run(
            ["java", class_name], cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=30,
        )
        return res.stdout.strip()
    except Exception:
        return None


@FEATURES.register("exec_output_similarity", family="execution")
class ExecOutputSimilarity(PairFeature):
    """Similarity of program standard output (the paper's additional feature)."""

    name = "exec_output_similarity"
    family = "execution"

    def __init__(self, meta_key: str = "output", allow_compute: bool = True):
        self.meta_key = meta_key
        self.allow_compute = allow_compute

    def extract(self, code1: str, code2: str, meta: Optional[Dict[str, Any]] = None) -> List[float]:
        # 1) Prefer a precomputed value carried by the dataset (fully reproducible).
        if meta and self.meta_key in meta and meta[self.meta_key] is not None:
            try:
                return [float(meta[self.meta_key])]
            except (TypeError, ValueError):
                pass
        # 2) Compute via the JVM if available.
        if self.allow_compute and _have_jdk():
            with tempfile.TemporaryDirectory() as wd:
                o1 = _run_java(code1, wd)
                o2 = _run_java(code2, wd)
            if o1 is None or o2 is None:
                return [0.0]
            return [SequenceMatcher(None, o1, o2).ratio()]
        # 3) Graceful fallback: feature unavailable.
        return [0.0]
