"""Core data types shared across FeatFuse."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CodePair:
    """A single labelled pair of source-code fragments.

    Attributes
    ----------
    code1, code2:
        Raw source code of the two fragments.
    label:
        Ground-truth class. For clone/similarity detection this is ``1`` (clone /
        similar) or ``0`` (not a clone). ``None`` for unlabelled inference inputs.
    index:
        Optional stable identifier, used to make results traceable to the dataset.
    meta:
        Free-form metadata (e.g. precomputed features, language, source file).
    """

    code1: str
    code2: str
    label: Optional[int] = None
    index: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, rec: Dict[str, Any]) -> "CodePair":
        """Build a :class:`CodePair` from a raw dataset record.

        Accepts the IR-Plag schema (``code1``/``code2``/``score``) as well as the
        common ``label`` key, and preserves any extra keys (such as the precomputed
        ``output`` execution-similarity feature) in :attr:`meta`.
        """
        label = rec.get("score", rec.get("label"))
        meta = {k: v for k, v in rec.items() if k not in {"code1", "code2", "score", "label", "index"}}
        return cls(
            code1=rec["code1"],
            code2=rec["code2"],
            label=None if label is None else int(label),
            index=rec.get("index"),
            meta=meta,
        )
