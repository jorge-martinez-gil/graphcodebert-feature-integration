"""Structural / software-metric engineered features.

These compare *structural* properties of the two fragments (nesting, branching,
size, keyword usage) rather than surface tokens. They act as cheap, interpretable
proxies for AST / control-flow signal and complement the lexical features.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..registry import FEATURES
from .base import PairFeature
from ._textutil import (
    DECISION_TOKENS,
    JAVA_KEYWORDS,
    count_any,
    identifiers,
    jaccard,
    max_nesting_depth,
    ratio_sim,
    tokenize,
)


@FEATURES.register("nesting_depth_ratio", family="structural")
class NestingDepthRatio(PairFeature):
    """Similarity of maximum block-nesting depth (a structural complexity proxy)."""

    name = "nesting_depth_ratio"
    family = "structural"

    def extract(self, code1: str, code2: str, meta: Optional[Dict[str, Any]] = None) -> List[float]:
        return [ratio_sim(max_nesting_depth(code1), max_nesting_depth(code2))]


@FEATURES.register("cyclomatic_ratio", family="structural")
class CyclomaticRatio(PairFeature):
    """Similarity of a cyclomatic-complexity proxy (count of decision points)."""

    name = "cyclomatic_ratio"
    family = "structural"

    def extract(self, code1: str, code2: str, meta: Optional[Dict[str, Any]] = None) -> List[float]:
        c1 = 1 + count_any(code1, DECISION_TOKENS)
        c2 = 1 + count_any(code2, DECISION_TOKENS)
        return [ratio_sim(c1, c2)]


@FEATURES.register("line_count_ratio", family="structural")
class LineCountRatio(PairFeature):
    """Similarity of the number of non-empty lines."""

    name = "line_count_ratio"
    family = "structural"

    def extract(self, code1: str, code2: str, meta: Optional[Dict[str, Any]] = None) -> List[float]:
        n1 = len([ln for ln in (code1 or "").splitlines() if ln.strip()])
        n2 = len([ln for ln in (code2 or "").splitlines() if ln.strip()])
        return [ratio_sim(n1, n2)]


@FEATURES.register("keyword_jaccard", family="structural")
class KeywordJaccard(PairFeature):
    """Jaccard overlap of the *reserved keywords* used by each fragment."""

    name = "keyword_jaccard"
    family = "structural"

    def extract(self, code1: str, code2: str, meta: Optional[Dict[str, Any]] = None) -> List[float]:
        k1 = {t for t in tokenize(code1) if t in JAVA_KEYWORDS}
        k2 = {t for t in tokenize(code2) if t in JAVA_KEYWORDS}
        return [jaccard(k1, k2)]


@FEATURES.register("unique_identifier_ratio", family="structural")
class UniqueIdentifierRatio(PairFeature):
    """Similarity of vocabulary richness (#unique identifiers)."""

    name = "unique_identifier_ratio"
    family = "structural"

    def extract(self, code1: str, code2: str, meta: Optional[Dict[str, Any]] = None) -> List[float]:
        return [ratio_sim(len(set(identifiers(code1))), len(set(identifiers(code2))))]
