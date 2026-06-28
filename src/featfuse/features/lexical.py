"""Lexical engineered features: surface-level similarity of two code fragments."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from ..registry import FEATURES
from .base import PairFeature
from ._textutil import char_ngrams, identifiers, jaccard, ratio_sim, tokenize


@FEATURES.register("token_jaccard", family="lexical")
class TokenJaccard(PairFeature):
    """Jaccard overlap of the two token *sets*."""

    name = "token_jaccard"
    family = "lexical"

    def extract(self, code1: str, code2: str, meta: Optional[Dict[str, Any]] = None) -> List[float]:
        return [jaccard(set(tokenize(code1)), set(tokenize(code2)))]


@FEATURES.register("char_ngram_jaccard", family="lexical")
class CharNgramJaccard(PairFeature):
    """Jaccard overlap of character 3-grams (robust to identifier renaming)."""

    name = "char_ngram_jaccard"
    family = "lexical"

    def __init__(self, n: int = 3):
        self.n = n

    def extract(self, code1: str, code2: str, meta: Optional[Dict[str, Any]] = None) -> List[float]:
        return [jaccard(char_ngrams(code1, self.n), char_ngrams(code2, self.n))]


@FEATURES.register("edit_ratio", family="lexical")
class EditRatio(PairFeature):
    """Normalised edit similarity (``difflib`` ratio) over the raw strings."""

    name = "edit_ratio"
    family = "lexical"

    def extract(self, code1: str, code2: str, meta: Optional[Dict[str, Any]] = None) -> List[float]:
        return [SequenceMatcher(None, code1 or "", code2 or "").ratio()]


@FEATURES.register("length_ratio", family="lexical")
class LengthRatio(PairFeature):
    """Ratio of token counts (shorter / longer)."""

    name = "length_ratio"
    family = "lexical"

    def extract(self, code1: str, code2: str, meta: Optional[Dict[str, Any]] = None) -> List[float]:
        return [ratio_sim(len(tokenize(code1)), len(tokenize(code2)))]


@FEATURES.register("identifier_jaccard", family="lexical")
class IdentifierJaccard(PairFeature):
    """Jaccard overlap of identifier sets (names of variables, methods, types)."""

    name = "identifier_jaccard"
    family = "lexical"

    def extract(self, code1: str, code2: str, meta: Optional[Dict[str, Any]] = None) -> List[float]:
        return [jaccard(set(identifiers(code1)), set(identifiers(code2)))]
