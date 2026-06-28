"""Small, dependency-free text/code utilities shared by feature plugins."""

from __future__ import annotations

import re
from typing import List, Set

_TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d+|==|!=|<=|>=|&&|\|\||[{}()\[\];,.<>+\-*/%=&|^!?:]")
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")

# A compact set of decision/branch constructs used as a cyclomatic-complexity proxy.
DECISION_TOKENS = ("if", "for", "while", "case", "catch", "&&", "||", "?", "elif")

JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
    "class", "const", "continue", "default", "do", "double", "else", "enum",
    "extends", "final", "finally", "float", "for", "goto", "if", "implements",
    "import", "instanceof", "int", "interface", "long", "native", "new", "package",
    "private", "protected", "public", "return", "short", "static", "strictfp",
    "super", "switch", "synchronized", "this", "throw", "throws", "transient",
    "try", "void", "volatile", "while",
}


def tokenize(code: str) -> List[str]:
    """Language-agnostic lexical tokenizer (identifiers, numbers, operators)."""
    return _TOKEN_RE.findall(code or "")


def identifiers(code: str) -> List[str]:
    return _IDENT_RE.findall(code or "")


def char_ngrams(code: str, n: int = 3) -> Set[str]:
    s = re.sub(r"\s+", " ", code or "").strip()
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def ratio_sim(a: float, b: float) -> float:
    """Symmetric ratio similarity in [0, 1]; 1.0 when both are equal (incl. 0)."""
    if a == 0 and b == 0:
        return 1.0
    hi = max(abs(a), abs(b))
    lo = min(abs(a), abs(b))
    return lo / hi if hi else 1.0


def max_nesting_depth(code: str) -> int:
    depth = 0
    cur = 0
    for ch in code or "":
        if ch == "{":
            cur += 1
            depth = max(depth, cur)
        elif ch == "}":
            cur = max(0, cur - 1)
    return depth


def count_any(code: str, needles) -> int:
    toks = tokenize(code)
    tokset = toks
    total = 0
    # word-like decision tokens are matched against the token stream;
    # operator-like ones (&&, ||, ?) are matched directly.
    for nd in needles:
        if nd.isalpha():
            total += sum(1 for t in tokset if t == nd)
        else:
            total += (code or "").count(nd)
    return total
