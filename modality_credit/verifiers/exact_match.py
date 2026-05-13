"""Exact-match and normalized-string verifiers."""
from __future__ import annotations

import re
import string


class ExactMatchVerifier:
    """Strict exact match after light normalization (lower, strip, collapse spaces)."""

    def __init__(self, *, lower: bool = True, strip_punct: bool = True):
        self.lower = lower
        self.strip_punct = strip_punct

    def __call__(self, generated: str, gold: str) -> bool:
        return self._norm(generated) == self._norm(gold)

    def _norm(self, s: str) -> str:
        s = s.strip()
        if self.lower:
            s = s.lower()
        if self.strip_punct:
            s = s.translate(str.maketrans("", "", string.punctuation))
        return re.sub(r"\s+", " ", s)
