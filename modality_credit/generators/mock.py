"""Deterministic mock generator for unit testing.

Implements a programmable Generator that returns pre-specified outputs based on
the input string. Used in tests to verify Owen estimator math without GPU.
"""
from __future__ import annotations

from collections.abc import Callable

from modality_credit.generators.base import BaseGenerator


class MockGenerator(BaseGenerator):
    """Deterministic generator driven by a user-supplied response function.

    Example:
        gen = MockGenerator(lambda q, ctx: "yes" if "tomato" in ctx else "no")

    Args:
        response_fn: callable (query, context_str) -> str
    """

    def __init__(self, response_fn: Callable[[str, str], str]):
        self._fn = response_fn
        self._n_calls = 0

    def generate(self, query: str, context_str: str, *,
                 max_new_tokens: int = 128) -> str:
        self._n_calls += 1
        return self._fn(query, context_str)

    @property
    def name(self) -> str:
        return "mock"

    @property
    def n_calls(self) -> int:
        return self._n_calls
