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
                 images: list | None = None,
                 max_new_tokens: int = 128) -> str:
        # Mock generator ignores images by default. If the response_fn wants
        # them, it can use the 3-arg signature: response_fn(query, ctx, images).
        self._n_calls += 1
        try:
            return self._fn(query, context_str, images)  # type: ignore[call-arg]
        except TypeError:
            return self._fn(query, context_str)

    @property
    def name(self) -> str:
        return "mock"

    @property
    def n_calls(self) -> int:
        return self._n_calls
