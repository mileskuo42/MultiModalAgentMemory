"""Concrete `Utility` — wires Generator + Masker + Verifier."""
from __future__ import annotations

from typing import Any

from modality_credit.protocols import Generator, Masker, Utility, Verifier
from modality_credit.types import ItemMask, ModalityMask, QueryInstance


class StandardUtility:
    """Default implementation of the `Utility` protocol.

    Wrap with `CachedUtility` (modality_credit.caching) to enforce idempotence
    across repeated calls — Owen estimator may query the same coalition 10×.

    Args:
        generator: any object satisfying `Generator`.
        verifier:  any callable matching `Verifier`.
        masker:    any object satisfying `Masker`.
    """

    def __init__(self, generator: Generator, verifier: Verifier, masker: Masker):
        self._gen = generator
        self._ver = verifier
        self._mask = masker
        self._n_calls = 0

    def evaluate(self,
                 query_inst: QueryInstance,
                 item_mask: ItemMask,
                 modality_masks: list[ModalityMask] | None = None) -> float:
        context = self._mask.apply(query_inst.memory, item_mask, modality_masks)
        y = self._gen.generate(query_inst.query, context)
        self._n_calls += 1
        return float(self._ver(y, query_inst.gold_answer))

    @property
    def n_calls(self) -> int:
        return self._n_calls

    @property
    def name(self) -> str:
        return f"{self._gen.name}+{type(self._mask).__name__}"
