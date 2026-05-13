"""Pruner base. Implementations: owen_threshold (default), item_threshold, random."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from modality_credit.protocols import Masker
from modality_credit.types import AttributionResult, PrunedContext, QueryInstance


class BasePruner(ABC):
    """ABC for pruning strategies."""

    def __init__(self, masker: Masker):
        self._masker = masker

    @abstractmethod
    def prune(self,
              query_inst: QueryInstance,
              attribution: AttributionResult,
              **kwargs: Any) -> PrunedContext:
        ...

    @property
    def name(self) -> str:
        return type(self).__name__
