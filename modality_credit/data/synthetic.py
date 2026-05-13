"""Synthetic generator for Claim 1.5 (theory validation).

Injects ground-truth decisive (item k*, modality ℓ*) so we can measure
attribution recall + variance ratio against a known answer.
"""
from __future__ import annotations

import random
from typing import Iterable

from modality_credit.data.base import BaseDataset
from modality_credit.types import MemoryItem, Modality, QueryInstance


class SyntheticDataset(BaseDataset):
    """K × L_k synthetic QA with injected decisive modality."""

    def __init__(self, K: int, L_k: int, n: int = 50,
                 modality_set: list[Modality] | None = None,
                 seed: int = 42):
        self.K = K
        self.L_k = L_k
        self.n = n
        self.modality_set = modality_set or ["vision", "text", "audio"][:L_k]
        self.seed = seed
        self._cache: list[QueryInstance] = []
        self._ground_truth: list[tuple[int, Modality]] = []
        # TODO(impl): pre-generate all `n` samples eagerly for reproducibility.
        raise NotImplementedError

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> QueryInstance:
        return self._cache[idx]

    def ground_truth(self, idx: int) -> tuple[int, Modality]:
        """Return injected (k*, ℓ*) for sample idx."""
        return self._ground_truth[idx]

    @property
    def name(self) -> str:
        return f"synthetic_K{self.K}_L{self.L_k}"
