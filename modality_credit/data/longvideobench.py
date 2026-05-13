"""LongVideoBench loader — Phase 4B (K=8 regime)."""
from __future__ import annotations

from pathlib import Path

from modality_credit.data.base import BaseDataset
from modality_credit.types import QueryInstance


class LongVideoBenchDataset(BaseDataset):
    """Long-form video QA, K configurable up to 8."""

    def __init__(self, root: str | Path, K: int = 8, n: int | None = None):
        self.root = Path(root)
        self.K = K
        self.n = n
        # TODO(impl): download via huggingface_hub, parse, build retrieval cache.
        # LongVideoBench source: https://huggingface.co/datasets/longvideobench/LongVideoBench
        raise NotImplementedError

    def __len__(self) -> int: raise NotImplementedError
    def __getitem__(self, idx: int) -> QueryInstance: raise NotImplementedError
