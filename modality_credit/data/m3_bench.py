"""M3-Bench loader (uses cached retrieval output from vendor/m3_agent)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from modality_credit.data.base import BaseDataset
from modality_credit.types import MemoryItem, QueryInstance

Split = Literal["train", "val", "test"]


class M3BenchDataset(BaseDataset):
    """Load M3-Bench QA + pre-retrieved top-K items.

    Expected layout (after running scripts/00_setup.sh):
        vendor/m3_agent/data/m3_bench.json   — QA pairs + retrieval cache
        vendor/m3_agent/data/items/*.json     — per-item modality content
    """

    def __init__(self, vendor_root: str | Path = "vendor/m3_agent",
                 split: Split = "val", K: int = 4, n: int | None = None):
        self.root = Path(vendor_root)
        self.split = split
        self.K = K
        self.n = n
        # TODO(impl):
        # 1. Load m3_bench.json
        # 2. Filter to `split` and at most `n` items.
        # 3. Pre-load items db (or load lazily in __getitem__).
        self._entries: list[dict] = []
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> QueryInstance:
        # TODO(impl):
        # 1. Pull entry = self._entries[idx]
        # 2. retrieved_ids = entry["retrieved_top_k"][:self.K]
        # 3. memory = [build_memory_item(self._items_db[i]) for i in retrieved_ids]
        # 4. Return QueryInstance(instance_id=entry["id"], query=..., memory=..., gold_answer=...)
        raise NotImplementedError


def build_memory_item(raw: dict) -> MemoryItem:
    """Convert a raw M3-Agent item record into our MemoryItem."""
    # TODO(impl): load vision frame from raw["frame_path"]; pull caption / audio
    # transcript / scene metadata; populate modalities dict.
    raise NotImplementedError
