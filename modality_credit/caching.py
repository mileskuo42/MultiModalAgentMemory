"""Cache wrapper for any `Utility`. Idempotence is critical for Shapley."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

from modality_credit.protocols import Utility
from modality_credit.types import ItemMask, ModalityMask, QueryInstance


class CachedUtility:
    """Decorate a `Utility` with memoization on (instance_id, item_mask, modality_masks).

    Args:
        inner:     underlying utility.
        disk_dir:  if set, also persist to disk (survives across runs).
        max_mem:   max entries in memory cache before LRU eviction.
    """

    def __init__(self, inner: Utility, disk_dir: str | Path | None = None,
                 max_mem: int = 50_000):
        self._inner = inner
        self._mem: dict[str, float] = {}
        self._disk = Path(disk_dir) if disk_dir else None
        self._max_mem = max_mem
        if self._disk:
            self._disk.mkdir(parents=True, exist_ok=True)

    def evaluate(self,
                 query_inst: QueryInstance,
                 item_mask: ItemMask,
                 modality_masks: list[ModalityMask] | None = None) -> float:
        key = self._make_key(query_inst, item_mask, modality_masks)
        if key in self._mem:
            return self._mem[key]
        if self._disk and (path := self._disk / f"{key}.pkl").exists():
            with open(path, "rb") as f:
                v = pickle.load(f)
            self._mem[key] = v
            return v
        v = self._inner.evaluate(query_inst, item_mask, modality_masks)
        self._mem[key] = v
        if self._disk:
            with open(self._disk / f"{key}.pkl", "wb") as f:
                pickle.dump(v, f)
        self._evict_if_full()
        return v

    @staticmethod
    def _make_key(qi: QueryInstance,
                  item_mask: ItemMask,
                  modality_masks: list[ModalityMask] | None) -> str:
        payload = {
            "iid": qi.instance_id,
            "im": item_mask,
            "mm": modality_masks,
        }
        s = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    def _evict_if_full(self) -> None:
        # TODO(impl): replace with proper LRU; for now drop oldest 10% when full.
        if len(self._mem) > self._max_mem:
            for k in list(self._mem.keys())[: self._max_mem // 10]:
                del self._mem[k]

    @property
    def n_calls(self) -> int:
        return self._inner.n_calls

    @property
    def name(self) -> str:
        return f"cached({self._inner.name})"
