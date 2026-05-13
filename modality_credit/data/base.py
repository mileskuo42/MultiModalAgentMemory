"""Dataset base — abstracts M3-Bench / WorldMM / LongVideoBench / synthetic."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from modality_credit.types import QueryInstance


class BaseDataset(ABC):
    """Concrete subclasses implement loading + indexing; iteration is provided."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> QueryInstance: ...

    def __iter__(self) -> Iterator[QueryInstance]:
        for i in range(len(self)):
            yield self[i]

    @property
    def name(self) -> str:
        return type(self).__name__

    def filter(self, predicate) -> "FilteredDataset":
        """Return a view that yields only instances passing `predicate`."""
        return FilteredDataset(self, predicate)


class FilteredDataset(BaseDataset):
    def __init__(self, parent: BaseDataset, predicate):
        self.parent = parent
        self._indices = [i for i in range(len(parent)) if predicate(parent[i])]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> QueryInstance:
        return self.parent[self._indices[idx]]
