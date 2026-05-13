"""Pluggable logger — write to WandB, local JSON, both, or nothing."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLogger(ABC):
    @abstractmethod
    def log(self, metrics: dict[str, Any], step: int | None = None) -> None: ...
    @abstractmethod
    def save_artifact(self, path: str, name: str) -> None: ...
    @abstractmethod
    def finish(self) -> None: ...


class NullLogger(BaseLogger):
    """No-op logger for unit tests / dry runs."""
    def log(self, metrics, step=None): pass
    def save_artifact(self, path, name): pass
    def finish(self): pass


class CompositeLogger(BaseLogger):
    """Fan-out to multiple backends."""
    def __init__(self, backends: list[BaseLogger]):
        self._backends = backends
    def log(self, metrics, step=None):
        for b in self._backends: b.log(metrics, step)
    def save_artifact(self, path, name):
        for b in self._backends: b.save_artifact(path, name)
    def finish(self):
        for b in self._backends: b.finish()
