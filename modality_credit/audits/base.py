"""Audit base — pre-pilot sanity check. Every audit returns a structured verdict."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from modality_credit.protocols import Utility
from modality_credit.types import QueryInstance


class BaseAudit(ABC):
    """Every audit MUST return a dict with keys: verdict, summary, data."""

    VERDICT_PASS = "PASS"
    VERDICT_WARN = "WARN"
    VERDICT_FAIL = "FAIL"

    @abstractmethod
    def run(self,
            utility: Utility,
            samples: list[QueryInstance]) -> dict[str, Any]:
        """Run the audit on the given samples.

        Returns:
            {"verdict": "PASS"|"WARN"|"FAIL", "summary": str, "data": ...}
        """
        ...

    @property
    def name(self) -> str:
        return type(self).__name__
