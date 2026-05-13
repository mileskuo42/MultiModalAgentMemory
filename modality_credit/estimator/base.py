"""ABC for estimators. Subclass for shared logic; or just implement the
`Estimator` Protocol directly (preferred for one-off baselines)."""
from __future__ import annotations

from abc import ABC, abstractmethod

from modality_credit.protocols import Utility
from modality_credit.types import AttributionResult, QueryInstance


class BaseEstimator(ABC):
    """Optional base class; provides `name` boilerplate."""

    def __init__(self, *, seed: int = 42):
        self.seed = seed

    @abstractmethod
    def estimate(self,
                 utility: Utility,
                 query_inst: QueryInstance,
                 budget_B: int = 500) -> AttributionResult:
        ...

    @property
    def name(self) -> str:
        return type(self).__name__
