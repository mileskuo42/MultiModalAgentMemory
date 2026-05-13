"""Top-level orchestrator. Wires Generator + Verifier + Masker + Utility + Estimator
into a single object that scripts can invoke.

Usage:
    pipe = Pipeline.from_config(cfg)
    attrs = pipe.attribute(samples)
    df    = pipe.pareto(samples, attrs, tau_grid)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

from modality_credit.caching import CachedUtility
from modality_credit.metrics.pareto import evaluate_pareto
from modality_credit.protocols import (
    Estimator, Generator, Logger, Masker, Pruner, Utility, Verifier,
)
from modality_credit.types import AttributionResult, QueryInstance
from modality_credit.utility import StandardUtility


@dataclass
class Pipeline:
    generator: Generator
    verifier: Verifier
    masker: Masker
    estimator: Estimator
    pruner: Pruner
    logger: Logger
    cache_dir: str | None = None

    def __post_init__(self):
        utility = StandardUtility(self.generator, self.verifier, self.masker)
        self.utility: Utility = (
            CachedUtility(utility, disk_dir=self.cache_dir)
            if self.cache_dir else utility
        )

    @classmethod
    def from_config(cls, cfg: Any) -> "Pipeline":
        """Build from Hydra config. Implementations of from_config should
        construct each component via importlib + factory pattern."""
        # TODO(impl): use cfg.generator._target_ to dynamically construct.
        raise NotImplementedError

    def attribute(self,
                  samples: Iterable[QueryInstance],
                  budget_B: int = 500) -> list[AttributionResult]:
        return [self.estimator.estimate(self.utility, s, budget_B=budget_B)
                for s in samples]

    def pareto(self,
               samples: list[QueryInstance],
               attributions: list[AttributionResult],
               tau_grid: Iterable[tuple[float, float]]) -> pd.DataFrame:
        return evaluate_pareto(self.utility, samples, attributions,
                               self.pruner, tau_grid)

    def close(self) -> None:
        self.logger.finish()
