"""Flat Shapley baseline: treat every (item, modality) pair as one player.

Computationally expensive at K=4, L_k=4 (65,536 subsets); we use the same
Monte Carlo permutation estimator over Σ L_k players. Used to demonstrate
the wedge in Claim 1.5 and as Phase 2 baseline B1.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from modality_credit.estimator.base import BaseEstimator
from modality_credit.protocols import Utility
from modality_credit.types import AttributionResult, QueryInstance


class FlatShapleyEstimator(BaseEstimator):
    """Player = every (item, modality) patch; outer hierarchy ignored."""

    def estimate(self, utility: Utility, query_inst: QueryInstance,
                 budget_B: int = 10_000) -> AttributionResult:
        # TODO(impl):
        # 1. Enumerate all (item, modality) patches as a flat player list.
        # 2. Run a single MC-permutation Shapley over the flat list with N=budget_B.
        # 3. Aggregate flat scores back into (phi, psi) shapes for fair comparison.
        raise NotImplementedError
