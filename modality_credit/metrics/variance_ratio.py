"""Claim 1.5 — empirical variance ratio validation for Corollary 2c."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from modality_credit.protocols import Estimator, Utility
from modality_credit.types import QueryInstance


def empirical_variance(estimator: Estimator,
                       utility: Utility,
                       samples: list[QueryInstance],
                       budget_B: int,
                       seeds: Iterable[int]) -> float:
    """Average across-seed variance of φ̂_k for a fixed sample set."""
    seeds = list(seeds)
    phi_runs = []  # shape: (n_seeds, n_samples, K)
    for s in seeds:
        # Re-instantiate estimator with new seed for stochasticity
        type(estimator).__init__(estimator, seed=s)
        phi_runs.append(np.array([
            estimator.estimate(utility, inst, budget_B=budget_B).phi
            for inst in samples
        ]))
    phi_runs = np.array(phi_runs)
    # Variance across seeds, then averaged over (sample, k)
    return float(np.var(phi_runs, axis=0).mean())


def theoretical_variance_ratio(K: int, L_ks: list[int]) -> float:
    """Cor 2c: (Σ 2^{L_k}) / 2^{Σ L_k}"""
    return float(sum(2 ** L for L in L_ks) / 2 ** sum(L_ks))
