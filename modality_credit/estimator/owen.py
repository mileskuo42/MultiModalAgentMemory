"""Owen-value estimator: outer Kernel SHAP + inner MC permutation on top-N items.

This is the primary contribution: composes the two estimators with Algorithm 1
for budget allocation, and produces a unified AttributionResult.
"""
from __future__ import annotations

import time

import numpy as np

from modality_credit.allocation import allocate_budget
from modality_credit.estimator.base import BaseEstimator
from modality_credit.estimator.kernel_shap import kernel_shap_outer
from modality_credit.estimator.mc_permutation import mc_perm_inner
from modality_credit.protocols import Utility
from modality_credit.types import AttributionResult, QueryInstance


class OwenEstimator(BaseEstimator):
    """Hierarchical Owen-value attribution over (item × modality).

    Args:
        top_n_inner:     number of top items (by φ) on which to compute ψ.
                         Default 2 trades inner-game fidelity for budget.
        seed_attention:  use Qwen-VL attention logits to seed KSHAP coalitions
                         (Claim 1.1 ablation).
        budget_floor:    minimum budget per game (passed to Algorithm 1).
        seed:            random seed.
    """

    def __init__(self, *,
                 top_n_inner: int = 2,
                 seed_attention: bool = False,
                 budget_floor: int = 10,
                 seed: int = 42):
        super().__init__(seed=seed)
        self.top_n_inner = top_n_inner
        self.seed_attention = seed_attention
        self.budget_floor = budget_floor

    def estimate(self, utility: Utility, query_inst: QueryInstance,
                 budget_B: int = 500) -> AttributionResult:
        K = query_inst.K
        L_ks = query_inst.L_ks
        t0 = time.time()

        # --- Algorithm 1: closed-form allocation
        N_out, N_in_list = allocate_budget(K, L_ks, budget_B,
                                           min_per_game=self.budget_floor,
                                           top_n_inner=self.top_n_inner)

        # --- Outer game: φ_k
        phi = kernel_shap_outer(utility, query_inst, N=N_out,
                                seed=self.seed,
                                seed_attention=self.seed_attention)

        # --- Pick top-N items for inner game
        top_idx = np.argsort(-phi)[: self.top_n_inner]

        # --- Inner games: ψ_{k,ℓ} for k ∈ top_idx
        max_L = max(L_ks)
        psi = np.full((K, max_L), np.nan, dtype=np.float64)
        for k in top_idx:
            psi_k = mc_perm_inner(utility, query_inst,
                                  item_idx=int(k),
                                  N=N_in_list[k],
                                  seed=self.seed + int(k))
            psi[k, :len(psi_k)] = psi_k

        # --- Conservation residual
        U_full = utility.evaluate(query_inst, [True] * K)
        U_empty = utility.evaluate(query_inst, [False] * K)
        denom = max(abs(U_full - U_empty), 1e-6)
        residual = float(abs(phi.sum() - (U_full - U_empty)) / denom)

        return AttributionResult(
            phi=phi,
            psi=psi,
            top_idx=top_idx,
            conservation_residual=residual,
            U_full=U_full,
            U_empty=U_empty,
            meta={
                "estimator": self.name,
                "N_out": N_out,
                "N_in": N_in_list,
                "budget_B": budget_B,
                "wall_clock_sec": time.time() - t0,
                "seed_attention": self.seed_attention,
            },
        )
