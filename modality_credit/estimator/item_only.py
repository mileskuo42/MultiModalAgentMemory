"""Item-only Shapley baseline: outer game only, ψ = φ / L_k (uniform split)."""
from __future__ import annotations

import numpy as np

from modality_credit.estimator.base import BaseEstimator
from modality_credit.estimator.kernel_shap import kernel_shap_outer
from modality_credit.protocols import Utility
from modality_credit.types import AttributionResult, QueryInstance


class ItemOnlyEstimator(BaseEstimator):
    """Computes φ_k via Kernel SHAP, splits φ_k uniformly across the item's modalities."""

    def estimate(self, utility: Utility, query_inst: QueryInstance,
                 budget_B: int = 500) -> AttributionResult:
        K = query_inst.K
        L_ks = query_inst.L_ks

        phi = kernel_shap_outer(utility, query_inst, N=budget_B, seed=self.seed)

        max_L = max(L_ks)
        psi = np.full((K, max_L), np.nan, dtype=np.float64)
        for k in range(K):
            psi[k, :L_ks[k]] = phi[k] / L_ks[k]

        U_full = utility.evaluate(query_inst, [True] * K)
        U_empty = utility.evaluate(query_inst, [False] * K)
        denom = max(abs(U_full - U_empty), 1e-6)
        residual = float(abs(phi.sum() - (U_full - U_empty)) / denom)

        return AttributionResult(
            phi=phi, psi=psi,
            top_idx=np.argsort(-phi)[:2],
            conservation_residual=residual,
            U_full=U_full, U_empty=U_empty,
            meta={"estimator": self.name, "budget_B": budget_B},
        )
