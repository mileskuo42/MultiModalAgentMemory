"""Algorithm 1 — closed-form sample allocation policy.

Derivation:
    Total variance:  V = 2^K / N_out + Σ_k 2^{L_k} / N_in^(k)
    Budget:          B = N_out + Σ_k N_in^(k)
    Lagrangian:      L = V + λ (Σ N - B)
    ∂L/∂N_i = 0  ⇒   N_i^* ∝ √(coefficient_i)

Therefore:
    N_out^*       ∝ √(2^K)
    N_in^(k)^*    ∝ √(2^{L_k})

This is the ONLY new algorithmic contribution at the budget level. Implementations
must preserve the closed-form (no learned components, no tuning).
"""
from __future__ import annotations

import numpy as np


def allocate_budget(K: int, L_ks: list[int], B: int,
                    min_per_game: int = 10) -> tuple[int, list[int]]:
    """Closed-form optimal allocation under Prop 2 variance bounds.

    Args:
        K:            number of items.
        L_ks:         list of within-item player counts (length K).
        B:            total utility-query budget.
        min_per_game: floor to keep variance estimable.

    Returns:
        (N_out, N_in_list), each guaranteed to be >= min_per_game.

    Invariant: sum(N_in_list) + N_out <= B + K * (min_per_game).
               (Floor may slightly inflate; we accept this for simplicity.)
    """
    assert K == len(L_ks), f"K={K} but len(L_ks)={len(L_ks)}"
    assert B >= (K + 1) * min_per_game, (
        f"Budget B={B} too small for K={K} games at min {min_per_game} each"
    )

    sqrt_outer = float(np.sqrt(2 ** K))
    sqrt_inners = np.sqrt(np.array([2 ** L for L in L_ks], dtype=float))
    Z = sqrt_outer + sqrt_inners.sum()

    N_out = max(int(np.floor(B * sqrt_outer / Z)), min_per_game)
    N_in = [max(int(np.floor(B * s / Z)), min_per_game) for s in sqrt_inners]
    return N_out, N_in


def variance_ratio_bound(K: int, L_ks: list[int]) -> float:
    """Corollary 2c: Var(hierarchical) / Var(flat) ≤ (Σ 2^{L_k}) / 2^{Σ L_k}."""
    return float(sum(2 ** L for L in L_ks) / 2 ** sum(L_ks))
