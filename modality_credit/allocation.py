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
                    min_per_game: int = 10,
                    top_n_inner: int | None = None) -> tuple[int, list[int]]:
    """Closed-form optimal allocation under Prop 2 variance bounds.

    Args:
        K:            number of items.
        L_ks:         list of within-item player counts (length K).
        B:            total utility-query budget.
        min_per_game: floor to keep variance estimable.
        top_n_inner:  number of items that will actually be inner-gamed.
                      None = K (legacy: spread budget across all items).
                      In practice, OwenEstimator only runs inner-game on the
                      top_n by φ; the budget should reflect that, otherwise
                      the closed-form is not at the empirical optimum.

    Returns:
        (N_out, N_in_list), each guaranteed to be >= min_per_game.
        N_in_list has length K; caller selects entries by top_idx.

    Invariant: N_out + top_n_inner * mean(N_in_top) ≈ B.
    """
    assert K == len(L_ks), f"K={K} but len(L_ks)={len(L_ks)}"
    if top_n_inner is None:
        top_n_inner = K
    assert 1 <= top_n_inner <= K, f"top_n_inner={top_n_inner} not in [1, K={K}]"
    assert B >= (top_n_inner + 1) * min_per_game, (
        f"Budget B={B} too small for {top_n_inner}+1 games at min {min_per_game} each"
    )

    sqrt_outer = float(np.sqrt(2 ** K))
    sqrt_inners = np.sqrt(np.array([2 ** L for L in L_ks], dtype=float))
    # The actual inner cost is incurred only by top_n_inner items. We don't know
    # which items they'll be pre-allocation, so we assume the top_n_inner LARGEST
    # √(2^{L_k}) values for the budget denominator (worst-case for non-uniform L).
    sqrt_inners_top = np.sort(sqrt_inners)[::-1][:top_n_inner]
    Z = sqrt_outer + sqrt_inners_top.sum()

    N_out = max(int(np.floor(B * sqrt_outer / Z)), min_per_game)
    N_in = [max(int(np.floor(B * s / Z)), min_per_game) for s in sqrt_inners]
    return N_out, N_in


def variance_ratio_bound(K: int, L_ks: list[int]) -> float:
    """Corollary 2c: Var(hierarchical) / Var(flat) ≤ (Σ 2^{L_k}) / 2^{Σ L_k}."""
    return float(sum(2 ** L for L in L_ks) / 2 ** sum(L_ks))
