"""Monte Carlo permutation Shapley for the inner game.

Each inner game has at most L_k ≤ 6 modalities — we can afford the classical
permutation estimator. All other items are frozen (full modality set retained).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from modality_credit.protocols import Utility
from modality_credit.types import Modality, QueryInstance


def mc_perm_inner(utility: Utility,
                  query_inst: QueryInstance,
                  item_idx: int,
                  N: int,
                  *,
                  seed: int = 42) -> NDArray[np.float64]:
    """Estimate ψ_{k,ℓ} via N permutation samples of item k's modalities.

    Args:
        item_idx:  k — only this item's modalities are toggled; others are
                   frozen with all modalities present.

    Returns: psi, shape (L_k,) in the modality order of the item's `modalities` dict.
    """
    K = query_inst.K
    item = query_inst.memory[item_idx]
    modalities: list[Modality] = list(item.modalities.keys())
    L = len(modalities)
    rng = np.random.default_rng(seed)

    psi = np.zeros(L, dtype=np.float64)
    all_items_on = [True] * K

    # Frozen other items: full modality set
    other_modality_masks_template = [
        {m: True for m in qi_item.modalities}
        for qi_item in query_inst.memory
    ]

    for _ in range(N):
        perm = rng.permutation(L)
        # start with all modalities of item k OFF
        current_mm = other_modality_masks_template.copy()
        current_mm[item_idx] = {m: False for m in modalities}

        u_prev = utility.evaluate(query_inst, all_items_on, current_mm)
        for ell_idx in perm:
            m = modalities[ell_idx]
            current_mm[item_idx][m] = True
            u_curr = utility.evaluate(query_inst, all_items_on, current_mm)
            psi[ell_idx] += (u_curr - u_prev)
            u_prev = u_curr

    psi /= N
    return psi
