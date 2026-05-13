"""Algorithm 3 — modality-pruned retrieval based on (φ, ψ) thresholds.

    M̃(Q) = { v_k^(ℓ) : φ_k > τ_1  ∧  ψ_{k,ℓ} > τ_2 }
"""
from __future__ import annotations

import numpy as np

from modality_credit.pruning.base import BasePruner
from modality_credit.types import (
    AttributionResult, ModalityMask, PrunedContext, QueryInstance,
)


class OwenThresholdPruner(BasePruner):
    """Keep (item, modality) patches whose Owen scores exceed (τ_1, τ_2)."""

    def prune(self,
              query_inst: QueryInstance,
              attribution: AttributionResult,
              *,
              tau_1: float = 0.1,
              tau_2: float = 0.1) -> PrunedContext:
        K = query_inst.K
        item_mask = [bool(attribution.phi[k] > tau_1) for k in range(K)]
        modality_masks: list[ModalityMask] = []
        total, kept = 0, 0
        for k, item in enumerate(query_inst.memory):
            mm: ModalityMask = {}
            for ell_idx, m in enumerate(item.modalities.keys()):
                psi_v = attribution.psi[k, ell_idx]
                keep = item_mask[k] and not np.isnan(psi_v) and (psi_v > tau_2)
                mm[m] = bool(keep)
                total += 1
                kept += int(keep)
            modality_masks.append(mm)

        context = self._masker.apply(query_inst.memory, item_mask, modality_masks)
        return PrunedContext(
            context_string=context,
            item_mask=item_mask,
            modality_masks=modality_masks,
            retention_ratio=kept / max(total, 1),
            meta={"tau_1": tau_1, "tau_2": tau_2, "pruner": self.name},
        )
