"""Random-drop pruner — critical control baseline for Claim 2.

We MUST significantly outperform random at matched retention or attribution
adds no value.
"""
from __future__ import annotations

import numpy as np

from modality_credit.pruning.base import BasePruner
from modality_credit.types import AttributionResult, ModalityMask, PrunedContext, QueryInstance


class RandomDropPruner(BasePruner):
    """Drop random (item, modality) patches until target retention is met.

    Ignores `attribution` entirely — this is the critical control baseline for
    Claim 2 (modality-pruned retrieval). Our Owen-pruned variant must
    significantly beat random at matched retention or attribution adds nothing.
    """

    def prune(self,
              query_inst: QueryInstance,
              attribution: AttributionResult,
              *,
              retention: float = 0.4,
              seed: int = 42) -> PrunedContext:
        rng = np.random.default_rng(seed)
        K = query_inst.K
        all_patches: list[tuple[int, str]] = []
        for k, item in enumerate(query_inst.memory):
            for m in item.modalities.keys():
                all_patches.append((k, m))
        total = len(all_patches)
        n_keep = max(int(np.floor(retention * total)), 0)
        keep_idx = (rng.choice(total, size=n_keep, replace=False)
                    if n_keep > 0 else np.array([], dtype=int))
        keep_set = {all_patches[i] for i in keep_idx}

        item_mask = [False] * K
        modality_masks: list[ModalityMask] = []
        for k, item in enumerate(query_inst.memory):
            mm: ModalityMask = {}
            for m in item.modalities.keys():
                kept = (k, m) in keep_set
                mm[m] = kept  # type: ignore[index]
                if kept:
                    item_mask[k] = True
            modality_masks.append(mm)

        context, _images = self._masker.apply(
            query_inst.memory, item_mask, modality_masks,
        )
        actual_retention = n_keep / total if total > 0 else 0.0
        return PrunedContext(
            context_string=context,
            item_mask=item_mask,
            modality_masks=modality_masks,
            retention_ratio=float(actual_retention),
            meta={
                "pruner": self.name,
                "target_retention": float(retention),
                "actual_retention": float(actual_retention),
                "seed": int(seed),
                "n_kept_patches": int(n_keep),
                "n_total_patches": int(total),
            },
        )
