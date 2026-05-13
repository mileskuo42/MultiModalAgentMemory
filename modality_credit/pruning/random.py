"""Random-drop pruner — critical control baseline for Claim 2.

We MUST significantly outperform random at matched retention or attribution
adds no value.
"""
from __future__ import annotations

import numpy as np

from modality_credit.pruning.base import BasePruner
from modality_credit.types import AttributionResult, ModalityMask, PrunedContext, QueryInstance


class RandomDropPruner(BasePruner):
    """Drop random (item, modality) patches until target retention is met."""

    def prune(self,
              query_inst: QueryInstance,
              attribution: AttributionResult,
              *,
              retention: float = 0.4,
              seed: int = 42) -> PrunedContext:
        rng = np.random.default_rng(seed)
        # TODO(impl):
        # 1. Enumerate all (k, ell) pairs.
        # 2. Sample floor(retention * total) of them uniformly.
        # 3. Build item_mask (item retained iff any of its modalities retained)
        #    and modality_masks accordingly.
        # 4. Materialize context via self._masker.
        raise NotImplementedError
