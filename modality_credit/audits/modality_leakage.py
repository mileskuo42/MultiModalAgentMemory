"""Audit 7.1 — verify modality masking actually removes information.

Test: for each (sample × item × keep_modality), compute single-modality utility.
Pass criteria (see method_plan §7.1):
  - vision-only U ≥ 0.20
  - text-only U ≥ 0.20
  - sum of single-mod U / full U  in [1.2, 3.0]   (some redundancy, not full)
"""
from __future__ import annotations

from typing import Any

import numpy as np

from modality_credit.audits.base import BaseAudit
from modality_credit.protocols import Utility
from modality_credit.types import QueryInstance


class ModalityLeakageAudit(BaseAudit):

    def run(self, utility: Utility, samples: list[QueryInstance]) -> dict[str, Any]:
        rows = []
        for inst in samples:
            K = inst.K
            for k in range(K):
                modalities = list(inst.memory[k].modalities.keys())
                for keep in modalities:
                    item_mask = [i == k for i in range(K)]
                    modality_masks = [
                        {m: True for m in inst.memory[j].modalities}
                        for j in range(K)
                    ]
                    modality_masks[k] = {m: (m == keep) for m in modalities}
                    u = utility.evaluate(inst, item_mask, modality_masks)
                    rows.append({"iid": inst.instance_id, "k": k, "keep": keep, "u": u})

        # TODO(impl):
        # 1. Aggregate to per-modality mean utility.
        # 2. Compute sum/full ratio per sample.
        # 3. Apply pass criteria (above); set verdict.
        # 4. Return data as a DataFrame or list of rows for the notebook.
        raise NotImplementedError
