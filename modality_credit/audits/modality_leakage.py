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
        u_full_per_inst: dict[str, float] = {}
        for inst in samples:
            K = inst.K
            # full-memory baseline
            u_full = utility.evaluate(inst, [True] * K)
            u_full_per_inst[inst.instance_id] = u_full

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

        # ---- aggregation
        # per-modality mean U across all (sample, item) pairs
        per_modality: dict[str, list[float]] = {}
        for r in rows:
            per_modality.setdefault(r["keep"], []).append(r["u"])
        mean_per_modality = {m: float(np.mean(vs)) for m, vs in per_modality.items()}

        # sum-of-single-mod / full ratio (proxy for redundancy)
        # For each sample, find the decisive item and its single-mod U values.
        ratios = []
        for iid, u_full in u_full_per_inst.items():
            if u_full < 0.01:
                continue  # skip samples where model fails even with full memory
            mods_seen = set()
            sample_rows = [r for r in rows if r["iid"] == iid]
            # Aggregate over (item, modality): mean single-modality U for each modality
            per_mod_max = {}
            for r in sample_rows:
                per_mod_max[r["keep"]] = max(per_mod_max.get(r["keep"], 0.0), r["u"])
                mods_seen.add(r["keep"])
            ratios.append(sum(per_mod_max.values()) / u_full)

        mean_ratio = float(np.mean(ratios)) if ratios else float("nan")

        # ---- verdict
        # Heuristic pass criteria from method_plan §7.1:
        #   - At least 2 modalities each show mean U ≥ 0.20 (each modality
        #     carries non-trivial info on at least some samples)
        #   - sum-of-single-mod / full ratio in [1.2, 3.0]
        non_trivial_mods = sum(1 for m, u in mean_per_modality.items() if u >= 0.20)
        ratio_ok = 1.2 <= mean_ratio <= 3.0 if not np.isnan(mean_ratio) else False

        if non_trivial_mods >= 2 and ratio_ok:
            verdict, summary = self.VERDICT_PASS, (
                f"{non_trivial_mods} modalities ≥ 0.20; sum/full ratio {mean_ratio:.2f} in [1.2, 3.0]"
            )
        elif non_trivial_mods >= 2 or (not np.isnan(mean_ratio) and 0.9 <= mean_ratio <= 4.0):
            verdict, summary = self.VERDICT_WARN, (
                f"{non_trivial_mods} modalities ≥ 0.20; sum/full ratio {mean_ratio:.2f}"
            )
        else:
            verdict, summary = self.VERDICT_FAIL, (
                f"only {non_trivial_mods} modalities ≥ 0.20; sum/full ratio {mean_ratio:.2f}"
            )

        return {
            "verdict": verdict,
            "summary": summary,
            "data": {
                "rows": rows,
                "mean_per_modality": mean_per_modality,
                "mean_sum_over_full_ratio": mean_ratio,
                "non_trivial_modalities_count": int(non_trivial_mods),
                "n_samples": len(samples),
            },
        }
