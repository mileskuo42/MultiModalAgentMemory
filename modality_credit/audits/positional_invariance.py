"""Audit 7.3 — check generator output stability under item-order permutation."""
from __future__ import annotations

import copy
from typing import Any

import numpy as np

from modality_credit.audits.base import BaseAudit
from modality_credit.protocols import Utility
from modality_credit.types import QueryInstance


class PositionalInvarianceAudit(BaseAudit):

    def __init__(self, n_shuffles: int = 5, seed: int = 42):
        self.n_shuffles = n_shuffles
        self.seed = seed

    def run(self, utility: Utility, samples: list[QueryInstance]) -> dict[str, Any]:
        rng = np.random.default_rng(self.seed)
        rows = []
        for inst in samples:
            K = inst.K
            u_values = []
            for _ in range(self.n_shuffles):
                perm = rng.permutation(K)
                shuffled = copy.copy(inst)
                shuffled.memory = [inst.memory[i] for i in perm]
                u_values.append(utility.evaluate(shuffled, [True] * K))
            rows.append({"iid": inst.instance_id,
                         "std": float(np.std(u_values)),
                         "mean": float(np.mean(u_values))})

        stds = np.array([r["std"] for r in rows])
        means = np.array([r["mean"] for r in rows])
        avg_ratio = float((stds / np.maximum(means, 1e-6)).mean())

        if avg_ratio < 0.05:
            verdict, summary = self.VERDICT_PASS, f"Permutation residual {avg_ratio:.1%} < 5%"
        elif avg_ratio < 0.10:
            verdict, summary = self.VERDICT_WARN, f"Permutation residual {avg_ratio:.1%} marginal"
        else:
            verdict, summary = self.VERDICT_FAIL, f"Permutation residual {avg_ratio:.1%} too high; must randomize position in estimator"

        return {"verdict": verdict, "summary": summary, "data": {"rows": rows, "avg_ratio": avg_ratio}}
