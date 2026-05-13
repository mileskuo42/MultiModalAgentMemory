"""Audit 7.2 — measure U(∅) distribution.

If U(∅) is high, conservation is dominated by query-only signal; we must
normalize u_norm(S) = (U(S) - U(∅)) / (U(M) - U(∅)).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from modality_credit.audits.base import BaseAudit
from modality_credit.protocols import Utility
from modality_credit.types import QueryInstance


class UEmptyBaselineAudit(BaseAudit):

    def run(self, utility: Utility, samples: list[QueryInstance]) -> dict[str, Any]:
        u_empties = []
        for inst in samples:
            u = utility.evaluate(inst, [False] * inst.K)
            u_empties.append(u)
        u_empties = np.array(u_empties)
        mean = float(u_empties.mean())

        if mean < 0.2:
            verdict, summary = self.VERDICT_PASS, "Low U(∅); no normalization needed"
        elif mean < 0.4:
            verdict, summary = self.VERDICT_WARN, "Moderate U(∅); enable u_norm in estimator"
        else:
            verdict, summary = self.VERDICT_FAIL, "High U(∅); query-only baseline too strong, reconsider benchmark"

        return {
            "verdict": verdict,
            "summary": summary,
            "data": {
                "mean": mean,
                "p25": float(np.percentile(u_empties, 25)),
                "p75": float(np.percentile(u_empties, 75)),
                "p_high": float((u_empties > 0.5).mean()),
                "values": u_empties.tolist(),
            },
        }
