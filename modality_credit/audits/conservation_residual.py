"""Audit — Verify Owen conservation Σ φ = U(M) - U(∅) holds in expectation.

This is technically embedded in OwenEstimator output, but having it as a
standalone audit lets Phase 1 fail-fast without running full attribution.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from modality_credit.audits.base import BaseAudit
from modality_credit.estimator.owen import OwenEstimator
from modality_credit.protocols import Utility
from modality_credit.types import QueryInstance


class ConservationResidualAudit(BaseAudit):

    def __init__(self, budget_B: int = 200, seed: int = 42):
        self.budget_B = budget_B
        self.seed = seed

    def run(self, utility: Utility, samples: list[QueryInstance]) -> dict[str, Any]:
        est = OwenEstimator(seed=self.seed)
        residuals = []
        for inst in samples:
            attr = est.estimate(utility, inst, budget_B=self.budget_B)
            residuals.append(attr.conservation_residual)
        residuals = np.array(residuals)
        mean = float(residuals.mean())

        if mean < 0.05:
            verdict, summary = self.VERDICT_PASS, f"Mean residual {mean:.2%} < 5%"
        elif mean < 0.10:
            verdict, summary = self.VERDICT_WARN, f"Mean residual {mean:.2%}"
        else:
            verdict, summary = self.VERDICT_FAIL, f"Mean residual {mean:.2%} > 10% — check normalization"

        return {"verdict": verdict, "summary": summary,
                "data": {"residuals": residuals.tolist(), "mean": mean}}
