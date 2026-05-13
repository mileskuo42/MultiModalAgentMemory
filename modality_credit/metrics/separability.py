"""Top-vs-bottom-quartile separability metric for Claim 1 detection.

Idea: rank all (item × modality) patches by attribution score across the
dataset, prune the top quartile vs the bottom quartile, compare accuracies.
A faithful attribution gives a large ΔAcc.
"""
from __future__ import annotations

import numpy as np

from modality_credit.protocols import Utility
from modality_credit.types import AttributionResult, QueryInstance


def top_bottom_quartile_delta_acc(utility: Utility,
                                  samples: list[QueryInstance],
                                  attributions: list[AttributionResult],
                                  ) -> dict[str, float]:
    """ΔAcc = accuracy(top-quartile-removed) − accuracy(bottom-quartile-removed).

    Returns:
        {"delta_acc": float, "acc_top_removed": float, "acc_bottom_removed": float}
    """
    # TODO(impl):
    # 1. For each (sample, attribution), build the "remove top-25% ψ patches" mask
    #    and the "remove bottom-25% ψ patches" mask.
    # 2. Call utility.evaluate under each mask, aggregate accuracy.
    # 3. Return delta + the two raw accuracies.
    raise NotImplementedError
