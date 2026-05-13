"""Pareto curve for modality-pruned retrieval (Claim 2 — must-accept hook)."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from modality_credit.protocols import Pruner, Utility
from modality_credit.types import AttributionResult, QueryInstance


def evaluate_pareto(utility: Utility,
                    samples: list[QueryInstance],
                    attributions: list[AttributionResult],
                    pruner: Pruner,
                    tau_grid: Iterable[tuple[float, float]],
                    ) -> pd.DataFrame:
    """Sweep (τ_1, τ_2) and record (retention, accuracy) for each setting.

    Returns: DataFrame with columns [tau_1, tau_2, retention, accuracy, n_samples].
    """
    rows = []
    for tau_1, tau_2 in tau_grid:
        retentions, accs = [], []
        for inst, attr in zip(samples, attributions):
            pruned = pruner.prune(inst, attr, tau_1=tau_1, tau_2=tau_2)
            retentions.append(pruned.retention_ratio)
            acc = utility.evaluate(inst, pruned.item_mask, pruned.modality_masks)
            accs.append(acc)
        rows.append({
            "tau_1": tau_1, "tau_2": tau_2,
            "retention": float(np.mean(retentions)),
            "accuracy": float(np.mean(accs)),
            "n_samples": len(samples),
        })
    return pd.DataFrame(rows)


def find_operating_point(pareto_df: pd.DataFrame,
                         target_retention: float = 0.4) -> dict:
    """Return the (τ_1, τ_2) that achieves retention closest to target."""
    idx = (pareto_df["retention"] - target_retention).abs().idxmin()
    return pareto_df.iloc[idx].to_dict()
