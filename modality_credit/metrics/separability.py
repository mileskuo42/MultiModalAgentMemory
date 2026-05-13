"""Top-vs-bottom-quartile separability metric for Claim 1 detection.

Idea: rank all (item × modality) patches by attribution score across the
dataset, prune the top quartile vs the bottom quartile, compare accuracies.
A faithful attribution gives a large ΔAcc.
"""
from __future__ import annotations

import numpy as np

from modality_credit.protocols import Utility
from modality_credit.types import AttributionResult, QueryInstance


def _patch_scores(sample: QueryInstance, attr: AttributionResult) -> list[tuple[int, str, float]]:
    """Return list of (item_idx, modality_name, score) for all patches.

    Score = ψ_{k, ell} if available (not NaN), else fall back to φ_k / L_k.
    This handles Owen attribution where ψ is only filled for top_n_inner
    items; the fallback gives each modality in remaining items its uniform
    share of the item-level Shapley value.
    """
    out: list[tuple[int, str, float]] = []
    for k, item in enumerate(sample.memory):
        L_k = len(item.modalities)
        modalities = list(item.modalities.keys())
        psi_k = attr.psi[k, :L_k]
        if not np.all(np.isnan(psi_k)):
            for ell_idx, m in enumerate(modalities):
                psi_val = psi_k[ell_idx]
                score = float(psi_val) if not np.isnan(psi_val) else float(attr.phi[k]) / L_k
                out.append((k, m, score))
        else:
            score_per_mod = float(attr.phi[k]) / L_k
            for m in modalities:
                out.append((k, m, score_per_mod))
    return out


def _build_quartile_mask(sample: QueryInstance, patches_to_remove: list[tuple[int, str, float]]
                        ) -> tuple[list[bool], list[dict[str, bool]]]:
    K = sample.K
    item_mask = [True] * K
    modality_masks: list[dict[str, bool]] = [
        {m: True for m in item.modalities} for item in sample.memory
    ]
    for k, m, _ in patches_to_remove:
        modality_masks[k][m] = False  # type: ignore[index]
    # If an item has no modalities kept, drop it entirely
    for k in range(K):
        if not any(modality_masks[k].values()):
            item_mask[k] = False
    return item_mask, modality_masks


def top_bottom_quartile_delta_acc(utility: Utility,
                                  samples: list[QueryInstance],
                                  attributions: list[AttributionResult],
                                  ) -> dict[str, float]:
    """ΔAcc = acc(top-quartile-removed) − acc(bottom-quartile-removed).

    A faithful attribution should give a LARGE NEGATIVE ΔAcc (top quartile
    matters more than bottom). Equivalently, by convention we usually report
    `acc_bottom_removed - acc_top_removed`, which is positive for good
    attributions. We follow the convention: report
        delta_acc = acc(bottom-removed) - acc(top-removed)   (≥ 0 expected)

    Returns:
        {"delta_acc": ..., "acc_top_removed": ..., "acc_bottom_removed": ...,
         "n_samples": ...}
    """
    accs_top: list[float] = []
    accs_bottom: list[float] = []

    for sample, attr in zip(samples, attributions):
        patches = _patch_scores(sample, attr)
        if not patches:
            continue
        patches.sort(key=lambda x: x[2], reverse=True)
        n = len(patches)
        n_q = max(n // 4, 1)
        top_q = patches[:n_q]
        bottom_q = patches[-n_q:]

        item_mask_t, mod_mask_t = _build_quartile_mask(sample, top_q)
        item_mask_b, mod_mask_b = _build_quartile_mask(sample, bottom_q)
        accs_top.append(utility.evaluate(sample, item_mask_t, mod_mask_t))
        accs_bottom.append(utility.evaluate(sample, item_mask_b, mod_mask_b))

    acc_top_mean = float(np.mean(accs_top)) if accs_top else float("nan")
    acc_bot_mean = float(np.mean(accs_bottom)) if accs_bottom else float("nan")
    return {
        "delta_acc": acc_bot_mean - acc_top_mean,
        "acc_top_removed": acc_top_mean,
        "acc_bottom_removed": acc_bot_mean,
        "n_samples": len(accs_top),
    }
