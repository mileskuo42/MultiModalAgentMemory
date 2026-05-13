"""Phase 4A — poisoning detection AUROC.

Signal = modality-inconsistency + φ-outlier-score derived from Owen attribution.
"""
from __future__ import annotations

import numpy as np

from modality_credit.types import AttributionResult


def modality_inconsistency_score(psi_for_item: np.ndarray) -> float:
    """High value = modalities of this item disagree (one positive, one negative).

    Computed as the std deviation of ψ values for the item, ignoring NaN.
    """
    valid = psi_for_item[~np.isnan(psi_for_item)]
    if len(valid) < 2:
        return 0.0
    return float(np.std(valid))


def phi_outlier_score(phi: np.ndarray, item_idx: int) -> float:
    """Z-score of φ_k relative to the other items."""
    others = np.delete(phi, item_idx)
    if len(others) < 2:
        return 0.0
    mu, sd = float(np.mean(others)), float(np.std(others))
    if sd < 1e-9:
        return 0.0
    return float(abs(phi[item_idx] - mu) / sd)


def detection_signal(attribution: AttributionResult, item_idx: int) -> float:
    """Combined signal: ψ-inconsistency + φ-outlier. Higher = more suspect."""
    return (modality_inconsistency_score(attribution.psi[item_idx])
            + phi_outlier_score(attribution.phi, item_idx))


def compute_auroc(scores: list[float], labels: list[int]) -> float:
    """Standard AUROC. Wrapper around sklearn for testability."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))
