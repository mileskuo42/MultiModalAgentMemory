"""Shared dataclasses and type aliases.

This module defines the data model the entire framework agrees on.
DO NOT extend with behavior — only data shapes. Behavior lives in protocols/ and
in concrete implementations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Modality: TypeAlias = Literal["vision", "text", "audio", "scene"]
"""The four modalities we model. Order matters for tensor indexing — never
re-order; add new modalities at the end."""

ItemMask: TypeAlias = list[bool]
"""Length-K mask over memory items. True = include in context."""

ModalityMask: TypeAlias = dict[Modality, bool]
"""Per-item mask over its modalities. True = include this modality."""

PhiArray: TypeAlias = NDArray[np.float64]  # shape (K,)
"""Outer-game (item-level) Shapley values."""

PsiArray: TypeAlias = NDArray[np.float64]  # shape (K, max_L)
"""Inner-game (modality-level) Shapley values, padded with NaN for items
not in top-N."""


# ---------------------------------------------------------------------------
# Memory + Query
# ---------------------------------------------------------------------------

@dataclass(frozen=False, slots=True)
class MemoryItem:
    """One retrieved episode.

    Invariants:
      - `item_id` is unique within a `QueryInstance`.
      - `modalities` contains at least one key, all from the `Modality` literal.
      - Each modality's content type is implementation-defined (PIL.Image for
        vision, str for text/audio/scene, etc.) but must round-trip through
        the configured Masker.
    """
    item_id: str
    modalities: dict[Modality, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_modalities(self) -> int:
        return len(self.modalities)


@dataclass(frozen=False, slots=True)
class QueryInstance:
    """One (Q, M, y*) tuple — the unit of attribution.

    Invariants:
      - `memory` is non-empty; `K = len(memory)`.
      - `gold_answer` is the reference for `Verifier`.
      - `instance_id` is unique within a Dataset.
    """
    instance_id: str
    query: str
    memory: list[MemoryItem]
    gold_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def K(self) -> int:
        return len(self.memory)

    @property
    def L_ks(self) -> list[int]:
        return [m.num_modalities for m in self.memory]


# ---------------------------------------------------------------------------
# Attribution result
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AttributionResult:
    """Output of any Estimator.estimate(query_inst).

    `phi[k]`           = item-level Shapley value (Owen between-group).
    `psi[k, ell_idx]`  = modality-level Shapley value within item k.
                          NaN for items not in top-N (when applicable).
    `top_idx`          = indices of items selected for inner-game estimation.
    `conservation_residual` = |Σ φ_k − (U(M) − U(∅))| / |U(M) − U(∅)|.
    `meta`             = sampler/estimator-specific diagnostics (sample count,
                          wall-clock, allocation policy used, etc.).
    """
    phi: PhiArray
    psi: PsiArray
    top_idx: NDArray[np.int_]
    conservation_residual: float
    U_full: float
    U_empty: float
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pruning result
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PrunedContext:
    """Output of any Pruner.prune(query_inst, attribution)."""
    context_string: str
    item_mask: ItemMask
    modality_masks: list[ModalityMask]
    retention_ratio: float  # kept_patches / total_patches
    meta: dict[str, Any] = field(default_factory=dict)
