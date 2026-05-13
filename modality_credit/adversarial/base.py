"""Adversarial attack base — for Phase 4A memory poisoning detection."""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod

from modality_credit.types import Modality, QueryInstance


class BaseAttack(ABC):
    """Adversarial perturbation injected into one (item, modality) location."""

    @abstractmethod
    def perturb_modality(self, target_modality: Modality, content: object) -> object:
        """Return adversarial content for the chosen modality."""
        ...

    @property
    @abstractmethod
    def affected_modality(self) -> Modality:
        """Which modality this attack tampers with."""
        ...

    def apply(self,
              query_inst: QueryInstance,
              target_item_idx: int,
              target_modality: Modality | None = None) -> QueryInstance:
        """Default implementation: deep-copy and substitute one modality."""
        if target_modality is None:
            target_modality = self.affected_modality
        out = copy.deepcopy(query_inst)
        item = out.memory[target_item_idx]
        item.modalities[target_modality] = self.perturb_modality(
            target_modality, item.modalities[target_modality]
        )
        out.metadata["attack"] = self.name
        out.metadata["attack_target"] = (target_item_idx, target_modality)
        return out

    @property
    def name(self) -> str:
        return type(self).__name__
