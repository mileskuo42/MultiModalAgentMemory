"""CaptionFlip attack: replace caption with semantically opposite text."""
from __future__ import annotations

from modality_credit.adversarial.base import BaseAttack
from modality_credit.types import Modality


class CaptionFlipAttack(BaseAttack):
    """Replace caption with a semantically opposite description (e.g., via LLM)."""

    @property
    def affected_modality(self) -> Modality:
        return "text"

    def perturb_modality(self, target_modality: Modality, content: object) -> object:
        # TODO(impl):
        # 1. Use a small LLM (GPT-4o-mini) prompted with:
        #    "Rewrite this caption to mean the opposite while keeping similar length:"
        # 2. Cache (content -> flipped) so repeated calls are deterministic.
        # 3. Fall back to simple negation if LLM unavailable.
        raise NotImplementedError
