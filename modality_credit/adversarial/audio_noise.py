"""AudioNoise attack: replace audio transcript with unrelated random text."""
from __future__ import annotations

from modality_credit.adversarial.base import BaseAttack
from modality_credit.types import Modality


class AudioNoiseAttack(BaseAttack):
    @property
    def affected_modality(self) -> Modality:
        return "audio"

    def perturb_modality(self, target_modality: Modality, content: object) -> object:
        # TODO(impl): sample a random transcript from a held-out corpus of
        # equivalent length, return it. Length-matching avoids trivial
        # detection by string length.
        raise NotImplementedError
