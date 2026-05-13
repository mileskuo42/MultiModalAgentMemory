"""FrameSwap attack: replace vision frame with a random unrelated frame."""
from __future__ import annotations

from modality_credit.adversarial.base import BaseAttack
from modality_credit.types import Modality


class FrameSwapAttack(BaseAttack):
    def __init__(self, donor_frames_path: str):
        """donor_frames_path: directory of held-out frames to sample from."""
        self.donor_path = donor_frames_path
        self._donor_cache = None

    @property
    def affected_modality(self) -> Modality:
        return "vision"

    def perturb_modality(self, target_modality: Modality, content: object) -> object:
        # TODO(impl):
        # 1. Lazy-load donor frames index from self.donor_path.
        # 2. Sample one frame (deterministically per attack instance for repro).
        # 3. Resize to match original `content.size`; return PIL.Image.
        raise NotImplementedError
