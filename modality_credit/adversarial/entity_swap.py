"""EntitySwap attack: rename one entity across all modalities of one item.

Example: "knife" → "spoon" in caption + audio transcript + scene metadata.
Vision is NOT modified, so this creates a deliberate vision/text disagreement
that our (item × modality) attribution should pinpoint to the text+audio side.
"""
from __future__ import annotations

import copy

from modality_credit.adversarial.base import BaseAttack
from modality_credit.types import Modality, QueryInstance


class EntitySwapAttack(BaseAttack):
    """Swap one entity name across all non-vision modalities of an item."""

    def __init__(self, swap_map: dict[str, str] | None = None):
        """swap_map: {original_word: replacement}. If None, use defaults."""
        self.swap_map = swap_map or {
            "knife": "spoon", "tomato": "potato", "cup": "bottle",
        }

    @property
    def affected_modality(self) -> Modality:
        # EntitySwap is the exception: hits multiple modalities. Default returns
        # "text"; `apply()` is overridden to hit text + audio + scene.
        return "text"

    def perturb_modality(self, target_modality: Modality, content: object) -> object:
        if not isinstance(content, str):
            return content
        out = content
        for k, v in self.swap_map.items():
            out = out.replace(k, v).replace(k.capitalize(), v.capitalize())
        return out

    def apply(self,
              query_inst: QueryInstance,
              target_item_idx: int,
              target_modality: Modality | None = None) -> QueryInstance:
        out = copy.deepcopy(query_inst)
        item = out.memory[target_item_idx]
        for m in ("text", "audio", "scene"):
            if m in item.modalities:
                item.modalities[m] = self.perturb_modality(m, item.modalities[m])
        out.metadata["attack"] = self.name
        out.metadata["attack_target"] = (target_item_idx, "text+audio+scene")
        return out
