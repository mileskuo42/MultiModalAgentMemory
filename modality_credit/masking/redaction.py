"""Redaction-style masking: replace masked-out content with neutral placeholder.

This is the DEFAULT strategy. Pros: preserves token-count and position; lower
spurious-pattern risk. Cons: placeholder string itself may leak (e.g., "[redacted]"
is a strong signal to the generator that something was here).

Leakage profile (must be validated by `audits.modality_leakage`):
  - Vision: black-frame replacement → near-zero residual visual info
  - Text:   "[caption redacted]" → string-level signal that a caption was hidden
  - Audio:  "[audio redacted]" → same
  - Scene:  "[scene metadata redacted]"
"""
from __future__ import annotations

from PIL import Image
import torch

from modality_credit.masking.base import BaseMasker
from modality_credit.types import ItemMask, MemoryItem, Modality, ModalityMask


class RedactionMasker(BaseMasker):
    """Default masker: replace masked modalities with [REDACTED] placeholders."""

    PLACEHOLDERS = {
        "text": "[caption redacted]",
        "audio": "[audio redacted]",
        "scene": "[scene metadata redacted]",
    }

    def apply(self,
              memory: list[MemoryItem],
              item_mask: ItemMask,
              modality_masks: list[ModalityMask] | None = None) -> str:
        blocks = []
        for k, item in enumerate(memory):
            if not item_mask[k]:
                continue
            kept = self._kept_modalities(item, modality_masks[k] if modality_masks else None)
            blocks.append(self._format_item(item, kept))
        return "\n\n".join(blocks)

    def _kept_modalities(self, item: MemoryItem,
                         mask: ModalityMask | None) -> dict[Modality, object]:
        out: dict[Modality, object] = {}
        for m, content in item.modalities.items():
            if mask is None or mask.get(m, True):
                out[m] = content
            else:
                out[m] = self._redact(m, content)
        return out

    def _redact(self, modality: Modality, content: object) -> object:
        if modality == "vision":
            return self._black_frame_like(content)
        return self.PLACEHOLDERS.get(modality, "[redacted]")

    @staticmethod
    def _black_frame_like(frame):
        if isinstance(frame, Image.Image):
            return Image.new("RGB", frame.size, (0, 0, 0))
        if isinstance(frame, torch.Tensor):
            return torch.zeros_like(frame)
        # TODO(impl): handle decord.VideoReader frames, raw numpy arrays
        raise TypeError(f"Unknown vision content type: {type(frame)}")
