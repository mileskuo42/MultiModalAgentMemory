"""Masker base class. Implementations: redaction (default), removal, black_frame."""
from __future__ import annotations

from abc import ABC, abstractmethod

from modality_credit.types import ItemMask, MemoryItem, ModalityMask


class BaseMasker(ABC):
    """Shared helpers; concrete subclasses implement `apply`."""

    @abstractmethod
    def apply(self,
              memory: list[MemoryItem],
              item_mask: ItemMask,
              modality_masks: list[ModalityMask] | None = None) -> str:
        """Build the context string fed to the generator.

        Postcondition: items with `item_mask[k] = False` MUST NOT influence the
        generator's output via any channel (token, position, image attachment).
        Implementations document their leakage profile in the class docstring.
        """
        ...

    def _format_item(self, item: MemoryItem, kept_mods: dict) -> str:
        """Default item-block formatter. Subclasses may override."""
        parts = [f"=== Memory item {item.item_id} ==="]
        for m, content in kept_mods.items():
            if m == "vision":
                parts.append(f"[image] (attached frame for item {item.item_id})")
            elif m == "text":
                parts.append(f"[caption] {content}")
            elif m == "audio":
                parts.append(f"[audio transcript] {content}")
            elif m == "scene":
                parts.append(f"[scene] {content}")
        return "\n".join(parts)
