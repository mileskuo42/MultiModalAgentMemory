"""Masker base class. Implementations: redaction (default), removal, black_frame."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from modality_credit.types import ItemMask, MemoryItem, ModalityMask


class BaseMasker(ABC):
    """Shared helpers; concrete subclasses implement `apply`."""

    @abstractmethod
    def apply(self,
              memory: list[MemoryItem],
              item_mask: ItemMask,
              modality_masks: list[ModalityMask] | None = None) -> tuple[str, list[Any]]:
        """Build the (context_str, images) pair fed to the generator.

        Returns:
            (context_str, images):
              context_str — text rendered from the masked memory.
              images      — PIL.Image objects (or other vision content), in the
                            same order as their `[image #N attached]` markers
                            appear in context_str.

        Postcondition: items with `item_mask[k] = False` MUST NOT influence
        the generator's output via any channel (token, position, image
        attachment). Implementations document their leakage profile.
        """
        ...

    def _format_item_with_images(self, item: MemoryItem, kept_mods: dict,
                                 image_offset: int = 0) -> tuple[str, list[Any]]:
        """Default item-block formatter. Returns (text_block, images).

        For vision modality:
          - If content is a PIL.Image / tensor / non-str object: emit
            "[image #N attached]" marker in text, collect the image.
          - If content is a string (synthetic/stub path): emit the legacy
            "[image] (attached frame for ...)" placeholder, no image collected.
        """
        parts = [f"=== Memory item {item.item_id} ==="]
        images: list[Any] = []
        for m, content in kept_mods.items():
            if m == "vision":
                if isinstance(content, str):
                    parts.append(f"[image] (attached frame for item {item.item_id})")
                else:
                    parts.append(f"[image #{image_offset + len(images) + 1} attached]")
                    images.append(content)
            elif m == "text":
                parts.append(f"[caption] {content}")
            elif m == "audio":
                parts.append(f"[audio transcript] {content}")
            elif m == "scene":
                parts.append(f"[scene] {content}")
        return "\n".join(parts), images

    def _format_item(self, item: MemoryItem, kept_mods: dict) -> str:
        """Legacy text-only formatter; subclasses may keep using when no image
        routing is needed."""
        text, _ = self._format_item_with_images(item, kept_mods)
        return text
