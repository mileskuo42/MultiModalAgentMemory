"""Generator base — wrap any multimodal LLM behind a uniform interface."""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    """ABC for multimodal LLM generators (Qwen-VL, InternVL, LLaVA, ...)."""

    @abstractmethod
    def generate(self, query: str, context_str: str, *,
                 max_new_tokens: int = 128) -> str:
        ...

    def get_attention(self, query: str, context_str: str):
        """Optional: return per-token attention for KSHAP seeding.
        Subclasses without this capability should raise NotImplementedError."""
        raise NotImplementedError(f"{self.name} does not expose attention logits")

    @property
    def name(self) -> str:
        return type(self).__name__
