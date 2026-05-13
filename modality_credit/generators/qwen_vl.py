"""Qwen-2.5-VL-7B generator. Primary model for M3-Agent."""
from __future__ import annotations

from modality_credit.generators.base import BaseGenerator


class QwenVLGenerator(BaseGenerator):
    """Wrap Qwen-2.5-VL-7B-Instruct.

    Args:
        model_path:    local path or HF repo id.
        device:        "cuda" / "cpu" / explicit ordinal.
        dtype:         torch dtype.
        deterministic: set torch.use_deterministic_algorithms and fix seed.
    """

    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 device: str = "cuda", dtype=None,
                 deterministic: bool = True, seed: int = 42):
        # TODO(impl):
        # 1. Load Qwen2_5_VLForConditionalGeneration + AutoProcessor
        # 2. .eval(), .to(device, dtype)
        # 3. If deterministic: torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
        # 4. Store model, processor, device for use in generate()
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.seed = seed
        self._model = None
        self._processor = None
        raise NotImplementedError

    def generate(self, query: str, context_str: str, *,
                 max_new_tokens: int = 128) -> str:
        # TODO(impl):
        # 1. Build conversation: system + context_str (text + image attachments) + query
        # 2. processor.apply_chat_template(...) → inputs
        # 3. model.generate(..., do_sample=False, max_new_tokens=max_new_tokens)
        # 4. Decode and strip prompt tokens
        raise NotImplementedError

    def get_attention(self, query: str, context_str: str):
        # TODO(impl): forward with output_attentions=True; aggregate per (item, modality) span
        raise NotImplementedError

    @property
    def name(self) -> str:
        return "qwen-vl-7b"
