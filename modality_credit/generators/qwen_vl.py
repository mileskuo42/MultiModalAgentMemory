"""Qwen-2.5-VL-7B generator. Primary model for M3-Agent."""
from __future__ import annotations

from modality_credit.generators.base import BaseGenerator


SYSTEM_PROMPT = (
    "You are a careful assistant. You will be given a set of memory items "
    "retrieved for a user query. Answer the query using ONLY information from "
    "the provided memory. If the memory is insufficient, respond with 'unknown'. "
    "Keep your answer concise — a single phrase if possible."
)


class QwenVLGenerator(BaseGenerator):
    """Wrap Qwen-2.5-VL-7B-Instruct.

    Current implementation is **text-only**: the upstream `RedactionMasker.apply()`
    formats every retrieved item into a single context string with vision-modality
    rendered as a text placeholder ("[image] (attached frame for item ...)"). To
    actually feed PIL images into the model, the `Generator` protocol needs to be
    extended to accept image content alongside `context_str`. See issue tracker.

    Args:
        model_path:    local path or HF repo id.
        device:        "cuda" / "cpu" / explicit ordinal ("cuda:1") / "auto".
        dtype:         torch dtype (None → "auto").
        deterministic: greedy decoding + fixed seed.
        seed:          torch random seed.
    """

    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 device: str = "cuda", dtype=None,
                 deterministic: bool = True, seed: int = 42):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.seed = seed
        self.deterministic = deterministic

        if deterministic:
            torch.manual_seed(seed)

        kwargs = {}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        else:
            kwargs["torch_dtype"] = "auto"

        if device == "auto":
            kwargs["device_map"] = "auto"
            self._target_device = None
        else:
            self._target_device = device

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, **kwargs
        )
        if self._target_device is not None:
            self._model = self._model.to(self._target_device)
        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(model_path)
        self._torch = torch

    def generate(self, query: str, context_str: str, *,
                 max_new_tokens: int = 128) -> str:
        user_content = (
            f"Memory:\n{context_str}\n\nQuestion: {query}" if context_str
            else f"Question: {query}\n\n(no memory was provided)"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        device = self._model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with self._torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic
            )
        # Strip the prompt prefix.
        prompt_len = inputs["input_ids"].shape[1]
        trimmed = generated_ids[:, prompt_len:]
        out = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return out.strip()

    def get_attention(self, query: str, context_str: str):
        # TODO(impl): forward with output_attentions=True; aggregate per (item, modality) span
        raise NotImplementedError("Attention extraction not yet implemented")

    @property
    def name(self) -> str:
        return "qwen-vl-7b"
