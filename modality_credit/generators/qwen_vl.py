"""Qwen-2.5-VL-7B generator. Primary model for M3-Agent."""
from __future__ import annotations

from modality_credit.generators.base import BaseGenerator


SYSTEM_PROMPT = (
    "You are a careful assistant. Answer the user's question based on the "
    "provided memory items (images and text). Give your single best answer "
    "as a short word or phrase. Do not explain and do not refuse — always "
    "commit to a concrete answer even if you are uncertain."
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
                 images: list | None = None,
                 max_new_tokens: int = 128) -> str:
        body_text = (
            f"Memory:\n{context_str}\n\nQuestion: {query}" if context_str
            else f"Question: {query}\n\n(no memory was provided)"
        )
        # Build user content: prepend image blocks (one per attached image)
        # before the text body. Qwen's processor inserts <|vision_start|>
        # ... <|vision_end|> tokens automatically when content kind is "image".
        # We reference attached images via "[image #N attached]" markers in the
        # text; the model can correlate position by image-index order.
        if images:
            user_content = [{"type": "image", "image": img} for img in images]
            user_content.append({"type": "text", "text": body_text})
        else:
            user_content = body_text

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        if images:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
            )
        else:
            inputs = self._processor(text=[text], return_tensors="pt", padding=True)

        device = self._model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with self._torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic
            )
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
