"""LLM-judge verifier — for open-ended QA where exact match is too strict.

Two modes supported:
- Local: reuse a loaded Qwen-VL (or any HF causal LM) for semantic-
  equivalence judging. No API cost. Self-judging bias when using the
  same model that generated; acceptable for first pass.
- OpenAI: lazy client from OPENAI_API_KEY (not implemented yet here;
  the local path is what we use in practice).
"""
from __future__ import annotations

from typing import Any


class LLMJudgeVerifier:
    """Use a local LLM (Qwen-VL or similar) to judge semantic equivalence.

    Args:
        judge_generator: an object exposing `_model`, `_processor`, `_torch`.
                         In practice we pass the QwenVLGenerator instance
                         being used for memory inference, reusing the same
                         loaded model so we don't double GPU memory.

    The judge prompt is text-only (no images). The model is asked to reply
    with "yes" or "no" on whether `generated` is semantically equivalent
    to `gold` for the given query.

    Caches results on (generated_normalized, gold_normalized) so repeated
    judgments are free.
    """

    PROMPT = (
        "You are a strict grader. The user asked a question and got an "
        "answer. Decide whether the model's answer is semantically equivalent "
        "to the reference answer. Reply with ONLY 'yes' or 'no'.\n\n"
        "Question: {query}\n"
        "Reference answer: {gold}\n"
        "Model answer: {gen}\n\n"
        "Are these semantically equivalent? Reply ONLY 'yes' or 'no':"
    )

    def __init__(self, judge_generator: Any | None = None, max_new_tokens: int = 6):
        self._gen = judge_generator
        self._max_new = max_new_tokens
        self._cache: dict[tuple[str, str], bool] = {}
        self._n_calls = 0

    def attach(self, judge_generator) -> None:
        """Set the underlying generator after construction (e.g., after
        the QwenVLGenerator has been loaded)."""
        self._gen = judge_generator

    @property
    def n_calls(self) -> int:
        return self._n_calls

    def __call__(self, generated: str, gold: str, *, query: str = "") -> bool:
        if self._gen is None:
            raise RuntimeError("LLMJudgeVerifier needs a judge_generator attached")
        key = (str(generated).strip().lower(), str(gold).strip().lower())
        # Trivial fast path
        if key[0] == key[1]:
            return True
        if key in self._cache:
            return self._cache[key]
        prompt = self.PROMPT.format(query=query or "(question hidden)",
                                    gold=gold, gen=generated)
        # Build a TEXT-ONLY Qwen chat message; no system memory prompt
        messages = [
            {"role": "system",
             "content": "You are a strict yes/no grader of answer equivalence."},
            {"role": "user", "content": prompt},
        ]
        proc = self._gen._processor
        torch = self._gen._torch
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], return_tensors="pt", padding=True)
        device = self._gen._model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            out_ids = self._gen._model.generate(
                **inputs, max_new_tokens=self._max_new, do_sample=False,
            )
        prompt_len = inputs["input_ids"].shape[1]
        trimmed = out_ids[:, prompt_len:]
        reply = proc.batch_decode(trimmed, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0]
        reply_low = reply.strip().lower()
        # Parse: True iff reply starts with 'yes'
        result = reply_low.startswith("yes")
        self._cache[key] = result
        self._n_calls += 1
        return result
