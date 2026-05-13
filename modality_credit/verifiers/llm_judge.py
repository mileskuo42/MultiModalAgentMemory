"""LLM-judge verifier — for open-ended QA where exact match is too strict.

Wraps a small LLM (default GPT-4o-mini) to score semantic equivalence.
Caching is recommended at the call-site since this adds API cost.
"""
from __future__ import annotations


class LLMJudgeVerifier:
    """Use an LLM to judge semantic equivalence.

    Args:
        model:     judge model name.
        client:    OpenAI-style client. If None, lazily construct one.
    """

    PROMPT = (
        "You are a strict grader. Decide whether the model's answer is "
        "semantically equivalent to the gold answer for the given question. "
        "Respond with ONLY 'yes' or 'no'.\n\n"
        "Gold answer: {gold}\nModel answer: {gen}"
    )

    def __init__(self, model: str = "gpt-4o-mini", client=None):
        self.model = model
        self._client = client

    def __call__(self, generated: str, gold: str) -> bool:
        # TODO(impl):
        # 1. lazily init self._client from OPENAI_API_KEY env
        # 2. call chat.completions.create with temperature=0, max_tokens=4
        # 3. parse response → bool
        raise NotImplementedError
