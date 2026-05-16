"""Smoke test the LLMJudgeVerifier on hand-crafted (gen, gold) pairs.

Verifies the judge actually distinguishes semantically equivalent answers
from non-equivalent ones, before we commit GPU time to the full R19 run.
"""
from __future__ import annotations

import time
import torch
from modality_credit.generators.qwen_vl import QwenVLGenerator
from modality_credit.verifiers.llm_judge import LLMJudgeVerifier


CASES = [
    # (query, gold, gen, expected_yes)
    ("What is the color of the hat?", "blue", "Blue", True),
    ("What is the color of the hat?", "blue", "The hat is blue.", True),
    ("What is the color of the hat?", "blue", "red", False),
    ("What is the dog's name?", "Buster", "The dog is called Buster.", True),
    ("What is the dog's name?", "Buster", "Max", False),
    ("Does the man ride a horse?", "no", "no, he is walking", True),
    ("Does the man ride a horse?", "no", "yes, he is riding", False),
    ("How much?", "$12500", "12,500 dollars", True),
    ("How much?", "$12500", "$15000", False),
    ("When is the meeting?", "9:30 AM", "9:30 in the morning", True),
]

print("[1/3] Loading Qwen-2.5-VL-7B on GPU (bf16)...")
t0 = time.time()
gen = QwenVLGenerator(device="cuda", dtype=torch.bfloat16)
print(f"      loaded in {time.time()-t0:.1f}s")

judge = LLMJudgeVerifier(judge_generator=gen)
print(f"\n[2/3] Smoke-testing judge on {len(CASES)} hand-crafted pairs...")
correct = 0
for i, (q, gold, gen_ans, expected) in enumerate(CASES):
    t = time.time()
    pred = judge(gen_ans, gold, query=q)
    ok = "✓" if pred == expected else "✗"
    if pred == expected:
        correct += 1
    print(f"  [{i+1}/{len(CASES)}] {ok} Q={q[:35]:<35} gold={gold!r:<12} gen={gen_ans!r:<35} pred={pred} (expected {expected}) {time.time()-t:.1f}s")

print(f"\n[3/3] Judge accuracy on smoke set: {correct}/{len(CASES)} = {correct/len(CASES):.0%}")
print(f"      (target: ≥ 80% to be useful as Phase 4A verifier)")
