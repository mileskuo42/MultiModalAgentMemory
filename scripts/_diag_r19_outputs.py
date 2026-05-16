"""Diagnostic: dump actual Qwen-VL outputs vs gold for R19 failed samples.

Determines whether U=0 is due to (a) judge being too strict on a correct
answer, or (b) model genuinely outputting a wrong answer.
"""
import json
import random
import time
import torch
from datasets import load_dataset
from modality_credit.generators.qwen_vl import QwenVLGenerator
from modality_credit.masking.redaction import RedactionMasker
from modality_credit.verifiers.llm_judge import LLMJudgeVerifier
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from r19_real_poisoning import filter_simple_samples, build_sample, _resize

# Reproduce R19's sample list (seed=42)
ds = load_dataset("lmms-lab/MMVet", split="test")
pool = filter_simple_samples(ds, max_answer_len=25)
rng = random.Random(42)
rng.shuffle(pool)
chosen = pool[:20]
samples = []
for i, mmvet in enumerate(chosen):
    is_poisoned = i >= 10
    samples.append(build_sample(i, mmvet, pool, is_poisoned, rng, K=4))

print(f"[1/3] Loading Qwen-VL...")
t0 = time.time()
gen = QwenVLGenerator(device="cuda", dtype=torch.bfloat16)
judge = LLMJudgeVerifier(judge_generator=gen)
print(f"      loaded in {time.time()-t0:.1f}s")
masker = RedactionMasker()

print(f"\n[2/3] Running full-memory generation on all 20 samples (no Owen — just U(M))...")
results = []
for i, qi in enumerate(samples):
    ctx, imgs = masker.apply(qi.memory, item_mask=[True] * qi.K)
    t = time.time()
    y = gen.generate(qi.query, ctx, images=imgs)
    judge_yes = judge(y, qi.gold_answer, query=qi.query)
    results.append({
        "idx": i,
        "is_poisoned": qi.metadata["is_poisoned"],
        "query": qi.query,
        "gold": qi.gold_answer,
        "gen": y,
        "judge_says_match": judge_yes,
        "wall": time.time() - t,
    })
    flag = "POISON" if qi.metadata["is_poisoned"] else "clean "
    print(f"  [{i+1}/20] {flag} judge={judge_yes!s:<5}  gold={qi.gold_answer!r:<22} gen={y!r}")

# Aggregate
clean = [r for r in results if not r["is_poisoned"]]
poison = [r for r in results if r["is_poisoned"]]
print(f"\n[3/3] === SUMMARY ===")
print(f"  Clean samples (n={len(clean)}):")
print(f"    judge said YES: {sum(r['judge_says_match'] for r in clean)}/{len(clean)}")
print(f"  Poisoned samples (n={len(poison)}):")
print(f"    judge said YES: {sum(r['judge_says_match'] for r in poison)}/{len(poison)}")

# For all the NO-match cases: is the model's answer clearly wrong, or is it close?
print(f"\n  NO-match cases — model output vs gold (you decide if judge was right):")
for r in results:
    if not r["judge_says_match"]:
        flag = "P" if r["is_poisoned"] else "C"
        print(f"    {flag} gold={r['gold']!r:<22} gen={r['gen'][:80]!r}")
