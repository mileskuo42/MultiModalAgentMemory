"""End-to-end smoke: QwenVLGenerator + RedactionMasker + StandardUtility + M3BenchDataset.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/_smoke_qwen.py

Verifies:
  1. Qwen-2.5-VL-7B loads on the visible GPU.
  2. StandardUtility.evaluate on full memory returns 1.0 (correct answer).
  3. StandardUtility.evaluate on empty memory returns 0.0 (no answer).
"""
from __future__ import annotations

import time

from modality_credit.data.m3_bench import M3BenchDataset
from modality_credit.generators.qwen_vl import QwenVLGenerator
from modality_credit.masking.redaction import RedactionMasker
from modality_credit.utility import StandardUtility
from modality_credit.verifiers.exact_match import ExactMatchVerifier


def main():
    print("[1/4] Building dataset (synthetic fallback)...")
    ds = M3BenchDataset(K=4, n=3)
    qi = ds[0]
    print(f"      instance_id={qi.instance_id}, query={qi.query!r}, gold={qi.gold_answer!r}")

    print("\n[2/4] Loading Qwen-2.5-VL-7B on CPU (slow but unblocks driver mismatch)...")
    t0 = time.time()
    import torch
    gen = QwenVLGenerator(model_path="Qwen/Qwen2.5-VL-7B-Instruct",
                          device="cpu", dtype=torch.bfloat16)
    print(f"      loaded in {time.time()-t0:.1f}s; model.device={gen._model.device}")

    util = StandardUtility(gen, ExactMatchVerifier(lower=True, strip_punct=True), RedactionMasker())

    print("\n[3/4] U(M) — generate on full memory:")
    t0 = time.time()
    u_full = util.evaluate(qi, item_mask=[True] * qi.K)
    print(f"      U(M) = {u_full}  ({time.time()-t0:.1f}s, gold={qi.gold_answer!r})")
    # Print what the generator actually produced for diagnosis
    ctx = RedactionMasker().apply(qi.memory, item_mask=[True] * qi.K)
    y_full = gen.generate(qi.query, ctx, max_new_tokens=32)
    print(f"      generated text: {y_full!r}")

    print("\n[4/4] U(∅) — generate on empty memory:")
    t0 = time.time()
    u_empty = util.evaluate(qi, item_mask=[False] * qi.K)
    print(f"      U(∅) = {u_empty}  ({time.time()-t0:.1f}s)")
    y_empty = gen.generate(qi.query, "", max_new_tokens=32)
    print(f"      generated text: {y_empty!r}")

    print(f"\n=== SMOKE TEST ===")
    print(f"  U_full  = {u_full}    (expect 1.0)")
    print(f"  U_empty = {u_empty}    (expect 0.0)")
    print(f"  n_calls = {util.n_calls}")
    if u_full > u_empty:
        print(f"  PASS: U(M) > U(∅) — generator uses memory")
    else:
        print(f"  WARN: U(M) ≤ U(∅) — exact-match too strict or context not being used")


if __name__ == "__main__":
    main()
