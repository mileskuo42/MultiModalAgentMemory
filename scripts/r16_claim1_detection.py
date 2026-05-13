"""R16 / Claim 1: top-vs-bottom quartile ΔAcc on Owen vs item-only Shapley.

The detection metric: rank all (item × modality) patches by attribution
score, remove top-25% vs bottom-25%, compare accuracies. A faithful
attribution gives a LARGE positive ΔAcc (top quartile matters more).

Owen attribution computed with top_n_inner=K (every item gets inner game)
so ψ is fully populated. Item-only Shapley uses φ_k/L_k uniform split.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/r16_claim1_detection.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from modality_credit.caching import CachedUtility
from modality_credit.data.synthetic import build_rotating_modality_dataset
from modality_credit.estimator.item_only import ItemOnlyEstimator
from modality_credit.estimator.owen import OwenEstimator
from modality_credit.generators.qwen_vl import QwenVLGenerator
from modality_credit.masking.redaction import RedactionMasker
from modality_credit.metrics.separability import top_bottom_quartile_delta_acc
from modality_credit.utility import StandardUtility
from modality_credit.verifiers.exact_match import ExactMatchVerifier


def main(n: int, K: int, budget_B: int, seed: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = build_rotating_modality_dataset(n, K, use_neutral_fillers=True)
    print(f"[1/4] Built {n} samples (neutral fillers, K={K})")

    print(f"\n[2/4] Loading Qwen-2.5-VL-7B...")
    t0 = time.time()
    gen = QwenVLGenerator(model_path="Qwen/Qwen2.5-VL-7B-Instruct",
                          device="cuda", dtype=torch.bfloat16, seed=seed)
    print(f"      loaded in {time.time()-t0:.1f}s")
    inner_util = StandardUtility(gen, ExactMatchVerifier(lower=True, strip_punct=True),
                                 RedactionMasker())
    util = CachedUtility(inner_util)

    print(f"\n[3a/4] Computing Owen attribution (B={budget_B}, top_n_inner=K={K} for full ψ)...")
    t_owen = time.time()
    owen_est = OwenEstimator(top_n_inner=K, seed=seed)
    owen_attrs = []
    for i, qi in enumerate(samples):
        a = owen_est.estimate(util, qi, budget_B=budget_B)
        owen_attrs.append(a)
        print(f"  Owen [{i+1}/{n}] phi={[f'{x:+.2f}' for x in a.phi]} U_full={a.U_full:.0f}")
    wall_owen = time.time() - t_owen
    print(f"  Owen total wall-clock: {wall_owen:.1f}s")

    print(f"\n[3b/4] Computing item-only Shapley (B={budget_B})...")
    t_io = time.time()
    io_est = ItemOnlyEstimator(seed=seed)
    io_attrs = []
    for i, qi in enumerate(samples):
        a = io_est.estimate(util, qi, budget_B=budget_B)
        io_attrs.append(a)
    wall_io = time.time() - t_io
    print(f"  item-only total wall-clock: {wall_io:.1f}s")

    print(f"\n[4/4] Computing top-vs-bottom quartile ΔAcc...")
    t_metric = time.time()
    owen_result = top_bottom_quartile_delta_acc(util, samples, owen_attrs)
    io_result = top_bottom_quartile_delta_acc(util, samples, io_attrs)
    wall_metric = time.time() - t_metric

    print(f"\n=== CLAIM 1 DETECTION (top-quartile removed vs bottom-quartile removed) ===")
    print(f"  method            acc(top_removed)  acc(bottom_removed)  ΔAcc")
    print(f"  Owen (top_n=K)    {owen_result['acc_top_removed']:.3f}             {owen_result['acc_bottom_removed']:.3f}                {owen_result['delta_acc']:+.3f}")
    print(f"  item-only         {io_result['acc_top_removed']:.3f}             {io_result['acc_bottom_removed']:.3f}                {io_result['delta_acc']:+.3f}")
    print(f"  Δ(Owen−ItemOnly): {owen_result['delta_acc'] - io_result['delta_acc']:+.3f}")
    print(f"  must-accept threshold for Claim 1: ΔAcc ≥ 0.15 (15pp)")

    summary = {
        "n": n, "K": K, "budget_B": budget_B, "seed": seed,
        "owen_top_n_inner": K,
        "owen": owen_result,
        "item_only": io_result,
        "wall_clock": {
            "owen_attribution_sec": wall_owen,
            "item_only_attribution_sec": wall_io,
            "metric_sec": wall_metric,
        },
    }
    (out_dir / "r16_claim1.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote: {out_dir/'r16_claim1.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("results/r16_claim1"))
    args = ap.parse_args()
    main(args.n, args.K, args.B, args.seed, args.out)
