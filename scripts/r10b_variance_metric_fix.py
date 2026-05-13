"""R10b: verify the correct variance metric for Phase 1.

The original 02_toy_pilot.py computes phi_stack.var(axis=0) ACROSS SAMPLES,
which is meaningless on heterogeneous samples (different true φ each).
Prop 2's bound Var(φ̂) ≤ 2^K/(4N) is about ESTIMATOR VARIANCE — i.e., the
variance of φ̂ across multiple MC seeds for a FIXED sample.

This script runs 3 samples × 5 seeds each, computes the correct across-seed
variance, and compares to bound. Conclusion expected: var_ratio ≪ 1.
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
from modality_credit.estimator.owen import OwenEstimator
from modality_credit.generators.qwen_vl import QwenVLGenerator
from modality_credit.masking.redaction import RedactionMasker
from modality_credit.utility import StandardUtility
from modality_credit.verifiers.exact_match import ExactMatchVerifier


def main(n_samples: int, n_seeds: int, K: int, budget_B: int, top_n_inner: int,
         out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = build_rotating_modality_dataset(n_samples, K)
    print(f"[1/3] Built {n_samples} samples")

    print(f"\n[2/3] Loading Qwen-VL...")
    t0 = time.time()
    gen = QwenVLGenerator(model_path="Qwen/Qwen2.5-VL-7B-Instruct",
                          device="cuda", dtype=torch.bfloat16)
    print(f"      loaded in {time.time()-t0:.1f}s")
    inner_util = StandardUtility(gen, ExactMatchVerifier(lower=True, strip_punct=True),
                                 RedactionMasker())
    util = CachedUtility(inner_util)

    print(f"\n[3/3] Running {n_samples} × {n_seeds} seeds...")
    # phi_per_sample[i] is (n_seeds, K)
    phi_per_sample = []
    rows = []
    for i, qi in enumerate(samples):
        seed_phis = []
        for s in range(n_seeds):
            est = OwenEstimator(top_n_inner=top_n_inner, seed=42 + s * 100)
            t = time.time()
            attr = est.estimate(util, qi, budget_B=budget_B)
            seed_phis.append(attr.phi.copy())
            print(f"      sample {i+1}/{n_samples} seed {s+1}/{n_seeds}: "
                  f"phi={[f'{x:+.2f}' for x in attr.phi]} "
                  f"({time.time()-t:.1f}s)")
        seed_phis = np.stack(seed_phis)  # (n_seeds, K)
        phi_per_sample.append(seed_phis)
        # Across-seed variance per k for this sample
        var_per_k_for_sample = seed_phis.var(axis=0, ddof=1)
        rows.append({
            "sample_id": qi.instance_id,
            "decisive_mod": qi.metadata["decisive_modality"],
            "phi_across_seeds_mean": seed_phis.mean(axis=0).tolist(),
            "phi_across_seeds_var": var_per_k_for_sample.tolist(),
            "phi_across_seeds_var_mean_over_k": float(var_per_k_for_sample.mean()),
        })

    # Aggregate across-seed variance — properly: mean over (sample, k) of var-across-seeds
    all_seed_vars = np.stack([r["phi_across_seeds_var"] for r in rows])  # (n_samples, K)
    correct_var = float(all_seed_vars.mean())
    bound = (2 ** K) / (4 * budget_B)

    # For comparison: the BUGGY across-samples metric
    phi_means_per_sample = np.stack([r["phi_across_seeds_mean"] for r in rows])  # (n_samples, K)
    buggy_var = float(phi_means_per_sample.var(axis=0, ddof=1).mean())

    summary = {
        "n_samples": n_samples, "n_seeds": n_seeds, "K": K, "budget_B": budget_B,
        "var_phi_bound": bound,
        "across_seed_var (correct, Prop 2)": correct_var,
        "across_seed_var_ratio_to_bound": correct_var / bound,
        "across_samples_var (BUGGY metric in 02_toy_pilot.py)": buggy_var,
        "across_samples_var_ratio_to_bound": buggy_var / bound,
        "rows": rows,
    }
    print(f"\n=== VARIANCE METRIC COMPARISON ===")
    print(f"  Prop 2 bound (2^K/(4N))                 = {bound:.4f}")
    print(f"  Correct across-seed variance            = {correct_var:.4f}")
    print(f"      ratio to bound                       = {correct_var/bound:.3f}  (target < 1)")
    print(f"  Buggy across-sample variance            = {buggy_var:.4f}")
    print(f"      ratio to bound                       = {buggy_var/bound:.3f}  (meaningless on heterogeneous samples)")

    (out_dir / "r10b_variance.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote: {out_dir/'r10b_variance.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="number of distinct samples")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--top-n-inner", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("results/r10b_variance"))
    args = ap.parse_args()
    main(args.n, args.seeds, args.K, args.B, args.top_n_inner, args.out)
