"""R11b (optional): Pareto curve of retention vs accuracy under
   - OwenThresholdPruner (our method)
   - RandomDropPruner (must-beat critical baseline)

For a small synthetic set, compute attribution once per sample (Owen), then
sweep retention targets and measure post-pruning accuracy under each strategy.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/r11b_pareto_sweep.py
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
from modality_credit.pruning.owen_threshold import OwenThresholdPruner
from modality_credit.pruning.random import RandomDropPruner
from modality_credit.utility import StandardUtility
from modality_credit.verifiers.exact_match import ExactMatchVerifier


def main(n: int, K: int, budget_B: int, top_n_inner: int, seed: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = build_rotating_modality_dataset(n, K)
    print(f"[1/4] Built {n} samples")

    print(f"\n[2/4] Loading Qwen-2.5-VL-7B...")
    t0 = time.time()
    gen = QwenVLGenerator(model_path="Qwen/Qwen2.5-VL-7B-Instruct",
                          device="cuda", dtype=torch.bfloat16, seed=seed)
    print(f"      loaded in {time.time()-t0:.1f}s")
    inner_util = StandardUtility(gen, ExactMatchVerifier(lower=True, strip_punct=True),
                                 RedactionMasker())
    util = CachedUtility(inner_util)

    print(f"\n[3/4] Computing Owen attribution per sample (B={budget_B})...")
    est = OwenEstimator(top_n_inner=top_n_inner, seed=seed)
    attrs = []
    for i, qi in enumerate(samples):
        t_s = time.time()
        a = est.estimate(util, qi, budget_B=budget_B)
        attrs.append(a)
        print(f"      [{i+1}/{n}] {qi.instance_id}: φ={[f'{x:+.2f}' for x in a.phi]} "
              f"({time.time()-t_s:.1f}s)")

    print(f"\n[4/4] Pareto sweep over retention targets...")
    owen_pruner = OwenThresholdPruner(RedactionMasker())
    rand_pruner = RandomDropPruner(RedactionMasker())

    # Each L_k=4, K=4, so 16 patches per sample. Sweep retentions.
    retentions = [0.25, 0.40, 0.50, 0.625, 0.75]
    rows = []
    for ret in retentions:
        # OwenThresholdPruner: choose tau to roughly match `ret`
        # Heuristic: tau is set so ~ret of patches get kept globally. For each
        # sample, find tau such that ψ > tau filters to target retention.
        # Simpler: use ranked thresholds from observed (φ,ψ) values.
        owen_accs, owen_rets = [], []
        rand_accs, rand_rets = [], []
        for qi, attr in zip(samples, attrs):
            # Owen pruner with target retention via tau search
            # We pick (tau_1, tau_2) such that observed retention ≈ ret
            # Brute-force tau_1=0 first (no item filter), then tune tau_2.
            # For simplicity, use a fixed grid and pick the one closest to ret.
            best_pruned = None
            best_diff = 999
            for tau_2 in [-1.0, -0.1, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
                p = owen_pruner.prune(qi, attr, tau_1=-1.0, tau_2=tau_2)
                d = abs(p.retention_ratio - ret)
                if d < best_diff:
                    best_diff = d
                    best_pruned = p
            owen_accs.append(util.evaluate(qi, best_pruned.item_mask,
                                           best_pruned.modality_masks))
            owen_rets.append(best_pruned.retention_ratio)
            r = rand_pruner.prune(qi, attr, retention=ret, seed=seed)
            rand_accs.append(util.evaluate(qi, r.item_mask, r.modality_masks))
            rand_rets.append(r.retention_ratio)
        row = {
            "target_retention": ret,
            "owen_mean_retention": float(np.mean(owen_rets)),
            "owen_accuracy": float(np.mean(owen_accs)),
            "rand_mean_retention": float(np.mean(rand_rets)),
            "rand_accuracy": float(np.mean(rand_accs)),
            "delta_acc": float(np.mean(owen_accs) - np.mean(rand_accs)),
        }
        rows.append(row)
        print(f"  ret≈{ret:.2f}: owen_acc={row['owen_accuracy']:.3f} "
              f"rand_acc={row['rand_accuracy']:.3f} Δ={row['delta_acc']:+.3f}")

    out = {"K": K, "budget_B": budget_B, "n_samples": n, "rows": rows}
    (out_dir / "r11b_pareto.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote: {out_dir/'r11b_pareto.json'}")
    print(f"\n=== CLAIM 2 (synthetic) ===")
    print(f"  ΔAcc at ~0.40 retention: {next(r['delta_acc'] for r in rows if r['target_retention'] == 0.40):+.3f}")
    print(f"  must-beat threshold:     +0.10 (10 pp) for top-venue case")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--top-n-inner", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("results/r11b_pareto"))
    args = ap.parse_args()
    main(args.n, args.K, args.B, args.top_n_inner, args.seed, args.out)
