"""Mini Owen-value attribution experiment with real Qwen-VL on synthetic data.

The first real-model run of the framework. Replaces MockGenerator in the V2
sanity harness with Qwen-2.5-VL-7B. Reports per-sample phi, psi, top_idx,
conservation_residual, and aggregate top-1 (item, modality) recovery.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/mini_owen_qwen.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from modality_credit.caching import CachedUtility
from modality_credit.data.m3_bench import M3BenchDataset
from modality_credit.estimator.owen import OwenEstimator
from modality_credit.generators.qwen_vl import QwenVLGenerator
from modality_credit.masking.redaction import RedactionMasker
from modality_credit.utility import StandardUtility
from modality_credit.verifiers.exact_match import ExactMatchVerifier


def main(n_samples: int, K: int, budget_B: int, top_n_inner: int, seed: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Building dataset (synthetic, n={n_samples}, K={K})...")
    ds = M3BenchDataset(K=K, n=n_samples)
    samples = [ds[i] for i in range(len(ds))]
    print(f"      loaded {len(samples)} samples; L_ks={samples[0].L_ks}")

    print(f"\n[2/4] Loading Qwen-2.5-VL-7B on GPU (bf16)...")
    t0 = time.time()
    gen = QwenVLGenerator(model_path="Qwen/Qwen2.5-VL-7B-Instruct",
                          device="cuda", dtype=torch.bfloat16, seed=seed)
    print(f"      loaded in {time.time()-t0:.1f}s; device={gen._model.device}")

    inner_util = StandardUtility(gen, ExactMatchVerifier(lower=True, strip_punct=True),
                                 RedactionMasker())
    util = CachedUtility(inner_util)
    est = OwenEstimator(top_n_inner=top_n_inner, seed=seed)

    print(f"\n[3/4] Running Owen attribution (budget_B={budget_B}, top_n_inner={top_n_inner})...")
    rows = []
    t_exp = time.time()
    for i, qi in enumerate(samples):
        t_s = time.time()
        attr = est.estimate(util, qi, budget_B=budget_B)
        wall_s = time.time() - t_s

        # ground-truth: the decisive item is the one whose text contains the
        # actual fact (see synthetic generator). We tagged it via metadata.
        gt_item_idx = next(
            (k for k, m in enumerate(qi.memory) if m.metadata.get("decisive", False)),
            None,
        )
        top1_item_correct = (gt_item_idx is not None
                             and int(attr.top_idx[0]) == gt_item_idx)

        # modality top-1 recovery — for synthetic, decisive modality is "text"
        top1_mod_correct = None
        psi_at_gt = None
        if gt_item_idx is not None and gt_item_idx in attr.top_idx:
            mods = list(qi.memory[gt_item_idx].modalities.keys())
            psi_k = attr.psi[gt_item_idx, :len(mods)]
            if not np.all(np.isnan(psi_k)):
                argmax_idx = int(np.nanargmax(psi_k))
                top1_mod_correct = bool(mods[argmax_idx] == "text")
                psi_at_gt = psi_k.tolist()

        row = {
            "instance_id": qi.instance_id,
            "query": qi.query,
            "gold": qi.gold_answer,
            "K": qi.K,
            "L_ks": qi.L_ks,
            "U_full": float(attr.U_full),
            "U_empty": float(attr.U_empty),
            "phi": [float(x) for x in attr.phi],
            "phi_sum": float(attr.phi.sum()),
            "conservation_residual": float(attr.conservation_residual),
            "top_idx": [int(x) for x in attr.top_idx],
            "gt_item_idx": gt_item_idx,
            "top1_item_correct": top1_item_correct,
            "top1_mod_correct": top1_mod_correct,
            "psi_at_gt_item": psi_at_gt,
            "wall_clock_sec": wall_s,
            "meta": attr.meta,
        }
        rows.append(row)
        print(f"      [{i+1}/{n_samples}] {qi.instance_id}: "
              f"phi={[f'{x:+.2f}' for x in attr.phi]} "
              f"top_idx={list(attr.top_idx)} (gt={gt_item_idx}) "
              f"U_full={attr.U_full:.0f} cons_res={attr.conservation_residual:.3f} "
              f"wall={wall_s:.1f}s")

    wall_total = time.time() - t_exp

    # ---- aggregate
    item_acc = float(np.mean([r["top1_item_correct"] for r in rows]))
    mod_correct = [r["top1_mod_correct"] for r in rows if r["top1_mod_correct"] is not None]
    mod_acc = float(np.mean(mod_correct)) if mod_correct else float("nan")
    cons_residuals = np.array([r["conservation_residual"] for r in rows])
    u_full = np.array([r["U_full"] for r in rows])
    u_empty = np.array([r["U_empty"] for r in rows])

    summary = {
        "n_samples": n_samples,
        "K": K,
        "budget_B": budget_B,
        "top_n_inner": top_n_inner,
        "seed": seed,
        "u_full_mean": float(u_full.mean()),
        "u_empty_mean": float(u_empty.mean()),
        "top1_item_recovery": item_acc,
        "top1_mod_recovery_conditional": mod_acc,
        "kstar_in_top_n_inner_rate": float(len(mod_correct) / max(len(rows), 1)),
        "conservation_residual_mean": float(cons_residuals.mean()),
        "conservation_residual_max": float(cons_residuals.max()),
        "wall_clock_total_sec": float(wall_total),
        "wall_clock_per_sample_sec": float(wall_total / max(len(rows), 1)),
        "n_calls_total": gen._model.device.type,  # placeholder; updated below
    }

    print(f"\n[4/4] === SUMMARY ===")
    for k, v in summary.items():
        if k == "n_calls_total":
            continue
        if isinstance(v, float):
            print(f"  {k:<35} {v:.4f}")
        else:
            print(f"  {k:<35} {v}")

    (out_dir / "mini_owen_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "mini_owen_per_sample.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nWrote: {out_dir/'mini_owen_summary.json'}")
    print(f"Wrote: {out_dir/'mini_owen_per_sample.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--B", type=int, default=64, help="Owen budget per sample")
    ap.add_argument("--top-n-inner", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("results/mini_owen_qwen"))
    args = ap.parse_args()
    main(args.n, args.K, args.B, args.top_n_inner, args.seed, args.out)
