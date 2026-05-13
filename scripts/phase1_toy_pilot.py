"""Phase 1 toy pilot — the go/no-go gate before full experiments.

Mirrors scripts/02_toy_pilot.py but uses the working stack (StandardUtility,
QwenVLGenerator, RedactionMasker with vision routing). 30 synthetic samples
with decisive modality rotated across {vision, text, audio, scene}.

Decision rule (from method plan / docs/README.md):
  - var_ratio_phi (empirical / theoretical) > 3      → FAIL pivot
  - conservation_residual_mean > 0.10               → FAIL pivot
  - either warning band (1.5–3× / 5–10%)            → WARN re-run
  - everything green                                 → PASS, proceed to Phase 2

Wall-clock budget: <2 h on a single A100.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/phase1_toy_pilot.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from modality_credit.caching import CachedUtility
from modality_credit.estimator.owen import OwenEstimator
from modality_credit.generators.qwen_vl import QwenVLGenerator
from modality_credit.masking.redaction import RedactionMasker
from modality_credit.utility import StandardUtility
from modality_credit.verifiers.exact_match import ExactMatchVerifier
from modality_credit.data.synthetic import build_rotating_modality_dataset as build_audit_dataset


def main(n: int, K: int, budget_B: int, top_n_inner: int, seed: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Building Phase 1 dataset (n={n}, K={K}, neutral_fillers=True)...")
    samples = build_audit_dataset(n, K, use_neutral_fillers=True)
    decisive_mod_dist = {}
    for s in samples:
        m = s.metadata["decisive_modality"]
        decisive_mod_dist[m] = decisive_mod_dist.get(m, 0) + 1
    print(f"      decisive modality distribution: {decisive_mod_dist}")

    print(f"\n[2/4] Loading Qwen-2.5-VL-7B on GPU (bf16)...")
    t0 = time.time()
    gen = QwenVLGenerator(model_path="Qwen/Qwen2.5-VL-7B-Instruct",
                          device="cuda", dtype=torch.bfloat16, seed=seed)
    print(f"      loaded in {time.time()-t0:.1f}s")

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
        gt_k = next(idx for idx, m in enumerate(qi.memory) if m.metadata["decisive"])
        gt_mod = qi.metadata["decisive_modality"]
        mods = list(qi.memory[gt_k].modalities.keys())
        psi_k = attr.psi[gt_k, :len(mods)] if gt_k in attr.top_idx else None
        psi_dict = None
        psi_argmax_mod = None
        if psi_k is not None and not np.all(np.isnan(psi_k)):
            psi_dict = {mods[j]: float(psi_k[j]) for j in range(len(mods))}
            psi_argmax_mod = mods[int(np.nanargmax(psi_k))]
        rows.append({
            "instance_id": qi.instance_id,
            "decisive_mod": gt_mod,
            "gt_item_idx": gt_k,
            "U_full": float(attr.U_full),
            "U_empty": float(attr.U_empty),
            "phi": [float(x) for x in attr.phi],
            "top_idx": [int(x) for x in attr.top_idx],
            "top1_item_correct": int(attr.top_idx[0]) == gt_k,
            "psi_argmax_mod": psi_argmax_mod,
            "top1_mod_correct": (psi_argmax_mod == gt_mod) if psi_argmax_mod is not None else None,
            "conservation_residual": float(attr.conservation_residual),
            "wall_clock_sec": wall_s,
            "psi_at_gt": psi_dict,
        })
        print(f"      [{i+1}/{n}] {qi.instance_id}: "
              f"gt_mod={gt_mod} top_item={attr.top_idx[0]}(={gt_k}? {rows[-1]['top1_item_correct']}) "
              f"psi_argmax={psi_argmax_mod}(={gt_mod}? {rows[-1]['top1_mod_correct']}) "
              f"U_full={attr.U_full:.0f} cons={attr.conservation_residual:.3f} "
              f"wall={wall_s:.1f}s")
    wall_total = time.time() - t_exp

    # ---- Phase 1 monitors
    cons = np.array([r["conservation_residual"] for r in rows])
    item_acc = float(np.mean([r["top1_item_correct"] for r in rows]))
    mod_correct = [r["top1_mod_correct"] for r in rows if r["top1_mod_correct"] is not None]
    mod_acc = float(np.mean(mod_correct)) if mod_correct else float("nan")
    u_full_mean = float(np.mean([r["U_full"] for r in rows]))
    u_empty_mean = float(np.mean([r["U_empty"] for r in rows]))

    # When U_full = 0, U(M)−U(∅) = 0, so Σφ = 0 → no signal to attribute.
    # The framework's correctness is meaningful only on the U_full=1 subset
    # (samples where the model could actually answer with full memory).
    rows_with_signal = [r for r in rows if r["U_full"] == 1.0]
    n_with_signal = len(rows_with_signal)
    item_acc_signal = (float(np.mean([r["top1_item_correct"] for r in rows_with_signal]))
                       if n_with_signal else float("nan"))
    mod_correct_signal = [r["top1_mod_correct"] for r in rows_with_signal
                          if r["top1_mod_correct"] is not None]
    mod_acc_signal = (float(np.mean(mod_correct_signal))
                      if mod_correct_signal else float("nan"))

    # NOTE: the original 02_toy_pilot.py template computed
    #   phi_stack.var(axis=0).mean()
    # i.e., variance ACROSS HETEROGENEOUS SAMPLES. Prop 2's Var(φ̂) bound is
    # about ESTIMATOR variance — i.e., variance across MC seeds for the SAME
    # sample. The across-sample variance is dominated by signal (different
    # true φ per sample), not estimator noise, so it is *not* a valid Prop 2
    # check. Including it for backward-compat with the template, but the
    # verdict no longer depends on it (see scripts/r10b_variance_metric_fix.py
    # for a proper across-seed variance check; ratio is 0.089 << 1 there).
    phi_stack = np.stack([np.array(r["phi"]) for r in rows])  # (n, K)
    across_sample_var = float(phi_stack.var(axis=0, ddof=1).mean())
    var_phi_bound = (2 ** K) / (4 * budget_B)
    across_sample_var_ratio = across_sample_var / var_phi_bound

    # per-decisive-modality breakdown
    per_mod = {}
    for r in rows:
        m = r["decisive_mod"]
        per_mod.setdefault(m, {"n": 0, "item_correct": 0, "mod_correct": 0})
        per_mod[m]["n"] += 1
        per_mod[m]["item_correct"] += int(r["top1_item_correct"])
        if r["top1_mod_correct"] is not None:
            per_mod[m]["mod_correct"] += int(r["top1_mod_correct"])

    # ---- decision
    # Verdict uses CONSERVATION + RECOVERY-ON-SIGNAL-SUBSET, not the
    # heterogeneous-sample variance (see note above).
    cons_mean = float(cons.mean())
    if cons_mean > 0.10:
        verdict = "FAIL — conservation > 10%; estimator implementation bug"
    elif n_with_signal == 0:
        verdict = "HALT — every sample had U_full=0; model can't answer at all (synthetic too hard?)"
    elif cons_mean > 0.05 or item_acc_signal < 0.85:
        verdict = (f"WARN — cons {cons_mean:.2%} or item-recovery-on-signal "
                   f"{item_acc_signal:.0%}; investigate before Phase 2")
    else:
        verdict = f"PASS — conservation {cons_mean:.2%}, item recovery {item_acc_signal:.0%} on {n_with_signal}/{n} signal samples"

    summary = {
        "n_samples": n,
        "K": K,
        "budget_B": budget_B,
        "top_n_inner": top_n_inner,
        "u_full_mean": u_full_mean,
        "u_empty_mean": u_empty_mean,
        "conservation_residual_mean": float(cons.mean()),
        "conservation_residual_max": float(cons.max()),
        "top1_item_recovery_all": item_acc,
        "top1_mod_recovery_all_conditional": mod_acc,
        "n_with_signal_u_full_eq_1": n_with_signal,
        "top1_item_recovery_signal_only": item_acc_signal,
        "top1_mod_recovery_signal_only": mod_acc_signal,
        "across_sample_var_NOT_prop2": across_sample_var,
        "across_sample_var_ratio_NOT_prop2": across_sample_var_ratio,
        "across_seed_var_bound_2K_over_4N": var_phi_bound,
        "wall_clock_total_sec": wall_total,
        "wall_clock_per_sample_sec": wall_total / max(n, 1),
        "per_decisive_modality": per_mod,
        "verdict": verdict,
    }

    print(f"\n[4/4] === PHASE 1 PILOT SUMMARY ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<35} {v:.4f}")
        elif isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"      {kk:<10} {vv}")
        else:
            print(f"  {k:<35} {v}")

    print(f"\n  >>> DECISION: {verdict}")

    (out_dir / "phase1_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_dir / "phase1_per_sample.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nWrote: {out_dir/'phase1_summary.json'}")
    print(f"Wrote: {out_dir/'phase1_per_sample.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--top-n-inner", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("results/phase1_pilot"))
    args = ap.parse_args()
    main(args.n, args.K, args.B, args.top_n_inner, args.seed, args.out)
