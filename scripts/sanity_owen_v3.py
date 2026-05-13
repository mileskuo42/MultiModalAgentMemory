"""V3: empirical variance of φ̂ vs theoretical bound 2^K / (4N).

Calls kernel_shap_outer directly (not OwenEstimator) to isolate outer-game
variance from inner-game noise. Single QueryInstance, sweep N × seeds.

Hypothesis (Prop 2a): Var(φ̂_k) ≤ 2^K / (4N) for every k.
Slope of log Var vs log N should be ≈ -1.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from modality_credit.estimator.kernel_shap import kernel_shap_outer
from modality_credit.generators.mock import MockGenerator
from modality_credit.masking.redaction import RedactionMasker
from modality_credit.types import MemoryItem, QueryInstance
from modality_credit.utility import StandardUtility
from modality_credit.verifiers.exact_match import ExactMatchVerifier


def build_fixed_qi(K: int, L: int) -> QueryInstance:
    """Single deterministic instance with TWO co-decisive items (0, 1).

    Items 0 and 1 carry markers "alpha" and "beta" in their text modality;
    items 2..K-1 carry irrelevant content. Gold answer requires BOTH markers.
    Closed-form Shapley:  φ = [0.5, 0.5, 0, 0, ...],  Σφ = 1.
    """
    mods = ["text", "audio", "scene"][:L]
    decisive_markers = {0: "alpha", 1: "beta"}
    mem = []
    for k in range(K):
        modalities = {}
        for m in mods:
            if k in decisive_markers and m == "text":
                modalities[m] = f"the keyword is {decisive_markers[k]}"
            else:
                modalities[m] = f"irrelevant_{k}_{m}"
        mem.append(MemoryItem(item_id=f"item_{k}", modalities=modalities))
    return QueryInstance(
        instance_id="v3_fixed_AND",
        query="State both keywords",
        memory=mem,
        gold_answer="alpha+beta",
    )


def _generator_fn(query: str, context: str) -> str:
    """Emit 'alpha+beta' iff BOTH keywords appear in context; else 'partial'/'none'."""
    has_a = "alpha" in context
    has_b = "beta" in context
    if has_a and has_b:
        return "alpha+beta"
    if has_a:
        return "alpha"
    if has_b:
        return "beta"
    return "none"


def make_util() -> StandardUtility:
    gen = MockGenerator(_generator_fn)
    return StandardUtility(gen, ExactMatchVerifier(), RedactionMasker())


def run_one(qi: QueryInstance, N: int, seed: int) -> np.ndarray:
    """Single φ estimate from kernel_shap_outer."""
    util = make_util()
    return kernel_shap_outer(util, qi, N=N, seed=seed)


def sweep(K: int, L: int, N_list: list[int], n_seeds: int) -> pd.DataFrame:
    qi = build_fixed_qi(K, L)
    rows = []
    for N in N_list:
        # collect phi across seeds
        phis = np.stack([run_one(qi, N=N, seed=s) for s in range(n_seeds)])  # (n_seeds, K)
        emp_var_per_k = phis.var(axis=0, ddof=1)  # unbiased; shape (K,)
        bound = (2 ** K) / (4 * N)
        for k in range(K):
            rows.append({
                "N": N,
                "k": k,
                "phi_mean": float(phis[:, k].mean()),
                "phi_std": float(phis[:, k].std(ddof=1)),
                "phi_var_empirical": float(emp_var_per_k[k]),
                "phi_var_bound": float(bound),
                "ratio_empirical_over_bound": float(emp_var_per_k[k] / bound),
            })
    return pd.DataFrame(rows)


def main(K: int, L: int, n_seeds: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    N_list = [25, 50, 100, 200, 400]
    print(f"V3 variance sweep: K={K} L={L} n_seeds={n_seeds} N_list={N_list}")
    t0 = time.time()
    df = sweep(K, L, N_list, n_seeds)
    wall = time.time() - t0

    # ---- summary
    df_per_N = df.groupby("N").agg(
        emp_var_max=("phi_var_empirical", "max"),
        emp_var_mean=("phi_var_empirical", "mean"),
        bound=("phi_var_bound", "first"),
        ratio_max=("ratio_empirical_over_bound", "max"),
    ).reset_index()
    df_per_N["pass"] = df_per_N["ratio_max"] < 1.0

    # Per-N mean of φ (across seeds) — should converge to closed-form [0.5, 0.5, 0, ...]
    df["k"] = df["k"].astype(int)
    phi_mean_table = df.pivot_table(index="N", columns="k", values="phi_mean")
    print(f"\n=== Mean φ across {n_seeds} seeds (true φ = [0.5, 0.5, 0, 0, ...]) ===")
    print(phi_mean_table.to_string(float_format=lambda x: f"{x:+.3f}"))

    # ---- slope check: log emp_var_mean vs log N
    log_N = np.log(df_per_N["N"].values)
    log_v = np.log(df_per_N["emp_var_mean"].values + 1e-30)  # guard against zero
    slope, intercept = np.polyfit(log_N, log_v, 1)

    print(f"\n=== V3 SUMMARY (K={K}, L={L}) — wall {wall:.2f}s ===")
    print(df_per_N.to_string(index=False))
    print(f"\nlog-log slope of empirical variance vs N: {slope:.3f}")
    print(f"  Prop 2 predicts slope = -1.0 (Var ∝ 1/N)")
    print(f"\nBound respected at all N: {df_per_N['pass'].all()} (max ratio = {df_per_N['ratio_max'].max():.3f})")

    # ---- artifacts
    df.to_csv(out_dir / "v3_per_k.csv", index=False)
    df_per_N.to_csv(out_dir / "v3_per_N.csv", index=False)
    (out_dir / "v3_summary.json").write_text(json.dumps({
        "K": K, "L": L, "n_seeds": n_seeds, "N_list": N_list,
        "log_log_slope": float(slope),
        "bound_respected_all_N": bool(df_per_N["pass"].all()),
        "max_ratio": float(df_per_N["ratio_max"].max()),
        "wall_clock_sec": wall,
    }, indent=2))
    print(f"\nWrote: {out_dir/'v3_per_k.csv'}")
    print(f"Wrote: {out_dir/'v3_per_N.csv'}")
    print(f"Wrote: {out_dir/'v3_summary.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("results/sanity/v3"))
    args = ap.parse_args()
    main(args.K, args.L, args.n_seeds, args.out)
