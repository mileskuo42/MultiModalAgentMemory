"""V4: Algorithm-1 optimality — does N_out* ∝ √(2^K), N_in^(k)* ∝ √(2^{L_k})
actually minimize total estimation variance under fixed budget B?

Strategy:
  - Fix K, L_k uniform, B
  - Define a grid of (N_out, N_in_each) allocations summing to ~B
  - For each grid point, run multiple seeds, measure
        TotalVar = Var(φ̂) + Σ_k Var(ψ̂^(k))   (outer + inner)
  - Plot TotalVar vs N_out / N_in ratio
  - Verify: Alg-1's prediction lies near the empirical minimum
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from modality_credit.allocation import allocate_budget
from modality_credit.estimator.kernel_shap import kernel_shap_outer
from modality_credit.estimator.mc_permutation import mc_perm_inner
from modality_credit.generators.mock import MockGenerator
from modality_credit.masking.redaction import RedactionMasker
from modality_credit.types import MemoryItem, QueryInstance
from modality_credit.utility import StandardUtility
from modality_credit.verifiers.exact_match import ExactMatchVerifier


def build_qi(K: int, L: int) -> QueryInstance:
    """K items, L modalities each.
    Items 0,1 carry markers in their text modality; their AUDIO modality also
    carries a partial-credit marker. This gives non-trivial inner-game φ as
    well as non-trivial outer-game ψ.
    """
    mods = ["text", "audio", "scene"][:L]
    mem = []
    for k in range(K):
        modalities = {}
        for m in mods:
            if k == 0 and m == "text":
                modalities[m] = "the keyword is alpha"
            elif k == 0 and m == "audio":
                modalities[m] = "audio also mentions alpha"
            elif k == 1 and m == "text":
                modalities[m] = "the keyword is beta"
            elif k == 1 and m == "audio":
                modalities[m] = "audio also mentions beta"
            else:
                modalities[m] = f"irrelevant_{k}_{m}"
        mem.append(MemoryItem(item_id=f"item_{k}", modalities=modalities))
    return QueryInstance(
        instance_id="v4_fixed_AND",
        query="State both keywords",
        memory=mem,
        gold_answer="alpha+beta",
    )


def _gen_fn(query: str, context: str) -> str:
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
    return StandardUtility(MockGenerator(_gen_fn), ExactMatchVerifier(), RedactionMasker())


def estimate_with_alloc(qi: QueryInstance, N_out: int, N_in_per_item: int,
                        top_n_inner: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Run outer + inner with manually-specified N. Returns (phi, psi)."""
    K = qi.K
    L_ks = qi.L_ks
    util = make_util()

    phi = kernel_shap_outer(util, qi, N=N_out, seed=seed)
    top_idx = np.argsort(-phi)[:top_n_inner]
    max_L = max(L_ks)
    psi = np.full((K, max_L), np.nan, dtype=np.float64)
    for k in top_idx:
        psi_k = mc_perm_inner(util, qi, item_idx=int(k),
                              N=N_in_per_item, seed=seed + int(k))
        psi[k, :len(psi_k)] = psi_k
    return phi, psi


def total_variance(phis: np.ndarray, psis: np.ndarray) -> dict:
    """phis: (n_seeds, K). psis: (n_seeds, K, max_L). Total variance metric."""
    outer_var = phis.var(axis=0, ddof=1).sum()  # sum across k
    # Only count inner var for items consistently picked into top_idx across seeds
    psi_valid = ~np.isnan(psis).any(axis=0)  # (K, max_L)
    if psi_valid.any():
        inner_var_arr = np.nanvar(psis, axis=0, ddof=1)
        inner_var = inner_var_arr[psi_valid].sum()
    else:
        inner_var = 0.0
    return {
        "outer_var": float(outer_var),
        "inner_var": float(inner_var),
        "total_var": float(outer_var + inner_var),
    }


def sweep(K: int, L: int, B: int, top_n_inner: int, n_seeds: int) -> tuple[pd.DataFrame, dict]:
    qi = build_qi(K, L)
    # Allocation grid — each entry: (N_out, N_in_per_item), should sum to ~B
    # In our setup, total = N_out + top_n_inner * N_in_per_item (only top items get inner)
    # Choose a grid covering different N_out / N_in ratios.
    grid_n_out = [25, 50, 100, 150, 200, 250, 350]
    # For each N_out, compute N_in = (B - N_out) // top_n_inner
    grid = []
    for n_o in grid_n_out:
        n_i = (B - n_o) // top_n_inner
        if n_i >= 10:  # keep min for variance estimable
            grid.append((n_o, n_i))

    # Alg-1's prediction (fixed: passes top_n_inner)
    N_out_pred, N_in_pred = allocate_budget(K, [L] * K, B,
                                            min_per_game=10,
                                            top_n_inner=top_n_inner)
    alg1_n_in_per_item = int(np.mean(N_in_pred))  # uniform L_k so they're equal
    grid.append((N_out_pred, alg1_n_in_per_item))
    grid = sorted(set(grid))

    rows = []
    for n_o, n_i in grid:
        phis, psis = [], []
        for s in range(n_seeds):
            phi, psi = estimate_with_alloc(qi, N_out=n_o, N_in_per_item=n_i,
                                           top_n_inner=top_n_inner, seed=s)
            phis.append(phi); psis.append(psi)
        phis = np.stack(phis)
        psis = np.stack(psis)
        var = total_variance(phis, psis)
        rows.append({
            "N_out": n_o, "N_in_per_item": n_i,
            "total_queries": n_o + top_n_inner * n_i,
            **var,
            "is_alg1_prediction": (n_o == N_out_pred and n_i == alg1_n_in_per_item),
        })
    return pd.DataFrame(rows), {"N_out_pred": N_out_pred, "N_in_pred": alg1_n_in_per_item}


def main(K: int, L: int, B: int, top_n_inner: int, n_seeds: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"V4 allocation sweep: K={K} L={L} B={B} top_n_inner={top_n_inner} n_seeds={n_seeds}")
    t0 = time.time()
    df, pred = sweep(K, L, B, top_n_inner, n_seeds)
    wall = time.time() - t0

    df_sorted = df.sort_values("total_var").reset_index(drop=True)
    min_var = df_sorted["total_var"].iloc[0]
    alg1_row = df[df["is_alg1_prediction"]].iloc[0]
    alg1_var = float(alg1_row["total_var"])

    print(f"\n=== V4 SUMMARY (wall {wall:.2f}s) ===")
    print(df.sort_values("N_out").to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nAlg-1 predicts: N_out={pred['N_out_pred']}, N_in_per_item={pred['N_in_pred']}")
    print(f"Alg-1's total_var:        {alg1_var:.5f}")
    print(f"Empirical min total_var:  {min_var:.5f}  (at N_out={df_sorted['N_out'].iloc[0]}, N_in={df_sorted['N_in_per_item'].iloc[0]})")
    print(f"Alg-1 / empirical-min:    {alg1_var/min_var:.3f}× (1.0 = perfect, <1.2 considered passing)")

    # ---- artifacts
    df.to_csv(out_dir / "v4_grid.csv", index=False)
    (out_dir / "v4_summary.json").write_text(json.dumps({
        "K": K, "L": L, "budget_B": B, "n_seeds": n_seeds, "top_n_inner": top_n_inner,
        "alg1_N_out": int(pred["N_out_pred"]),
        "alg1_N_in_per_item": int(pred["N_in_pred"]),
        "alg1_total_var": alg1_var,
        "empirical_min_total_var": float(min_var),
        "alg1_over_min": float(alg1_var / min_var),
        "wall_clock_sec": wall,
    }, indent=2))
    print(f"\nWrote: {out_dir/'v4_grid.csv'}, {out_dir/'v4_summary.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--top-n-inner", type=int, default=2)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("results/sanity/v4"))
    args = ap.parse_args()
    main(args.K, args.L, args.B, args.top_n_inner, args.n_seeds, args.out)
