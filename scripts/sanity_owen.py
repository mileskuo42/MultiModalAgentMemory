"""CPU-only sanity validation of the Owen-value estimator.

Runs V1 (conservation) + V2 (top-N recovery) on a synthetic benchmark where
ground-truth (k*, ℓ*) is injected as a unique marker substring. The
MockGenerator returns the marker iff it appears in the redacted context.

Reads thread caps from env; ensure
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
when running on shared hardware.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from modality_credit.estimator.owen import OwenEstimator
from modality_credit.generators.mock import MockGenerator
from modality_credit.masking.redaction import RedactionMasker
from modality_credit.types import MemoryItem, Modality, QueryInstance
from modality_credit.utility import StandardUtility
from modality_credit.verifiers.exact_match import ExactMatchVerifier

MODALITY_POOL: list[Modality] = ["text", "audio", "scene"]  # no vision: see R1 notes


def make_sample(idx: int, K: int, L: int, rng: np.random.Generator) -> tuple[QueryInstance, int, Modality]:
    """Build one (Q, M, y*) with a unique marker in a randomly chosen (k*, ℓ*)."""
    assert L <= len(MODALITY_POOL), f"L={L} > {len(MODALITY_POOL)}"
    mods = MODALITY_POOL[:L]
    k_star = int(rng.integers(0, K))
    ell_star = mods[int(rng.integers(0, L))]
    marker = f"marker_{idx}_{k_star}_{ell_star}"
    mem = []
    for k in range(K):
        modalities: dict[Modality, str] = {}
        for m in mods:
            if k == k_star and m == ell_star:
                modalities[m] = f"the answer is {marker}"
            else:
                modalities[m] = f"irrelevant content for item {k} modality {m}"
        mem.append(MemoryItem(item_id=f"sample{idx}_item{k}", modalities=modalities))
    qi = QueryInstance(
        instance_id=f"sample_{idx}",
        query="What is the marker token?",
        memory=mem,
        gold_answer=marker,
    )
    return qi, k_star, ell_star


def decisive_response_fn(marker: str):
    """Generator: return the marker iff it appears in context, else 'unknown'."""
    def fn(query: str, context: str) -> str:
        return marker if marker in context else "unknown"
    return fn


@dataclass
class SampleResult:
    sample_id: int
    K: int
    L: int
    budget_B: int
    seed: int
    k_star: int
    ell_star: str
    U_full: float
    U_empty: float
    phi_sum: float
    conservation_residual: float
    top1_item_correct: bool
    top1_mod_correct: bool | None  # None if k* not in top_idx
    phi: list[float]
    psi_at_kstar: list[float] | None
    wall_clock_sec: float
    n_calls: int


def run_one(sample_idx: int, K: int, L: int, budget_B: int, seed: int) -> SampleResult:
    rng = np.random.default_rng(seed + sample_idx)
    qi, k_star, ell_star = make_sample(sample_idx, K, L, rng)
    gen = MockGenerator(decisive_response_fn(qi.gold_answer))
    util = StandardUtility(gen, ExactMatchVerifier(), RedactionMasker())
    est = OwenEstimator(top_n_inner=2, seed=seed + sample_idx)

    t0 = time.time()
    attr = est.estimate(util, qi, budget_B=budget_B)
    wall = time.time() - t0

    top1_item_correct = bool(attr.top_idx[0] == k_star)

    # modality top-1 — only meaningful if k* was picked into the inner game
    top1_mod_correct: bool | None = None
    psi_at_kstar = None
    if k_star in attr.top_idx:
        modalities = list(qi.memory[k_star].modalities.keys())
        L_kstar = len(modalities)
        psi_k = attr.psi[k_star, :L_kstar]
        if not np.all(np.isnan(psi_k)):
            argmax_idx = int(np.nanargmax(psi_k))
            top1_mod_correct = bool(modalities[argmax_idx] == ell_star)
            psi_at_kstar = psi_k.tolist()

    return SampleResult(
        sample_id=sample_idx, K=K, L=L, budget_B=budget_B, seed=seed,
        k_star=k_star, ell_star=ell_star,
        U_full=float(attr.U_full), U_empty=float(attr.U_empty),
        phi_sum=float(attr.phi.sum()),
        conservation_residual=float(attr.conservation_residual),
        top1_item_correct=top1_item_correct,
        top1_mod_correct=top1_mod_correct,
        phi=attr.phi.tolist(),
        psi_at_kstar=psi_at_kstar,
        wall_clock_sec=wall,
        n_calls=util.n_calls,
    )


def summarize(results: list[SampleResult]) -> dict:
    arr = lambda key: np.array([getattr(r, key) for r in results])
    cons = arr("conservation_residual")
    item_acc = arr("top1_item_correct").mean()
    mod_results = [r.top1_mod_correct for r in results if r.top1_mod_correct is not None]
    mod_acc = float(np.mean(mod_results)) if mod_results else float("nan")
    mod_kstar_in_top = sum(r.top1_mod_correct is not None for r in results) / len(results)
    return {
        "n_samples": len(results),
        "K": results[0].K,
        "L": results[0].L,
        "budget_B": results[0].budget_B,
        "conservation_mean": float(cons.mean()),
        "conservation_max": float(cons.max()),
        "conservation_p90": float(np.percentile(cons, 90)),
        "top1_item_recovery": float(item_acc),
        "top1_mod_recovery_conditional": mod_acc,
        "kstar_in_top_n_inner_rate": float(mod_kstar_in_top),
        "wall_clock_total_sec": float(arr("wall_clock_sec").sum()),
        "wall_clock_per_sample_sec": float(arr("wall_clock_sec").mean()),
        "n_calls_per_sample": float(arr("n_calls").mean()),
    }


def main(K: int, L: int, n: int, budget_B: int, seed: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running V1+V2 sanity: K={K} L={L} n={n} budget_B={budget_B} seed={seed}")
    t_start = time.time()
    results = [run_one(i, K, L, budget_B, seed) for i in range(n)]
    total = time.time() - t_start

    summary = summarize(results)
    summary["wall_clock_wall_sec"] = total

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<35} {v:.4f}")
        else:
            print(f"  {k:<35} {v}")

    # Save artifacts
    (out_dir / "v1v2_summary.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / "v1v2_per_sample.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nWrote: {out_dir/'v1v2_summary.json'}")
    print(f"Wrote: {out_dir/'v1v2_per_sample.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--budget-B", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("results/sanity"))
    args = ap.parse_args()
    main(args.K, args.L, args.n, args.budget_B, args.seed, args.out)
