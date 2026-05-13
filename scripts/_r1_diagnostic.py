"""R1 diagnostic: print Owen estimator outputs on the toy instance.

Not a test — quick numeric print so we can see what's actually happening.
"""
from modality_credit.estimator.owen import OwenEstimator
from modality_credit.generators.mock import MockGenerator
from modality_credit.masking.redaction import RedactionMasker
from modality_credit.types import MemoryItem, QueryInstance
from modality_credit.utility import StandardUtility
from modality_credit.verifiers.exact_match import ExactMatchVerifier


def toy(K=2, L=2, decisive_k=0, decisive_modality="text", marker="tomato"):
    """Build a K-item × L-modality instance, marker is only in (decisive_k, decisive_modality).

    Uses only string-content modalities (text/audio/scene), max L=3. We avoid the
    "vision" modality because RedactionMasker._black_frame_like expects PIL/Tensor,
    not strings — fine for real data, not for CPU sanity checks.
    """
    mods = ["text", "audio", "scene"][:L]
    assert L <= 3, "Synthetic harness supports L<=3 (text/audio/scene); use real data for vision"
    mem = []
    for k in range(K):
        modalities = {}
        for m in mods:
            if k == decisive_k and m == decisive_modality:
                modalities[m] = marker
            else:
                modalities[m] = "irrelevant"
        mem.append(MemoryItem(f"m_{k}", modalities))
    return QueryInstance(
        instance_id=f"toy_K{K}L{L}",
        query="What is the object?",
        memory=mem,
        gold_answer=marker,
    )


def run(K, L, budget_B, seed=42):
    qi = toy(K=K, L=L)
    gen = MockGenerator(lambda q, ctx: "tomato" if "tomato" in ctx else "unknown")
    util = StandardUtility(gen, ExactMatchVerifier(), RedactionMasker())
    est = OwenEstimator(top_n_inner=min(2, K), seed=seed)
    attr = est.estimate(util, qi, budget_B=budget_B)
    return attr, util


if __name__ == "__main__":
    print(f"{'config':<24} {'U_full':>7} {'U_empty':>8} {'Σφ':>7} {'residual':>9} {'top_idx':>10} {'n_calls':>8}")
    for K, L, B in [(2, 2, 64), (3, 3, 128), (4, 3, 256), (4, 3, 500), (6, 3, 500)]:
        attr, util = run(K, L, B)
        print(
            f"K={K},L={L},B={B:<6} "
            f"{attr.U_full:>7.3f} {attr.U_empty:>8.3f} "
            f"{attr.phi.sum():>7.3f} {attr.conservation_residual:>9.4f} "
            f"{str(list(attr.top_idx)):>10} {util.n_calls:>8}"
        )
