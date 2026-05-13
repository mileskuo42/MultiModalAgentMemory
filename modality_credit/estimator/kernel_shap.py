"""Kernel SHAP outer game.

Reference: Lundberg & Lee 2017; ContextCite 2024 uses the same surrogate idea.

Why we hand-roll vs `shap.KernelExplainer`:
  - Their evaluator interface assumes vectorized inputs; our Utility is per-call.
  - We need access to `attention_logit` seeded coalition sampling (see seeding.py).
  - We want explicit control over the efficiency constraint Σ φ_k = U(M) − U(∅).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import comb

from modality_credit.protocols import Utility
from modality_credit.types import QueryInstance


def kernel_shap_outer(utility: Utility,
                      query_inst: QueryInstance,
                      N: int,
                      *,
                      seed: int = 42,
                      seed_attention: bool = False) -> NDArray[np.float64]:
    """Estimate item-level Shapley φ via weighted least squares over N coalitions.

    Returns: phi, shape (K,), satisfying Σ phi_k ≈ U(M) − U(∅).
    """
    K = query_inst.K
    rng = np.random.default_rng(seed)

    # 1. sample N coalitions (each a length-K bool mask)
    coalitions = (_attention_seeded_coalitions(query_inst, N, rng)
                  if seed_attention
                  else _shap_kernel_coalitions(K, N, rng))

    # 2. evaluate utility on each coalition (single forward pass each)
    u_vals = np.array([utility.evaluate(query_inst, c.tolist()) for c in coalitions])

    # 3. solve constrained WLS for phi
    u_empty = utility.evaluate(query_inst, [False] * K)
    u_full = utility.evaluate(query_inst, [True] * K)
    phi = _solve_constrained_wls(coalitions, u_vals, u_empty, u_full, K)
    return phi


def _shap_kernel_coalitions(K: int, N: int, rng: np.random.Generator) -> NDArray:
    """Sample coalitions with SHAP kernel weights π(|S|) ∝ (K-1) / (C(K,|S|) |S| (K-|S|))."""
    sizes = np.arange(1, K)
    w = (K - 1) / (comb(K, sizes) * sizes * (K - sizes))
    w /= w.sum()

    out = np.zeros((N, K), dtype=bool)
    for i in range(N):
        s = rng.choice(sizes, p=w)
        idx = rng.choice(K, size=int(s), replace=False)
        out[i, idx] = True
    return out


def _attention_seeded_coalitions(qi: QueryInstance, N: int,
                                 rng: np.random.Generator) -> NDArray:
    """Stub: seed coalitions using Qwen-VL attention logits."""
    # TODO(impl): extract attention from generator's last layer, normalize per-item,
    # bias coalition sampling toward high-attention items.
    raise NotImplementedError(
        "Attention-seeded coalitions require Generator.get_attention(). "
        "Implement after baseline KSHAP path is verified."
    )


def _solve_constrained_wls(coalitions: NDArray, u_vals: NDArray,
                           u_empty: float, u_full: float, K: int) -> NDArray:
    """Constrained least squares: minimize ||Xφ - (u - u_empty)||² s.t. Σ φ = u_full - u_empty."""
    X = coalitions.astype(float)
    y = u_vals - u_empty
    # Unconstrained solution
    phi, *_ = np.linalg.lstsq(X, y, rcond=None)
    # Project onto efficiency constraint
    correction = ((u_full - u_empty) - phi.sum()) / K
    return phi + correction
