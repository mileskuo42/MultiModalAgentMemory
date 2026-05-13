"""Unit tests for Algorithm 1 — closed-form sample allocation.

These tests don't need a GPU and run in <1s.
"""
from __future__ import annotations

import numpy as np
import pytest

from modality_credit.allocation import allocate_budget, variance_ratio_bound


def test_basic_invariants():
    N_out, N_in = allocate_budget(K=4, L_ks=[4, 4, 4, 4], B=500)
    assert N_out >= 10
    assert all(n >= 10 for n in N_in)
    # Sum (approximately) respects budget (floor may shrink by up to K+1)
    assert N_out + sum(N_in) <= 500


def test_ratio_matches_closed_form():
    # K=4, all L_k=4: N_out / N_in[0] = √(2^4) / √(2^4) = 1
    N_out, N_in = allocate_budget(K=4, L_ks=[4, 4, 4, 4], B=10_000)
    assert abs(N_out / N_in[0] - 1.0) < 0.05


def test_larger_L_gets_more_inner_samples():
    # If L_1 = 6 and L_2..L_4 = 2, inner_1 should be much larger
    N_out, N_in = allocate_budget(K=4, L_ks=[6, 2, 2, 2], B=10_000)
    assert N_in[0] > 3 * N_in[1]      # √(64)/√(4) = 4


def test_variance_ratio_bound_decreases_with_L():
    # Cor 2c: ratio shrinks fast with L_k
    r2 = variance_ratio_bound(K=4, L_ks=[2, 2, 2, 2])
    r4 = variance_ratio_bound(K=4, L_ks=[4, 4, 4, 4])
    r6 = variance_ratio_bound(K=4, L_ks=[6, 6, 6, 6])
    assert r6 < r4 < r2


def test_rejects_too_small_budget():
    with pytest.raises(AssertionError):
        allocate_budget(K=4, L_ks=[4, 4, 4, 4], B=10)
