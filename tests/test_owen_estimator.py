"""Owen estimator sanity tests using MockGenerator (no GPU).

Verifies that conservation holds approximately on a constructed toy problem
where we know what attribution should look like.
"""
from __future__ import annotations

import numpy as np
import pytest

from modality_credit.estimator.owen import OwenEstimator
from modality_credit.generators.mock import MockGenerator
from modality_credit.masking.redaction import RedactionMasker
from modality_credit.types import MemoryItem, QueryInstance
from modality_credit.utility import StandardUtility
from modality_credit.verifiers.exact_match import ExactMatchVerifier


def _toy_instance() -> QueryInstance:
    """Build a 2-item × 2-modality instance with a deterministic decisive modality."""
    return QueryInstance(
        instance_id="toy_0",
        query="What is the object?",
        memory=[
            MemoryItem("m_0", {"text": "tomato", "audio": "irrelevant"}),
            MemoryItem("m_1", {"text": "irrelevant", "audio": "irrelevant"}),
        ],
        gold_answer="tomato",
    )


def _decisive_generator():
    """Returns 'tomato' iff the substring 'tomato' is present in context."""
    return MockGenerator(lambda q, ctx: "tomato" if "tomato" in ctx else "unknown")


def test_conservation_holds_approximately():
    qi = _toy_instance()
    gen = _decisive_generator()
    util = StandardUtility(gen, ExactMatchVerifier(), RedactionMasker())
    est = OwenEstimator(top_n_inner=2, seed=42)
    attr = est.estimate(util, qi, budget_B=64)
    assert attr.conservation_residual < 0.10, (
        f"residual={attr.conservation_residual:.4f}, "
        f"phi={attr.phi}, U_full={attr.U_full}, U_empty={attr.U_empty}"
    )


def test_top_idx_picks_decisive_item():
    qi = _toy_instance()
    gen = _decisive_generator()
    util = StandardUtility(gen, ExactMatchVerifier(), RedactionMasker())
    est = OwenEstimator(top_n_inner=1, seed=42)
    attr = est.estimate(util, qi, budget_B=64)
    assert attr.top_idx[0] == 0, f"top_idx={attr.top_idx}, phi={attr.phi}"
