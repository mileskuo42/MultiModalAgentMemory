"""Phase 1 — Toy pilot. Decisive go/no-go gate before full experiments.

Run:
    python scripts/02_toy_pilot.py --config-path ../configs --config-name phase1_toy_pilot

What this script does:
  1. Build pipeline (Generator, Verifier, Masker, Utility, Estimator).
  2. Load 30 M3-Bench samples at K=4.
  3. Run pre-pilot audits (modality leakage, U(∅) baseline, positional invariance).
  4. If any audit FAILS → halt and report.
  5. Run Owen attribution on all 30 samples.
  6. Compute three Phase 1 monitors and a verdict.
  7. Log everything to WandB + local.
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

# TODO(impl): full wiring of these — kept minimal here for skeleton clarity
from modality_credit.audits.modality_leakage import ModalityLeakageAudit
from modality_credit.audits.positional_invariance import PositionalInvarianceAudit
from modality_credit.audits.u_empty_baseline import UEmptyBaselineAudit
from modality_credit.data.m3_bench import M3BenchDataset
from modality_credit.estimator.owen import OwenEstimator
from modality_credit.logging.local import LocalLogger


def _build_utility(cfg):
    """TODO(impl): build StandardUtility from cfg (Generator + Verifier + Masker)."""
    raise NotImplementedError


def _run_audits(utility, samples, logger):
    audits = [
        ModalityLeakageAudit(),
        UEmptyBaselineAudit(),
        PositionalInvarianceAudit(),
    ]
    any_fail = False
    for a in audits:
        result = a.run(utility, samples)
        logger.log({f"audit/{a.name}/verdict": result["verdict"],
                    f"audit/{a.name}/summary": result["summary"]})
        if result["verdict"] == "FAIL":
            any_fail = True
            print(f"❌ {a.name}: {result['summary']}")
        else:
            print(f"✅ {a.name}: {result['summary']}")
    return not any_fail


def _phase1_monitors(samples, attrs, K, budget_B):
    """Return dict with Phase 1 decisive metrics."""
    phi_stack = np.stack([a.phi for a in attrs])  # (N, K)
    var_phi = float(phi_stack.var(axis=0).mean())
    bound = (2 ** K) / (4 * budget_B)
    return {
        "var_phi_empirical": var_phi,
        "var_phi_bound": bound,
        "var_ratio": var_phi / bound,
        "conservation_residual_mean": float(np.mean([a.conservation_residual for a in attrs])),
        # TODO(impl): compute Spearman across 5 seeds for ψ̂
        "spearman_5seeds": float("nan"),
    }


def _verdict(monitors):
    """Phase 1 decision tree."""
    if monitors["var_ratio"] > 3 or monitors["conservation_residual_mean"] > 0.10:
        return "FAIL — pivot to backup direction (CMA for write-policy)"
    if monitors["var_ratio"] > 1.5 or monitors["conservation_residual_mean"] > 0.05:
        return "WARN — increase N or check implementation; re-run before Phase 2"
    return "PASS — proceed to Phase 2"


@hydra.main(version_base=None, config_path="../configs", config_name="phase1_toy_pilot")
def main(cfg: DictConfig) -> None:
    logger = LocalLogger(cfg.phase.output_dir)
    utility = _build_utility(cfg)
    dataset = M3BenchDataset(split="val", K=cfg.data.K, n=cfg.data.n)
    samples = list(dataset)

    if not _run_audits(utility, samples, logger):
        logger.log({"phase1/status": "halted_at_audit"})
        sys.exit(1)

    estimator = OwenEstimator(top_n_inner=cfg.estimator.top_n_inner,
                              seed=cfg.estimator.seed)
    attrs = [estimator.estimate(utility, s, budget_B=cfg.phase.budget_B) for s in samples]

    monitors = _phase1_monitors(samples, attrs, K=cfg.data.K, budget_B=cfg.phase.budget_B)
    verdict = _verdict(monitors)
    logger.log({"phase1/verdict": verdict, **{f"phase1/{k}": v for k, v in monitors.items()}})
    print(f"\n=== PHASE 1 VERDICT ===\n{verdict}\n")


if __name__ == "__main__":
    main()
