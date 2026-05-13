"""Phase 0 audits — runs the four pre-pilot sanity checks on synthetic
multi-modal samples (text/audio/scene/vision; vision rendered as PIL).

Verdicts: PASS / WARN / FAIL per audit. Decision rule for proceeding to
Phase 1: all audits PASS or WARN; any FAIL halts the pilot.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/phase0_audits.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from modality_credit.audits.conservation_residual import ConservationResidualAudit
from modality_credit.audits.modality_leakage import ModalityLeakageAudit
from modality_credit.audits.positional_invariance import PositionalInvarianceAudit
from modality_credit.audits.u_empty_baseline import UEmptyBaselineAudit
from modality_credit.caching import CachedUtility
from modality_credit.data.synthetic import build_rotating_modality_dataset, render_text_image
from modality_credit.generators.qwen_vl import QwenVLGenerator
from modality_credit.masking.redaction import RedactionMasker
from modality_credit.types import MemoryItem, QueryInstance
from modality_credit.utility import StandardUtility
from modality_credit.verifiers.exact_match import ExactMatchVerifier


# build_audit_dataset is provided by modality_credit.data.synthetic
# as build_rotating_modality_dataset; alias here for backward-compat with the
# pre-promotion runner.
build_audit_dataset = build_rotating_modality_dataset


def main(n: int, K: int, audit_filter: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Building Phase 0 dataset (n={n}, K={K}, rotating decisive modality)...")
    samples = build_audit_dataset(n, K)
    for s in samples[:5]:
        print(f"      {s.instance_id}: decisive_mod={s.metadata['decisive_modality']}, gold={s.gold_answer!r}")
    if n > 5:
        print(f"      ... and {n-5} more")

    print(f"\n[2/4] Loading Qwen-2.5-VL-7B on GPU (bf16)...")
    t0 = time.time()
    gen = QwenVLGenerator(model_path="Qwen/Qwen2.5-VL-7B-Instruct",
                          device="cuda", dtype=torch.bfloat16)
    print(f"      loaded in {time.time()-t0:.1f}s")

    inner_util = StandardUtility(gen, ExactMatchVerifier(lower=True, strip_punct=True),
                                 RedactionMasker())
    util = CachedUtility(inner_util)

    # Audits to run. Conservation residual is heavy (calls OwenEstimator); skipped
    # by default — flag via --audit "all" or "conservation".
    all_audits = {
        "u_empty": UEmptyBaselineAudit(),
        "positional": PositionalInvarianceAudit(n_shuffles=3),
        "leakage": ModalityLeakageAudit(),
        "conservation": ConservationResidualAudit(budget_B=64),
    }
    if audit_filter == "fast":
        audits = {k: v for k, v in all_audits.items() if k != "conservation"}
    else:
        audits = all_audits

    print(f"\n[3/4] Running audits: {list(audits.keys())}")
    results = {}
    for name, audit in audits.items():
        print(f"\n  --- {name} ---")
        t = time.time()
        try:
            r = audit.run(util, samples)
            wall = time.time() - t
            verdict = r["verdict"]
            summary = r["summary"]
            print(f"  [{verdict}] {summary}  (wall {wall:.1f}s)")
            results[name] = {"verdict": verdict, "summary": summary,
                             "wall_clock_sec": wall,
                             "data": _trim_data(r["data"])}
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            results[name] = {"verdict": "ERROR", "summary": str(e)}

    # Overall decision
    verdicts = [v["verdict"] for v in results.values()]
    if "FAIL" in verdicts or "ERROR" in verdicts:
        decision = "HALT — fix failing audit before Phase 1"
    elif "WARN" in verdicts:
        decision = "PROCEED WITH CAUTION — note warning, run Phase 1 anyway"
    else:
        decision = "PROCEED to Phase 1"

    print(f"\n[4/4] === PHASE 0 DECISION ===")
    print(f"  {decision}")
    for name, r in results.items():
        print(f"  - {name:<14} {r['verdict']:<6} {r['summary']}")

    out = {"results": results, "decision": decision, "n_samples": n, "K": K}
    (out_dir / "phase0_audits.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote: {out_dir/'phase0_audits.json'}")


def _trim_data(d: dict) -> dict:
    """Strip large arrays from audit data dict to keep JSON small."""
    if not isinstance(d, dict):
        return d
    out = {}
    for k, v in d.items():
        if isinstance(v, list) and len(v) > 30:
            out[k] = {"_summary": f"<list of {len(v)} entries, trimmed>"}
        elif isinstance(v, dict):
            out[k] = _trim_data(v)
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--audit", choices=["fast", "all"], default="fast",
                    help="'fast' skips conservation (which runs Owen estimator per sample)")
    ap.add_argument("--out", type=Path, default=Path("results/phase0"))
    args = ap.parse_args()
    main(args.n, args.K, args.audit, args.out)
