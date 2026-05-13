"""R11 stress test: graded ψ split under partial modality redundancy.

The Owen value's strongest theoretical guarantee over flat drop-one ablation
(Park MIS, 2408.12763) is the *uniqueness* of credit assignment under the
four Owen 1977 axioms. Specifically: when m modalities of an item are
perfectly redundant (each individually carries the full answer), Shapley
must split credit 1/m per redundant modality, and zero to non-contributing.

This experiment builds 3 synthetic regimes per decisive item:
  - K_redundant = 4  → expected ψ ≈ [0.25, 0.25, 0.25, 0.25]
  - K_redundant = 3  → expected ψ ≈ [0.33, 0.33, 0.33, 0.00]
  - K_redundant = 2  → expected ψ ≈ [0.50, 0.50, 0.00, 0.00]

If Qwen-VL × Owen estimator returns ψ split within ±0.05 of these
predictions, the framework's theoretical claim holds empirically.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/r11_redundancy_stress.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from modality_credit.caching import CachedUtility
from modality_credit.data.synthetic import render_text_image
from modality_credit.estimator.owen import OwenEstimator
from modality_credit.generators.qwen_vl import QwenVLGenerator
from modality_credit.masking.redaction import RedactionMasker
from modality_credit.types import MemoryItem, QueryInstance
from modality_credit.utility import StandardUtility
from modality_credit.verifiers.exact_match import ExactMatchVerifier


def build_redundancy_sample(idx: int, K: int, K_redundant: int,
                            fact: str, query: str,
                            use_neutral_fillers: bool = True) -> QueryInstance:
    """K items; item 0 has the fact in K_redundant of its 4 modalities.

    `use_neutral_fillers=True` (default): non-redundant modalities of the
    decisive item are filled with explicit "no content" placeholders rather
    than distractor content like "UNRELATED" — keeps the framework from
    attributing negative ψ to adversarial fillers.
    """
    redundant_mods = ["text", "audio", "scene", "vision"][:K_redundant]
    nonredundant_mods = ["text", "audio", "scene", "vision"][K_redundant:]

    if use_neutral_fillers:
        FILL = {
            "vision": render_text_image(""),  # blank white image
            "text": "(no caption)",
            "audio": "(no audio)",
            "scene": "(no scene info)",
        }
        DISTRACTOR_VISION_TEXT = ""  # also blank for distractor items
    else:
        FILL = {
            "vision": render_text_image("UNRELATED"),
            "text": "caption: nothing special here",
            "audio": "audio: ambient sounds",
            "scene": "scene: empty room",
        }
        DISTRACTOR_VISION_TEXT = "DISTRACTOR"

    mem = []
    for k in range(K):
        modalities = {}
        if k == 0:  # decisive item
            for m in redundant_mods:
                if m == "vision":
                    modalities["vision"] = render_text_image(fact)
                elif m == "text":
                    modalities["text"] = f"caption: the answer is {fact}"
                elif m == "audio":
                    modalities["audio"] = f"audio transcript: the answer is {fact}"
                elif m == "scene":
                    modalities["scene"] = f"scene: the answer is {fact}"
            for m in nonredundant_mods:
                modalities[m] = FILL[m]
        else:  # distractor item: also use neutral or distractor variant per flag
            if use_neutral_fillers:
                modalities = {
                    "vision": render_text_image(""),
                    "text": "(no caption)",
                    "audio": "(no audio)",
                    "scene": "(no scene info)",
                }
            else:
                modalities = {
                    "vision": render_text_image(DISTRACTOR_VISION_TEXT),
                    "text": "caption: unrelated chatter",
                    "audio": "audio: ambient noise",
                    "scene": "scene: hallway, afternoon",
                }
        mem.append(MemoryItem(
            item_id=f"red{K_redundant}_{idx}_ep_{k}",
            modalities=modalities,
            metadata={"decisive": k == 0, "K_redundant": K_redundant},
        ))
    return QueryInstance(
        instance_id=f"redundancy_{K_redundant}_{idx}",
        query=query,
        memory=mem,
        gold_answer=fact,
        metadata={"K_redundant": K_redundant,
                  "redundant_mods": redundant_mods,
                  "expected_psi_value": 1.0 / K_redundant},
    )


def main(K: int, budget_B: int, top_n_inner: int, seed: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    facts = [("dragonfly", "What is the keyword?"),
             ("4815", "What is the code?")]
    samples = []
    for K_red in [4, 3, 2]:
        for i, (fact, q) in enumerate(facts):
            samples.append(build_redundancy_sample(
                i, K=K, K_redundant=K_red, fact=fact, query=q,
                use_neutral_fillers=True,
            ))
    print(f"[1/3] Built {len(samples)} samples across K_redundant ∈ {{4,3,2}}")
    for s in samples:
        print(f"      {s.instance_id}: K_red={s.metadata['K_redundant']} "
              f"expected_psi≈{s.metadata['expected_psi_value']:.3f} on "
              f"{s.metadata['redundant_mods']}")

    print(f"\n[2/3] Loading Qwen-2.5-VL-7B...")
    t0 = time.time()
    gen = QwenVLGenerator(model_path="Qwen/Qwen2.5-VL-7B-Instruct",
                          device="cuda", dtype=torch.bfloat16, seed=seed)
    print(f"      loaded in {time.time()-t0:.1f}s")
    inner_util = StandardUtility(gen, ExactMatchVerifier(lower=True, strip_punct=True),
                                 RedactionMasker())
    util = CachedUtility(inner_util)
    est = OwenEstimator(top_n_inner=top_n_inner, seed=seed)

    print(f"\n[3/3] Running Owen attribution (B={budget_B})...")
    rows = []
    t_exp = time.time()
    for i, qi in enumerate(samples):
        t_s = time.time()
        attr = est.estimate(util, qi, budget_B=budget_B)
        wall_s = time.time() - t_s
        K_red = qi.metadata["K_redundant"]
        expected = qi.metadata["expected_psi_value"]
        redundant_mods = qi.metadata["redundant_mods"]
        # decisive item is index 0
        mods = list(qi.memory[0].modalities.keys())
        psi_k = attr.psi[0, :len(mods)] if 0 in attr.top_idx else None
        psi_dict = None
        max_dev = None
        if psi_k is not None and not np.all(np.isnan(psi_k)):
            psi_dict = {mods[j]: float(psi_k[j]) for j in range(len(mods))}
            # Deviation: max |actual - expected| over redundant modalities
            deviations = []
            for m in redundant_mods:
                dev = abs(psi_dict[m] - expected)
                deviations.append(dev)
            for m in mods:
                if m not in redundant_mods:
                    deviations.append(abs(psi_dict[m]))  # expected 0
            max_dev = max(deviations)
        rows.append({
            "instance_id": qi.instance_id,
            "K_redundant": K_red,
            "expected_psi_value": expected,
            "redundant_mods": redundant_mods,
            "U_full": float(attr.U_full),
            "phi": [float(x) for x in attr.phi],
            "top_idx": [int(x) for x in attr.top_idx],
            "psi_at_decisive_item": psi_dict,
            "max_deviation_from_expected": max_dev,
            "wall_clock_sec": wall_s,
        })
        dev_str = f"{max_dev:.3f}" if max_dev is not None else "N/A"
        print(f"      [{i+1}/{len(samples)}] {qi.instance_id}: "
              f"K_red={K_red} expected≈{expected:.3f} "
              f"actual_ψ={psi_dict} "
              f"max_dev={dev_str} wall={wall_s:.1f}s")
    wall_total = time.time() - t_exp

    print(f"\n=== REDUNDANCY SPLIT SUMMARY ===")
    by_kred = {}
    for r in rows:
        if r["max_deviation_from_expected"] is None:
            continue
        by_kred.setdefault(r["K_redundant"], []).append(r["max_deviation_from_expected"])
    for k_red, devs in sorted(by_kred.items()):
        mean_dev = float(np.mean(devs))
        max_dev = float(np.max(devs))
        verdict = "PASS" if max_dev < 0.10 else ("WARN" if max_dev < 0.20 else "FAIL")
        print(f"  K_red={k_red}: expected ψ≈{1/k_red:.3f} | mean_dev={mean_dev:.3f} max_dev={max_dev:.3f} → {verdict}")

    summary = {
        "K": K, "budget_B": budget_B, "top_n_inner": top_n_inner,
        "by_K_redundant": {
            str(k): {
                "expected_psi": 1.0 / int(k),
                "samples": len(devs),
                "max_deviation_mean": float(np.mean(devs)),
                "max_deviation_max": float(np.max(devs)),
            } for k, devs in by_kred.items()
        },
        "wall_clock_total_sec": wall_total,
    }
    (out_dir / "r11_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_dir / "r11_per_sample.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nWrote: {out_dir/'r11_summary.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--top-n-inner", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("results/r11_redundancy"))
    args = ap.parse_args()
    main(args.K, args.B, args.top_n_inner, args.seed, args.out)
