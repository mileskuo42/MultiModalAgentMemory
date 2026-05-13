"""R8 vision-routing experiment: same Owen attribution as mini_owen_qwen.py
but with the decisive answer rendered into a PIL image (OCR-style). Text and
audio modalities carry only distractors. If the model answers correctly and
ψ_vision dominates, vision routing is working.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/mini_owen_vision.py
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


def build_sample(idx: int, K: int, decisive_k: int,
                 fact_in_image: str, query: str, gold_answer: str) -> QueryInstance:
    """K items, decisive item has the fact ONLY in its rendered vision image.
    Other modalities of the decisive item are distractors."""
    mem = []
    for k in range(K):
        if k == decisive_k:
            modalities = {
                "vision": render_text_image(fact_in_image),
                "text": "caption: a generic kitchen scene, no specific info",
                "audio": "audio: ambient sounds, no spoken content",
                "scene": "scene: indoor kitchen, midday",
            }
        else:
            modalities = {
                "vision": render_text_image(f"DISTRACTOR {k}"),
                "text": f"caption {k}: routine background description",
                "audio": f"audio {k}: ambient room noise",
                "scene": f"scene {k}: office, afternoon",
            }
        mem.append(MemoryItem(
            item_id=f"v{idx}_ep_{k}",
            modalities=modalities,
            metadata={"decisive": k == decisive_k},
        ))
    return QueryInstance(
        instance_id=f"vision_sample_{idx}",
        query=query,
        memory=mem,
        gold_answer=gold_answer,
    )


def main(n: int, K: int, budget_B: int, top_n_inner: int, seed: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3 vision-decisive samples (fact only readable from the image)
    sample_specs = [
        ("The code is 4815", "What is the code?", "4815"),
        ("Buster is the dog", "What is the dog's name?", "Buster"),
        ("Meeting at 9:30 AM", "When is the meeting?", "9:30 AM"),
    ]
    samples = []
    for i, (fact_img, q, gold) in enumerate(sample_specs[:n]):
        decisive_k = i % K
        samples.append(build_sample(i, K=K, decisive_k=decisive_k,
                                    fact_in_image=fact_img,
                                    query=q, gold_answer=gold))

    print(f"[1/4] Built {len(samples)} vision-decisive samples (K={K})")
    for s in samples:
        gt_k = next(i for i, m in enumerate(s.memory) if m.metadata["decisive"])
        print(f"      {s.instance_id}: gt_item={gt_k}, query={s.query!r}, gold={s.gold_answer!r}")

    print(f"\n[2/4] Loading Qwen-2.5-VL-7B on GPU (bf16)...")
    t0 = time.time()
    gen = QwenVLGenerator(model_path="Qwen/Qwen2.5-VL-7B-Instruct",
                          device="cuda", dtype=torch.bfloat16, seed=seed)
    print(f"      loaded in {time.time()-t0:.1f}s; device={gen._model.device}")

    inner_util = StandardUtility(gen, ExactMatchVerifier(lower=True, strip_punct=True),
                                 RedactionMasker())
    util = CachedUtility(inner_util)

    # Pre-flight: does the model actually answer correctly with full memory?
    print("\n[3a/4] Pre-flight: does the model READ the image?")
    for qi in samples:
        gt_k = next(i for i, m in enumerate(qi.memory) if m.metadata["decisive"])
        ctx, imgs = RedactionMasker().apply(qi.memory, item_mask=[True] * qi.K)
        t = time.time()
        out = gen.generate(qi.query, ctx, images=imgs)
        print(f"      {qi.instance_id}: gold={qi.gold_answer!r} → gen={out!r}  ({time.time()-t:.1f}s)")

    print("\n[3b/4] Running Owen attribution...")
    est = OwenEstimator(top_n_inner=top_n_inner, seed=seed)
    rows = []
    t_exp = time.time()
    for i, qi in enumerate(samples):
        t_s = time.time()
        attr = est.estimate(util, qi, budget_B=budget_B)
        wall_s = time.time() - t_s
        gt_k = next(idx for idx, m in enumerate(qi.memory) if m.metadata["decisive"])
        mods = list(qi.memory[gt_k].modalities.keys())
        psi_k = attr.psi[gt_k, :len(mods)] if gt_k in attr.top_idx else None
        psi_dict = None
        if psi_k is not None and not np.all(np.isnan(psi_k)):
            psi_dict = {mods[j]: float(psi_k[j]) for j in range(len(mods))}

        row = {
            "instance_id": qi.instance_id,
            "query": qi.query,
            "gold": qi.gold_answer,
            "phi": [float(x) for x in attr.phi],
            "top_idx": [int(x) for x in attr.top_idx],
            "gt_item_idx": gt_k,
            "top1_item_correct": int(attr.top_idx[0]) == gt_k,
            "psi_at_gt_item": psi_dict,
            "psi_argmax_modality": (
                mods[int(np.nanargmax(psi_k))] if psi_k is not None and not np.all(np.isnan(psi_k))
                else None
            ),
            "U_full": float(attr.U_full),
            "U_empty": float(attr.U_empty),
            "conservation_residual": float(attr.conservation_residual),
            "wall_clock_sec": wall_s,
        }
        rows.append(row)
        print(f"      [{i+1}/{len(samples)}] {qi.instance_id}: "
              f"phi={[f'{x:+.2f}' for x in attr.phi]} "
              f"top_idx={list(attr.top_idx)} (gt={gt_k}) "
              f"U_full={attr.U_full:.0f} psi_argmax={row['psi_argmax_modality']} "
              f"wall={wall_s:.1f}s")
    wall_total = time.time() - t_exp

    # Aggregate
    item_acc = float(np.mean([r["top1_item_correct"] for r in rows]))
    u_full_mean = float(np.mean([r["U_full"] for r in rows]))
    psi_argmax_vision_rate = float(np.mean(
        [r["psi_argmax_modality"] == "vision" for r in rows
         if r["psi_argmax_modality"] is not None]
    ))

    summary = {
        "n_samples": len(rows),
        "K": K,
        "budget_B": budget_B,
        "top_n_inner": top_n_inner,
        "u_full_mean": u_full_mean,
        "top1_item_recovery": item_acc,
        "psi_argmax_vision_rate": psi_argmax_vision_rate,
        "wall_clock_total_sec": wall_total,
        "wall_clock_per_sample_sec": wall_total / max(len(rows), 1),
    }

    print(f"\n[4/4] === SUMMARY ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<35} {v:.4f}")
        else:
            print(f"  {k:<35} {v}")
    print(f"\n  ψ argmax = vision rate: target ≥ 0.66 (most decisive credit goes to image)")

    (out_dir / "vision_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "vision_per_sample.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nWrote: {out_dir/'vision_summary.json'}")
    print(f"Wrote: {out_dir/'vision_per_sample.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--top-n-inner", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("results/mini_owen_vision"))
    args = ap.parse_args()
    main(args.n, args.K, args.B, args.top_n_inner, args.seed, args.out)
