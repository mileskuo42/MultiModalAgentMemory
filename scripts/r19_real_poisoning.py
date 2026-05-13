"""R19 · Real-data memory poisoning detection on MMVet images.

Drops the toy synthetic. Uses real MMVet samples (real photos, real
factual questions, real answers). Builds memory with K=4 items where
the decisive evidence is in item 0's REAL image. For half the samples,
item 1 is poisoned with a misleading caption (text says wrong answer
while vision shows an unrelated image).

Then runs Owen attribution and asks: can the signed ψ distinguish
poisoned samples from clean ones? Reports per-sample signals and AUROC.

This is the killer use case for paper Phase 4A: drop-one ablation
cannot output negative attribution to a misleading modality, but
Owen value can — and the sign + magnitude is the detection signal.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/r19_real_poisoning.py
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import roc_auc_score

from modality_credit.caching import CachedUtility
from modality_credit.data.synthetic import render_text_image
from modality_credit.estimator.owen import OwenEstimator
from modality_credit.generators.qwen_vl import QwenVLGenerator
from modality_credit.masking.redaction import RedactionMasker
from modality_credit.types import MemoryItem, QueryInstance
from modality_credit.utility import StandardUtility
from modality_credit.verifiers.exact_match import ExactMatchVerifier


def _resize(img: Image.Image, max_dim: int = 448) -> Image.Image:
    """Resize image so the longest side <= max_dim. Preserves aspect."""
    if max(img.size) <= max_dim:
        return img.convert("RGB") if img.mode != "RGB" else img
    scale = max_dim / max(img.size)
    new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
    return img.resize(new_size, Image.LANCZOS).convert("RGB")


def filter_simple_samples(ds, max_answer_len: int = 25, limit: int | None = None):
    """Filter MMVet to samples with simple single-answer questions."""
    out = []
    for s in ds:
        a = s["answer"]
        if len(a) <= max_answer_len and "<" not in a:
            out.append(s)
            if limit is not None and len(out) >= limit:
                break
    return out


def build_sample(idx: int, mmvet_sample: dict, distractor_pool: list[dict],
                 is_poisoned: bool, rng: random.Random, K: int = 4) -> QueryInstance:
    """K items. item 0 has the real evidence image. item 1 is poisoned (if flag).
    items 2, 3 are unrelated distractor images.
    """
    # Pick distractors that are different from this sample
    distractors = [d for d in distractor_pool if d["question_id"] != mmvet_sample["question_id"]]
    chosen = rng.sample(distractors, K - 1)
    target_image = _resize(mmvet_sample["image"])
    real_answer = mmvet_sample["answer"]
    # For poisoning, pick a wrong answer from a different sample
    wrong_answer = chosen[0]["answer"]
    while wrong_answer == real_answer:
        wrong_answer = rng.choice(distractors)["answer"]

    memory = []
    # item 0: the honest evidence (real image)
    memory.append(MemoryItem(
        item_id=f"r19_{idx}_evidence",
        modalities={
            "vision": target_image,
            "text": "(no caption)",
            "audio": "(no audio)",
            "scene": "(no scene info)",
        },
        metadata={"role": "evidence"},
    ))
    # item 1: possibly poisoned — unrelated image + caption that may contradict
    poison_image = _resize(chosen[0]["image"])
    if is_poisoned:
        poison_caption = f"caption: the answer is {wrong_answer}"
    else:
        # clean: still has unrelated image but no misleading caption
        poison_caption = "(no caption)"
    memory.append(MemoryItem(
        item_id=f"r19_{idx}_poisonable",
        modalities={
            "vision": poison_image,
            "text": poison_caption,
            "audio": "(no audio)",
            "scene": "(no scene info)",
        },
        metadata={"role": "poisonable", "is_poisoned": is_poisoned,
                  "claimed_answer": wrong_answer if is_poisoned else None},
    ))
    # items 2, 3: pure distractors (other unrelated MMVet images)
    for k in range(2, K):
        dimg = _resize(chosen[k - 1]["image"])
        memory.append(MemoryItem(
            item_id=f"r19_{idx}_distractor_{k}",
            modalities={
                "vision": dimg,
                "text": "(no caption)",
                "audio": "(no audio)",
                "scene": "(no scene info)",
            },
            metadata={"role": "distractor"},
        ))

    return QueryInstance(
        instance_id=f"r19_{'poisoned' if is_poisoned else 'clean'}_{idx}",
        query=mmvet_sample["question"],
        memory=memory,
        gold_answer=real_answer,
        metadata={"is_poisoned": is_poisoned, "real_answer": real_answer,
                  "wrong_answer": wrong_answer if is_poisoned else None,
                  "question_id": mmvet_sample["question_id"]},
    )


def detection_signal(attr) -> dict:
    """Compute several poisoning detection signals from an AttributionResult."""
    # Look at item 1 (the poisonable item) specifically. In real attack we
    # don't know which item is poisoned, but for proof of concept assume we
    # do; later we extend to "any item with negative ψ" signal.
    K = attr.phi.shape[0]
    psi = attr.psi  # (K, max_L)
    phi = attr.phi
    # Per-item signed ψ inconsistency (only valid where ψ has values)
    per_item_signals = []
    for k in range(K):
        psi_k = psi[k]
        valid = ~np.isnan(psi_k)
        if valid.sum() < 2:
            per_item_signals.append({"min_psi": np.nan, "max_psi": np.nan, "range": np.nan})
            continue
        psi_valid = psi_k[valid]
        per_item_signals.append({
            "min_psi": float(psi_valid.min()),
            "max_psi": float(psi_valid.max()),
            "range": float(psi_valid.max() - psi_valid.min()),
        })
    # Overall signals
    psi_flat = psi[~np.isnan(psi)]
    return {
        "min_psi_global": float(psi_flat.min()) if psi_flat.size > 0 else np.nan,
        "max_phi_minus_mean": float(phi.max() - phi.mean()),
        "per_item": per_item_signals,
        "phi": [float(x) for x in phi],
    }


def main(n_per_class: int, K: int, budget_B: int, seed: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    print(f"[1/5] Loading MMVet from HF cache (218 samples)...")
    t0 = time.time()
    ds = load_dataset("lmms-lab/MMVet", split="test")
    pool = filter_simple_samples(ds, max_answer_len=25)
    rng.shuffle(pool)
    print(f"      {len(pool)} samples with simple factual answers")
    print(f"      data loaded in {time.time()-t0:.1f}s")

    # Build n_per_class clean + n_per_class poisoned
    chosen = pool[: 2 * n_per_class]
    samples = []
    for i, mmvet in enumerate(chosen):
        is_poisoned = i >= n_per_class  # first half clean, second half poisoned
        samples.append(build_sample(i, mmvet, pool, is_poisoned, rng, K=K))
    print(f"      built {len(samples)} samples ({n_per_class} clean + {n_per_class} poisoned)")
    for s in samples[:3] + samples[-3:]:
        print(f"      {s.instance_id}: Q='{s.query[:50]}...' gold='{s.gold_answer}' "
              f"poisoned={s.metadata['is_poisoned']} wrong='{s.metadata.get('wrong_answer')}'")

    print(f"\n[2/5] Loading Qwen-2.5-VL-7B on GPU (bf16)...")
    t0 = time.time()
    gen = QwenVLGenerator(model_path="Qwen/Qwen2.5-VL-7B-Instruct",
                          device="cuda", dtype=torch.bfloat16, seed=seed)
    print(f"      loaded in {time.time()-t0:.1f}s")
    inner_util = StandardUtility(gen, ExactMatchVerifier(lower=True, strip_punct=True),
                                 RedactionMasker())
    util = CachedUtility(inner_util)

    print(f"\n[3/5] Running Owen attribution (B={budget_B}, top_n_inner=2)...")
    est = OwenEstimator(top_n_inner=2, seed=seed)
    rows = []
    t_run = time.time()
    for i, qi in enumerate(samples):
        t = time.time()
        attr = est.estimate(util, qi, budget_B=budget_B)
        sig = detection_signal(attr)
        rows.append({
            "instance_id": qi.instance_id,
            "is_poisoned": qi.metadata["is_poisoned"],
            "real_answer": qi.metadata["real_answer"],
            "wrong_answer": qi.metadata.get("wrong_answer"),
            "U_full": float(attr.U_full),
            "U_empty": float(attr.U_empty),
            "phi": sig["phi"],
            "top_idx": [int(x) for x in attr.top_idx],
            "conservation_residual": float(attr.conservation_residual),
            "min_psi_global": sig["min_psi_global"],
            "max_phi_minus_mean": sig["max_phi_minus_mean"],
            "per_item": sig["per_item"],
            "wall_clock_sec": time.time() - t,
        })
        flag = "POISONED" if qi.metadata["is_poisoned"] else "clean   "
        print(f"      [{i+1}/{len(samples)}] {flag} {qi.instance_id}: U_full={attr.U_full:.0f} "
              f"min_psi={sig['min_psi_global']:+.3f} cons={attr.conservation_residual:.3f} "
              f"phi={[f'{x:+.2f}' for x in attr.phi]} wall={time.time()-t:.1f}s")
    wall_total = time.time() - t_run

    print(f"\n[4/5] Detection AUROC analysis...")
    labels = [int(r["is_poisoned"]) for r in rows]
    # Try several detection signals
    signals = {
        "neg_min_psi_global": [-r["min_psi_global"] if not np.isnan(r["min_psi_global"]) else 0 for r in rows],
        "max_phi_minus_mean": [r["max_phi_minus_mean"] for r in rows],
        "U_full_failed": [1 - r["U_full"] for r in rows],  # poisoned more likely to fail
    }
    auroc_results = {}
    for name, sig_vals in signals.items():
        try:
            auroc = roc_auc_score(labels, sig_vals)
            auroc_results[name] = float(auroc)
        except Exception as e:
            auroc_results[name] = f"ERROR: {e}"

    # Also: at item 1 specifically (the poisonable item), report mean min_psi for poisoned vs clean
    item1_min_psi_poisoned = []
    item1_min_psi_clean = []
    for r in rows:
        v = r["per_item"][1]["min_psi"]
        if np.isnan(v):
            continue
        (item1_min_psi_poisoned if r["is_poisoned"] else item1_min_psi_clean).append(v)

    print(f"\n=== R19 POISONING DETECTION ON REAL MMVET ===")
    print(f"  n={len(rows)}, {sum(labels)} poisoned, {len(labels)-sum(labels)} clean")
    print(f"  U_full(clean):     mean={np.mean([r['U_full'] for r in rows if not r['is_poisoned']]):.2f}")
    print(f"  U_full(poisoned):  mean={np.mean([r['U_full'] for r in rows if r['is_poisoned']]):.2f}")
    print(f"  conservation residual mean: {np.mean([r['conservation_residual'] for r in rows]):.4f}")
    print(f"\n  Item-1 (poisonable) ψ_min stats:")
    print(f"    clean    n={len(item1_min_psi_clean)} mean min_psi={np.mean(item1_min_psi_clean) if item1_min_psi_clean else float('nan'):.3f}")
    print(f"    poisoned n={len(item1_min_psi_poisoned)} mean min_psi={np.mean(item1_min_psi_poisoned) if item1_min_psi_poisoned else float('nan'):.3f}")
    print(f"\n  Detection AUROC by signal:")
    for name, a in auroc_results.items():
        print(f"    {name:<25} AUROC = {a if isinstance(a, str) else f'{a:.3f}'}")
    print(f"\n  wall-clock total: {wall_total:.1f}s ({wall_total/len(samples):.1f}s per sample)")

    summary = {
        "n_per_class": n_per_class, "K": K, "budget_B": budget_B,
        "wall_clock_total_sec": wall_total,
        "u_full_clean_mean": float(np.mean([r["U_full"] for r in rows if not r["is_poisoned"]])),
        "u_full_poisoned_mean": float(np.mean([r["U_full"] for r in rows if r["is_poisoned"]])),
        "auroc": auroc_results,
        "item1_min_psi_clean_mean": float(np.mean(item1_min_psi_clean)) if item1_min_psi_clean else None,
        "item1_min_psi_poisoned_mean": float(np.mean(item1_min_psi_poisoned)) if item1_min_psi_poisoned else None,
    }
    (out_dir / "r19_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_dir / "r19_per_sample.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\n  Wrote: {out_dir/'r19_summary.json'}")
    print(f"  Wrote: {out_dir/'r19_per_sample.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="samples per class (clean / poisoned)")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("results/r19_real_poisoning"))
    args = ap.parse_args()
    main(args.n, args.K, args.B, args.seed, args.out)
