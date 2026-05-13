# modality_credit

Owen-value attribution for multimodal agent memory.

## For implementation agents: start here

Read [`docs/README.md`](docs/README.md) first — it links to all design documents
in the correct order. The minimum context to start coding is:
1. [`docs/proposal_brief.md`](docs/proposal_brief.md) — 5-minute overview
2. [`docs/experimental_framework.md`](docs/experimental_framework.md) — what to implement and why
3. [`modality_credit/protocols.py`](modality_credit/protocols.py) — the contract you must satisfy

## Repo layout

```
modality_credit/        # package
  types.py              # shared dataclasses (QueryInstance, MemoryItem, AttributionResult)
  protocols.py          # ALL pluggable interfaces (Generator, Utility, Estimator, ...)
  utility.py            # StandardUtility — wires Generator + Verifier + Masker
  caching.py            # CachedUtility wrapper (memo + disk)
  allocation.py         # Algorithm 1 — closed-form sample allocation
  pipeline.py           # Top-level orchestrator

  generators/           # Concrete Generator implementations
    base.py
    qwen_vl.py          # Qwen-2.5-VL-7B (primary)
    mock.py             # Deterministic, for tests

  verifiers/            # Output verifiers
    exact_match.py
    llm_judge.py

  masking/              # Context-construction strategies
    base.py
    redaction.py        # DEFAULT

  estimator/            # Attribution algorithms
    base.py
    owen.py             # Our method
    kernel_shap.py      # Outer game (called by owen.py)
    mc_permutation.py   # Inner game (called by owen.py)
    flat_shapley.py     # Baseline B1
    item_only.py        # Baseline B2

  pruning/              # Context-pruning strategies
    base.py
    owen_threshold.py   # Algorithm 3 (our main empirical method)
    random.py           # Critical control baseline

  adversarial/          # Phase 4A poisoning attacks
    base.py
    caption_flip.py
    audio_noise.py
    frame_swap.py
    entity_swap.py

  audits/               # Pre-pilot sanity checks (PHASE 0)
    base.py
    modality_leakage.py
    u_empty_baseline.py
    positional_invariance.py
    conservation_residual.py

  metrics/              # Reporting metrics
    separability.py     # Claim 1 detection
    pareto.py           # Claim 2 retention-acc curve
    detection_auroc.py  # Phase 4A
    variance_ratio.py   # Claim 1.5

  data/                 # Dataset loaders
    base.py
    m3_bench.py
    longvideobench.py
    synthetic.py        # Claim 1.5 GT-injected data

  logging/              # Pluggable loggers
    base.py
    local.py
    wandb_logger.py

scripts/                # Phase-by-phase entry points
configs/                # Hydra configs
tests/                  # Unit tests (no GPU)
results/                # Experiment outputs
```

## How the abstractions compose

```
QueryInstance + Utility
        ↓
   Estimator  →  AttributionResult
        ↓
    Pruner    →  PrunedContext
        ↓
   Utility    →  accuracy
```

Every arrow is mediated by a Protocol — swap any component without touching others.

## To implement (priority order)

1. `generators/qwen_vl.py`        — required for everything
2. `utility.py::StandardUtility.evaluate`   — central abstraction
3. `data/m3_bench.py`              — required for Phase 1
4. `estimator/kernel_shap.py::_attention_seeded_coalitions` — Claim 1.1
5. `estimator/flat_shapley.py`     — Baseline B1
6. `pruning/random.py`              — Claim 2 control
7. `audits/modality_leakage.py::run` — Pre-pilot
8. `adversarial/*.py`               — Phase 4A
9. `data/synthetic.py`              — Claim 1.5
10. `data/longvideobench.py`        — Phase 4B

## How to run

```bash
# Install
pip install -e .[test,qwen]

# Tests (no GPU needed)
pytest tests/

# Phase 1 toy pilot
python scripts/02_toy_pilot.py
```

## Design principles

- **Every "expensive" call goes through `Utility.evaluate()`.** Cache liberally.
- **Protocols over inheritance.** Subclass only when sharing real logic.
- **Type hints everywhere.** This codebase will be read by reviewers.
- **One file per responsibility.** Easy to swap, easy to test in isolation.
