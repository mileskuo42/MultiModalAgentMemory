"""Abstract protocols for all pluggable framework components.

Why Protocols (PEP 544) instead of ABCs:
  - Structural typing: third-party classes (e.g., HuggingFace pipelines) can
    satisfy our protocols without inheriting.
  - Implementations stay decoupled from this file.

Convention:
  - Every protocol method has a complete docstring including its contract
    (preconditions, postconditions, what may NOT change).
  - Implementations live in their own subpackage (generators/, estimators/, etc.)
    and may also subclass a `BaseX` helper if shared logic is useful.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable, Any

from modality_credit.types import (
    AttributionResult, ItemMask, ModalityMask, MemoryItem, PrunedContext,
    QueryInstance, PhiArray, PsiArray,
)


# ---------------------------------------------------------------------------
# Generator + Verifier: define what U(S) actually returns
# ---------------------------------------------------------------------------

@runtime_checkable
class Generator(Protocol):
    """A frozen multimodal generator (e.g., Qwen-2.5-VL-7B).

    Contract:
      - Stateless across calls (no hidden cache that affects determinism).
      - `generate(query, context_str)` must be reproducible given the same
        inputs and a fixed seed at __init__.
    """

    def generate(self, query: str, context_str: str, *,
                 max_new_tokens: int = 128) -> str:
        """Run one forward pass; return the decoded answer string."""
        ...

    @property
    def name(self) -> str:
        """Short identifier used in WandB / file paths."""
        ...


@runtime_checkable
class Verifier(Protocol):
    """Judge whether generator output matches gold answer.

    Returns: True (correct) or False. We treat this as the {0,1}-valued
    `u(y, y*)` in the Shapley framework.
    """

    def __call__(self, generated: str, gold: str) -> bool: ...


# ---------------------------------------------------------------------------
# Utility: the central abstraction U(S)
# ---------------------------------------------------------------------------

@runtime_checkable
class Utility(Protocol):
    """U(S) = P(model 答对 | Q, S) under masking.

    Contract:
      - `evaluate(qi, item_mask, modality_masks)` returns a float in [0, 1].
        With a deterministic Verifier this is in {0, 1}; with an LLM-judge
        it may be continuous.
      - Calls with identical arguments MUST return identical values
        (use Caching to enforce when wrapping a stochastic generator).
      - `evaluate` is the only "expensive" call in the framework; everything
        else is O(K * L_k) bookkeeping.
    """

    def evaluate(self,
                 query_inst: QueryInstance,
                 item_mask: ItemMask,
                 modality_masks: list[ModalityMask] | None = None) -> float:
        ...

    @property
    def n_calls(self) -> int:
        """Total forward passes so far. For accounting only."""
        ...


# ---------------------------------------------------------------------------
# Masker: convert (memory, masks) -> context string fed to the generator
# ---------------------------------------------------------------------------

@runtime_checkable
class Masker(Protocol):
    """Materialize a context string from masked memory.

    Different strategies (redaction / removal / black-frame) MUST satisfy
    the leakage audit (audits.modality_leakage). Implementations should
    document their assumed leakage profile.
    """

    def apply(self,
              memory: list[MemoryItem],
              item_mask: ItemMask,
              modality_masks: list[ModalityMask] | None = None) -> str:
        ...


# ---------------------------------------------------------------------------
# Estimator: produces AttributionResult from (Utility, QueryInstance)
# ---------------------------------------------------------------------------

@runtime_checkable
class Estimator(Protocol):
    """Compute attribution scores. Subsumes Owen, flat Shapley, item-only,
    attention rollout, LLM self-report, etc.

    Contract:
      - MUST set `phi` (shape (K,)) and `conservation_residual`.
      - SHOULD set `psi` (shape (K, max_L)) when computing item × modality;
        otherwise return a NaN-filled array.
      - MAY use `budget_B` as a hint; ignore for non-Shapley methods.
    """

    def estimate(self,
                 utility: Utility,
                 query_inst: QueryInstance,
                 budget_B: int = 500) -> AttributionResult:
        ...

    @property
    def name(self) -> str:
        ...


# ---------------------------------------------------------------------------
# Pruner: convert AttributionResult -> PrunedContext
# ---------------------------------------------------------------------------

@runtime_checkable
class Pruner(Protocol):
    """Decide which (item × modality) patches to keep, given attribution."""

    def prune(self,
              query_inst: QueryInstance,
              attribution: AttributionResult,
              **kwargs: Any) -> PrunedContext:
        ...

    @property
    def name(self) -> str: ...


# ---------------------------------------------------------------------------
# Attack: produces a modified QueryInstance (for Phase 4A poisoning detection)
# ---------------------------------------------------------------------------

@runtime_checkable
class Attack(Protocol):
    """Adversarial perturbation of one memory item's one modality.

    Returns: new QueryInstance with `metadata["attack"] = self.name` and
    `metadata["attack_target"] = (item_idx, modality)`.
    """

    def apply(self,
              query_inst: QueryInstance,
              target_item_idx: int,
              target_modality: str | None = None) -> QueryInstance:
        ...

    @property
    def name(self) -> str: ...


# ---------------------------------------------------------------------------
# Audit: pre-pilot sanity check
# ---------------------------------------------------------------------------

@runtime_checkable
class Audit(Protocol):
    """A sanity check run before Phase 1. Each audit produces a structured
    result that includes a clear PASS / WARN / FAIL verdict."""

    def run(self,
            utility: Utility,
            samples: list[QueryInstance]) -> dict[str, Any]:
        """Returns dict containing at minimum: {"verdict": "PASS"|"WARN"|"FAIL",
        "summary": str, "data": pd.DataFrame | dict}."""
        ...

    @property
    def name(self) -> str: ...


# ---------------------------------------------------------------------------
# Dataset: load (Q, M, y*) tuples from a benchmark
# ---------------------------------------------------------------------------

@runtime_checkable
class Dataset(Protocol):
    """Stream of QueryInstance — abstracts M3-Bench / WorldMM / LongVideoBench /
    synthetic / adversarial. Must be iterable and indexable."""

    def __iter__(self) -> "Dataset": ...
    def __next__(self) -> QueryInstance: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> QueryInstance: ...

    @property
    def name(self) -> str: ...


# ---------------------------------------------------------------------------
# Logger: pluggable WandB / local / mock
# ---------------------------------------------------------------------------

@runtime_checkable
class Logger(Protocol):
    """Backend-agnostic metric logger."""

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None: ...
    def save_artifact(self, path: str, name: str) -> None: ...
    def finish(self) -> None: ...
