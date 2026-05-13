"""Pydantic config schemas. Hydra-friendly via `_target_` discriminators."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class GeneratorCfg(BaseModel):
    target: str = Field(..., alias="_target_")
    model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    device: str = "cuda"


class VerifierCfg(BaseModel):
    target: str = Field(..., alias="_target_")
    kind: Literal["exact_match", "llm_judge"] = "exact_match"


class MaskerCfg(BaseModel):
    target: str = Field(..., alias="_target_")
    strategy: Literal["redaction", "removal", "black_frame"] = "redaction"


class EstimatorCfg(BaseModel):
    target: str = Field(..., alias="_target_")
    top_n_inner: int = 2
    seed_attention: bool = False
    seed: int = 42


class DataCfg(BaseModel):
    name: Literal["m3_bench", "longvideobench", "worldmm", "mma", "synthetic", "adversarial"]
    split: str = "val"
    K: int = 4
    n: int | None = 100


class PhaseCfg(BaseModel):
    phase: str
    budget_B: int = 500
    tau_grid: list[tuple[float, float]] = [(0.05, 0.05), (0.1, 0.1), (0.2, 0.2)]
    output_dir: str = "results/"
    wandb_project: str = "modality_credit"


class FullCfg(BaseModel):
    """Top-level config consumed by scripts/."""
    generator: GeneratorCfg
    verifier: VerifierCfg
    masker: MaskerCfg
    estimator: EstimatorCfg
    data: DataCfg
    phase: PhaseCfg
    seed: int = 42
