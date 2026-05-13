"""WandB backend."""
from __future__ import annotations

from typing import Any

from modality_credit.logging.base import BaseLogger


class WandbLogger(BaseLogger):
    """Thin wrapper. Initialize a run at construction; clean up on finish()."""

    def __init__(self, project: str, name: str | None = None,
                 config: dict | None = None, tags: list[str] | None = None):
        import wandb
        self._wandb = wandb
        self._run = wandb.init(project=project, name=name, config=config, tags=tags)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self._run.log(metrics, step=step)

    def save_artifact(self, path: str, name: str) -> None:
        artifact = self._wandb.Artifact(name=name, type="dataset")
        artifact.add_file(path)
        self._run.log_artifact(artifact)

    def finish(self) -> None:
        self._run.finish()
