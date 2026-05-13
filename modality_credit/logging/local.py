"""Local-JSON backend — for offline runs / debugging."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from modality_credit.logging.base import BaseLogger


class LocalLogger(BaseLogger):
    def __init__(self, output_dir: str | Path):
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.dir / "metrics.jsonl", "a")

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        record = {"_t": time.time(), "_step": step, **metrics}
        self._fp.write(json.dumps(record, default=str) + "\n")
        self._fp.flush()

    def save_artifact(self, path: str, name: str) -> None:
        shutil.copy(path, self.dir / f"{name}__{Path(path).name}")

    def finish(self) -> None:
        self._fp.close()
