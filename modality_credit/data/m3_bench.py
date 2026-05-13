"""M3-Bench loader.

Reads (Q, M, y*) tuples from a JSON file. Schema:

    [
      {
        "instance_id": "sample_0",
        "query": "What did Alice mention about the meeting?",
        "gold_answer": "Wednesday at 3pm",
        "memory": [
          {
            "item_id": "ep_42",
            "modalities": {
              "text": "Alice's caption: 'meeting wed 3pm'",
              "audio": "audio transcript ...",
              "scene": "scene metadata: office, daytime"
            }
          },
          ... (K items total)
        ]
      },
      ...
    ]

Real M3-Bench JSON lives at `vendor/m3_agent/data/m3_bench.json` after
running `scripts/00_setup_m3_agent.sh` (not yet implemented). For now, if
the file is absent, the loader falls back to a built-in synthetic dataset
useful for smoke-testing the pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from modality_credit.data.base import BaseDataset
from modality_credit.types import MemoryItem, Modality, QueryInstance

Split = Literal["train", "val", "test"]


class M3BenchDataset(BaseDataset):
    """Load M3-Bench QA + retrieved memory items, or fall back to synthetic.

    Args:
        vendor_root:     directory containing the M3-Bench JSON. Default
                         "vendor/m3_agent".
        split:           "train" / "val" / "test" — selects the JSON file
                         `vendor_root/data/m3_bench_{split}.json`. If the
                         single-file `m3_bench.json` exists, splits live as
                         keys within it.
        K:               truncate each instance's memory list to first K items.
        n:               cap the number of returned instances; None = all.
        use_synthetic_fallback: if the real file is missing, build a small
                         synthetic dataset for smoke testing.
    """

    def __init__(self, vendor_root: str | Path = "vendor/m3_agent",
                 split: Split = "val", K: int = 4, n: int | None = None,
                 use_synthetic_fallback: bool = True):
        self.root = Path(vendor_root)
        self.split = split
        self.K = K
        self.n = n
        self._entries: list[dict] = self._load_or_synthesize(use_synthetic_fallback)

    def _load_or_synthesize(self, allow_synth: bool) -> list[dict]:
        for candidate in [
            self.root / "data" / f"m3_bench_{self.split}.json",
            self.root / "data" / "m3_bench.json",
        ]:
            if candidate.exists():
                with open(candidate) as f:
                    raw = json.load(f)
                # `m3_bench.json` may be either a flat list or a split-keyed dict.
                if isinstance(raw, dict):
                    raw = raw.get(self.split, [])
                if self.n is not None:
                    raw = raw[: self.n]
                return raw
        if allow_synth:
            return _synthetic_entries(K=self.K, n=self.n or 10)
        raise FileNotFoundError(
            f"No M3-Bench data found under {self.root}/data and synthetic fallback is off"
        )

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> QueryInstance:
        entry = self._entries[idx]
        memory = []
        for raw_item in entry["memory"][: self.K]:
            modalities: dict[Modality, str] = {
                m: v for m, v in raw_item["modalities"].items()
            }
            memory.append(MemoryItem(
                item_id=raw_item["item_id"],
                modalities=modalities,  # type: ignore[arg-type]
                metadata=raw_item.get("metadata", {}),
            ))
        return QueryInstance(
            instance_id=entry["instance_id"],
            query=entry["query"],
            memory=memory,
            gold_answer=entry["gold_answer"],
            metadata=entry.get("metadata", {}),
        )


def _synthetic_entries(K: int, n: int) -> list[dict]:
    """Minimal smoke-test dataset (no vendored M3-Agent needed).

    Each instance: K items, one of which contains a unique fact in its text
    modality. Question asks for that fact. Other items are distractors.
    Vision modality omitted (would break RedactionMasker._black_frame_like).
    """
    facts = [
        ("Alice's birthday is April 12", "When is Alice's birthday?", "April 12"),
        ("The meeting is on Wednesday at 3pm", "When is the meeting?", "Wednesday at 3pm"),
        ("The vault password is 4815", "What is the vault password?", "4815"),
        ("The dog's name is Buster", "What is the dog called?", "Buster"),
        ("The package shipped via FedEx tracking 7Q9X", "What's the tracking number?", "7Q9X"),
        ("Project status is delayed by two weeks", "How delayed is the project?", "two weeks"),
        ("The new office is at 422 Elm Street", "Where is the new office?", "422 Elm Street"),
        ("The client paid $12,500", "How much did the client pay?", "$12,500"),
        ("The keyword for entry is dragonfly", "What's the entry keyword?", "dragonfly"),
        ("The lab's daily start time is 9:30", "When does the lab start?", "9:30"),
    ]
    entries = []
    for i in range(min(n, len(facts))):
        fact, query, answer = facts[i]
        decisive_k = i % K
        memory = []
        for k in range(K):
            if k == decisive_k:
                text = fact
                audio = f"Audio transcript: '{fact.lower()}'"
                scene = f"context: someone discussing this in conversation"
            else:
                text = f"Item {k} caption: unrelated background chatter"
                audio = f"Item {k} audio: ambient room noise"
                scene = f"Item {k} scene: office, midday"
            memory.append({
                "item_id": f"synth_{i}_ep_{k}",
                "modalities": {"text": text, "audio": audio, "scene": scene},
                "metadata": {"decisive": (k == decisive_k)},
            })
        entries.append({
            "instance_id": f"synth_{i}",
            "query": query,
            "gold_answer": answer,
            "memory": memory,
            "metadata": {"source": "synthetic"},
        })
    return entries


def build_memory_item(raw: dict) -> MemoryItem:
    """Convert a raw M3-Agent item record into our MemoryItem.

    For now this just lifts the modalities dict directly. Once the real
    M3-Agent vendor JSON schema is known, this function is the seam that
    bridges it to our types.
    """
    return MemoryItem(
        item_id=raw["item_id"],
        modalities=raw["modalities"],
        metadata=raw.get("metadata", {}),
    )
