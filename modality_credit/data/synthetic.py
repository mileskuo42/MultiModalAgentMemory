"""Synthetic generator for Claim 1.5 (theory validation).

Injects ground-truth decisive (item k*, modality ℓ*) so we can measure
attribution recall + variance ratio against a known answer.
"""
from __future__ import annotations

import random
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from modality_credit.data.base import BaseDataset
from modality_credit.types import MemoryItem, Modality, QueryInstance


def render_text_image(text: str, size: tuple[int, int] = (448, 224),
                      bg_color: str = "white", fg_color: str = "black",
                      font_size: int = 42) -> Image.Image:
    """Render text centered on a colored background. Used to inject decisive
    facts into the 'vision' modality of synthetic samples."""
    img = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(img)
    font = None
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            font = ImageFont.truetype(path, size=font_size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        tw, th = draw.textsize(text, font=font)
    draw.text(((size[0] - tw) / 2, (size[1] - th) / 2 - 4), text, fill=fg_color, font=font)
    return img


def build_rotating_modality_dataset(n: int, K: int) -> list[QueryInstance]:
    """Build n samples with all 4 modalities; decisive modality rotates across
    {vision, text, audio, scene} so attribution audits / pilots see every
    modality act as the decisive one on some subset of samples.

    Vision-decisive samples have the fact rendered into the image; other-modality-
    decisive samples have distractor images.
    """
    sample_specs = [
        ("4815", "What is the code?", "4815"),
        ("Buster", "What is the dog's name?", "Buster"),
        ("9:30 AM", "When is the meeting?", "9:30 AM"),
        ("dragonfly", "What is the keyword?", "dragonfly"),
        ("Wednesday", "What day?", "Wednesday"),
        ("$12500", "How much?", "$12500"),
        ("FedEx", "Which carrier?", "FedEx"),
        ("April 12", "When is Alice's birthday?", "April 12"),
        ("422 Elm", "What address?", "422 Elm"),
        ("8 hours", "How long?", "8 hours"),
    ]
    decisive_modalities = ["vision", "text", "audio", "scene"]
    out: list[QueryInstance] = []
    for i in range(n):
        fact, q, gold = sample_specs[i % len(sample_specs)]
        decisive_mod = decisive_modalities[i % len(decisive_modalities)]
        decisive_k = i % K
        mem: list[MemoryItem] = []
        for k in range(K):
            modalities: dict[Modality, object] = {
                "vision": render_text_image("DISTRACTOR"),
                "text": "caption: routine background description",
                "audio": "audio transcript: ambient room noise",
                "scene": "scene: office, midday",
            }
            if k == decisive_k:
                if decisive_mod == "vision":
                    modalities["vision"] = render_text_image(fact)
                elif decisive_mod == "text":
                    modalities["text"] = f"caption: the answer is {fact}"
                elif decisive_mod == "audio":
                    modalities["audio"] = f"audio transcript: the answer is {fact}"
                elif decisive_mod == "scene":
                    modalities["scene"] = f"scene: the answer is {fact}"
            mem.append(MemoryItem(
                item_id=f"rot_{i}_ep_{k}",
                modalities=modalities,
                metadata={"decisive": k == decisive_k, "decisive_mod": decisive_mod},
            ))
        out.append(QueryInstance(
            instance_id=f"rot_{i}_{decisive_mod}",
            query=q, memory=mem, gold_answer=fact,
            metadata={"decisive_modality": decisive_mod},
        ))
    return out


class SyntheticDataset(BaseDataset):
    """K × L_k synthetic QA with injected decisive modality."""

    def __init__(self, K: int, L_k: int, n: int = 50,
                 modality_set: list[Modality] | None = None,
                 seed: int = 42):
        self.K = K
        self.L_k = L_k
        self.n = n
        self.modality_set = modality_set or ["vision", "text", "audio"][:L_k]
        self.seed = seed
        self._cache: list[QueryInstance] = []
        self._ground_truth: list[tuple[int, Modality]] = []
        # TODO(impl): pre-generate all `n` samples eagerly for reproducibility.
        raise NotImplementedError

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> QueryInstance:
        return self._cache[idx]

    def ground_truth(self, idx: int) -> tuple[int, Modality]:
        """Return injected (k*, ℓ*) for sample idx."""
        return self._ground_truth[idx]

    @property
    def name(self) -> str:
        return f"synthetic_K{self.K}_L{self.L_k}"
