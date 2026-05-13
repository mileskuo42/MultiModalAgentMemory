"""Masking sanity tests — no GPU."""
from __future__ import annotations

from modality_credit.masking.redaction import RedactionMasker
from modality_credit.types import MemoryItem


def test_item_mask_removes_item():
    items = [MemoryItem("a", {"text": "alpha"}), MemoryItem("b", {"text": "beta"})]
    ctx = RedactionMasker().apply(items, item_mask=[True, False])
    assert "alpha" in ctx
    assert "beta" not in ctx


def test_modality_mask_redacts_content():
    items = [MemoryItem("a", {"text": "alpha", "audio": "ringtone"})]
    ctx = RedactionMasker().apply(items, item_mask=[True],
                                  modality_masks=[{"text": True, "audio": False}])
    assert "alpha" in ctx
    assert "ringtone" not in ctx
    assert "[audio redacted]" in ctx
