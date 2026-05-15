"""Top-level CodingJEPA module placeholder; real implementation lands in #64.

See RFC-0003 §D10 and docs/spec/02-public-api.md §codingjepa.model.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class CodingJEPA(nn.Module):
    """Top-level JEPA module mirroring LeWorldModel's `JEPA`. Placeholder.

    Composition is fixed: encoder + projector + ARPredictor + pred_proj +
    intent_embedder + sigreg. Methods raise NotImplementedError until #64
    lands.
    """

    def encode(self, chunk_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def predict(self, ctx_emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def embed(self, chunk_tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


__all__ = ["CodingJEPA"]
