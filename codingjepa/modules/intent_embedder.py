"""Intent embedder (RFC-0003 §D5).

Maps discrete intent indices `{0, ..., 7, NONE}` to embeddings of size
`embed_dim`. Index 8 is `[I_NONE]` and is used during unconditional pretraining.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class IntentEmbedder(nn.Module):
    """nn.Embedding(9, embed_dim): maps intent index to embedding.

    Indices 0-7 correspond to the 8 refactor intents.
    Index 8 is `[I_NONE]` (used during unconditional pretraining).

    Total parameters: ``N_INTENTS * embed_dim`` (~4.6k at embed_dim=512).
    """

    NONE_IDX: int = 8
    N_INTENTS: int = 9

    def __init__(self, embed_dim: int = 512) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(self.N_INTENTS, embed_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, intent_idx: torch.Tensor) -> torch.Tensor:
        """(B,) int64 -> (B, embed_dim)."""
        return cast(torch.Tensor, self.embedding(intent_idx))

    def none_embedding(
        self,
        batch_size: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Return (B, embed_dim) all-NONE embeddings for unconditional pretraining."""
        target_device = device if device is not None else self.embedding.weight.device
        idx = torch.full(
            (batch_size,),
            fill_value=self.NONE_IDX,
            dtype=torch.long,
            device=target_device,
        )
        return cast(torch.Tensor, self.embedding(idx))


__all__ = ["IntentEmbedder"]
