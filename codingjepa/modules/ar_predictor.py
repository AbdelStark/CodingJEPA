"""Autoregressive transformer predictor over chunk embeddings (RFC-0003 §D3).

The predictor takes a history of `H` context chunk embeddings together with a
parallel sequence of action / intent embeddings, sums the action embeddings
position-wise onto the context (same convention as LeWorldModel's
``predictor(emb, act_emb)``), runs the result through a small transformer
stack, and returns the last ``n_preds`` positions as predicted embeddings.

Implements issue #60.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class ARPredictor(nn.Module):
    """Autoregressive transformer predictor over chunk embeddings (RFC-0003 §D3).

    Takes a history of ``H`` context embeddings (with action / intent embeddings
    summed in), runs them through a 4-layer pre-norm transformer, and outputs
    ``n_preds`` predicted embeddings (the last ``n_preds`` positions).

    The history is always length ``H`` and fully visible, so no causal mask is
    required. Action embeddings are added onto the context BEFORE the
    transformer (same as LeWM convention).

    Parameters
    ----------
    embed_dim:
        Embedding dimension. Default ``512``.
    n_layers:
        Transformer encoder layers. Default ``4``.
    n_heads:
        Attention heads. Default ``8``.
    ffn_dim:
        Feed-forward hidden dimension. Default ``2048``.
    dropout:
        Dropout rate inside the transformer. Default ``0.1``.
    history_len:
        Number of context steps ``H``. Default ``3``.
    n_preds:
        Number of predicted steps to emit. Default ``1``.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        history_len: int = 3,
        n_preds: int = 1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.history_len = history_len
        self.n_preds = n_preds

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Final LayerNorm is standard for pre-norm transformers.
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        ctx_emb: torch.Tensor,
        act_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``(B, n_preds, D)`` predicted embeddings.

        Parameters
        ----------
        ctx_emb:
            Context embeddings of shape ``(B, H, D)``.
        act_emb:
            Action / intent embeddings of shape ``(B, H, D)``. Summed
            position-wise onto ``ctx_emb`` before the transformer.
        """
        # Sum action embeddings onto context embeddings (LeWM convention).
        x = ctx_emb + act_emb
        x = cast(torch.Tensor, self.transformer(x))
        # Return the last ``n_preds`` positions.
        return x[:, -self.n_preds :, :]


__all__ = ["ARPredictor"]
