"""Sliced Isotropic Gaussian Regularizer (RFC-0003 §D6).

Projects a batch of embeddings onto K random unit directions; for each
direction, penalizes deviation from N(0, 1/d). Reference: LeWorldModel
`module.py:SIGReg`. The single hyperparameter `lambda` is applied by the caller
in the loss, not here.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn


class SIGReg(nn.Module):
    """Sliced Isotropic Gaussian Regularizer (RFC-0003 §D6).

    For each batch of embeddings:

    1. Sample K=256 random unit vectors (resampled every step).
    2. Project embeddings onto each unit vector -> (B, K) projections.
    3. For each slice: compute mean and variance.
    4. Loss = mean(|var_k - 1/d| + |mean_k|) where d = embed_dim.

    Single hyperparameter λ: applied by the caller, not here.
    """

    def __init__(self, embed_dim: int = 512, n_slices: int = 256) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_slices = n_slices
        # Target variance is 1/d for N(0, 1/d) per axis. Stored as a buffer so
        # it follows .to(device) calls.
        self.register_buffer(
            "target_variance",
            torch.tensor(1.0 / embed_dim, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """(B, D) -> scalar loss tensor.

        Resamples K random unit directions each call. Uses an unbiased
        variance estimator (correction=1) per slice.
        """
        if embeddings.ndim != 2:
            raise ValueError(
                f"SIGReg expects (B, D) embeddings, got shape {tuple(embeddings.shape)}"
            )
        if embeddings.shape[-1] != self.embed_dim:
            raise ValueError(
                f"SIGReg expected embed_dim={self.embed_dim}, "
                f"got embeddings of dim {embeddings.shape[-1]}"
            )

        # Sample random directions on the unit sphere (resampled each step
        # per RFC-0003 §D6: "refreshed each step").
        directions = torch.randn(
            self.n_slices,
            self.embed_dim,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        directions = F.normalize(directions, p=2, dim=-1)  # (K, D)

        # Project: (B, D) @ (D, K) -> (B, K).
        projections = embeddings @ directions.t()

        # Per-slice mean and variance.
        # correction=1 -> unbiased (Bessel-corrected) variance.
        # If B < 2 we fall back to correction=0 to avoid NaN.
        if projections.shape[0] >= 2:
            variances = projections.var(dim=0, unbiased=True)  # (K,)
        else:
            variances = projections.var(dim=0, unbiased=False)
        means = projections.mean(dim=0)  # (K,)

        target_var = cast(torch.Tensor, self.target_variance).to(dtype=variances.dtype)
        per_slice_loss = (variances - target_var).abs() + means.abs()
        return per_slice_loss.mean()


__all__ = ["SIGReg"]
