"""Predictor-output projector (RFC-0003 §D4).

Same architecture as `Projector` but distinct parameters. Applied after the
ARPredictor and before the prediction loss.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class PredProj(nn.Module):
    """Linear(512, 2048) -> BatchNorm1d -> ReLU -> Linear(2048, 512).

    Applied after the ARPredictor, before the prediction loss. ~1.6M parameters
    (actual: 2.1M including BatchNorm affine).

    Identical to `Projector` in shape; the two modules deliberately do **not**
    share parameters per RFC-0003 §D4.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 2048,
        output_dim: int = 512,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, D) -> (B, output_dim)."""
        h = self.fc1(x)
        h = self.bn(h)
        h = self.act(h)
        return cast(torch.Tensor, self.fc2(h))


__all__ = ["PredProj"]
