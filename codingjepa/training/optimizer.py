"""AdamW + LinearWarmupCosineAnnealingLR per RFC-0008 §D4 (issue #69).

Implements the optimizer factories used by the training pipeline:

* :func:`build_optimizer` — AdamW with the RFC-0008 defaults
  (``lr=3e-4``, ``weight_decay=0.05``, ``betas=(0.9, 0.95)``, ``eps=1e-8``).
* :func:`build_scheduler` — Linear warmup for ``warmup_steps`` followed by
  cosine decay to ``min_lr``. Implemented from scratch with
  :class:`torch.optim.lr_scheduler.LambdaLR` so the schedule is fully
  inspectable from the unit tests.
"""

from __future__ import annotations

import math

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(
    model: torch.nn.Module,
    *,
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> AdamW:
    """Build the AdamW optimizer per RFC-0008 §D4.

    Parameters
    ----------
    model:
        Any ``nn.Module``; ``model.parameters()`` is fed straight into AdamW.
    lr:
        Base learning rate. Default 3e-4 per RFC-0008.
    weight_decay:
        Weight-decay coefficient. Default 0.05 per RFC-0008.
    betas:
        AdamW exponential-decay betas. Default ``(0.9, 0.95)`` per RFC-0008.
    eps:
        AdamW numerical-stability constant. Default 1e-8 per RFC-0008.
    """
    return AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )


def build_scheduler(
    optimizer: AdamW,
    *,
    warmup_steps: int = 5_000,
    total_steps: int = 200_000,
    min_lr: float = 1e-5,
) -> LambdaLR:
    """Build a Linear-warmup + Cosine-annealing scheduler per RFC-0008 §D4.

    Returns a :class:`LambdaLR` whose lambda implements:

    * step in ``[0, warmup_steps)`` → linear ramp from ``0`` to ``base_lr``.
    * step in ``[warmup_steps, total_steps]`` → cosine decay from ``base_lr``
      to ``min_lr``.

    Parameters
    ----------
    optimizer:
        The optimizer whose ``param_groups[0]['lr']`` provides ``base_lr``.
    warmup_steps:
        Number of linear-warmup steps. Default 5,000 per RFC-0008.
    total_steps:
        Total number of optimizer steps over which the schedule is defined.
        After ``total_steps`` the LR clamps to ``min_lr`` (the lambda returns
        ``min_scale`` for any further step).
    min_lr:
        Floor learning rate reached at ``step == total_steps``. Default 1e-5.
    """
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
    if total_steps <= warmup_steps:
        raise ValueError(f"total_steps ({total_steps}) must be > warmup_steps ({warmup_steps})")

    base_lr = float(optimizer.param_groups[0]["lr"])
    if base_lr <= 0:
        raise ValueError(f"optimizer base lr must be positive, got {base_lr}")
    min_scale = float(min_lr) / base_lr

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            # Linear warmup from 0 to base_lr.
            return float(step) / float(warmup_steps)
        # Cosine decay from base_lr to min_lr over (total_steps - warmup_steps).
        denom = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / denom)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return float(min_scale + (1 - min_scale) * cosine)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


__all__ = ["build_optimizer", "build_scheduler"]
