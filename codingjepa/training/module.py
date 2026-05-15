"""LeWM-style training module for CodingJEPA (issue #66, RFC-0008 §D3).

Wraps a :class:`codingjepa.model.CodingJEPA` with:

* an optimizer (typically :class:`torch.optim.AdamW`),
* an LR scheduler (typically a LambdaLR per :mod:`codingjepa.training.optimizer`),
* gradient clipping (RFC-0008 §D4),
* optional mixed-precision training in bf16 (RFC-0008 §D4).

The public surface is intentionally narrow: :meth:`training_step` and
:meth:`validation_step` consume a minibatch and return a metrics dict, while
:attr:`global_step` and :attr:`current_lr` expose schedule state for callbacks
and loggers.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:  # pragma: no cover - import-cycle / type-only imports
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    from codingjepa.model import CodingJEPA


def _resolve_device(device: torch.device | None) -> torch.device:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TrainingModule(nn.Module):
    """Training wrapper for CodingJEPA (RFC-0008 §D3).

    Wraps the :class:`CodingJEPA` model with the training-time loss
    computation, gradient clipping, and mixed-precision support.

    Parameters
    ----------
    model:
        The CodingJEPA network being trained.
    optimizer:
        AdamW optimizer (or any optimizer with a compatible interface).
    scheduler:
        LR scheduler stepped once per :meth:`training_step` call.
    grad_clip:
        Global L2 gradient-norm clip. Default 1.0 (RFC-0008 §D4).
    use_amp:
        If ``True``, wrap the forward pass in ``torch.amp.autocast`` with
        ``dtype=torch.bfloat16``. Default ``True``. We do not use
        ``torch.cuda.amp.GradScaler`` because bf16 does not require it.
    device:
        Device on which the model lives. Default: ``cuda`` if available,
        else ``cpu``.
    """

    def __init__(
        self,
        model: CodingJEPA,
        optimizer: AdamW,
        scheduler: LambdaLR,
        *,
        grad_clip: float = 1.0,
        use_amp: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.device = _resolve_device(device)
        self.model.to(self.device)
        self._global_step = 0

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    @property
    def global_step(self) -> int:
        """Number of completed optimizer steps."""
        return self._global_step

    @property
    def current_lr(self) -> float:
        """Current learning rate (from ``optimizer.param_groups[0]``)."""
        return float(self.optimizer.param_groups[0]["lr"])

    # ------------------------------------------------------------------
    # Training / validation steps
    # ------------------------------------------------------------------

    def training_step(
        self,
        token_ids: torch.Tensor,
        intent_idx: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Run one training step. Returns a metrics dict.

        Logged metrics (RFC-0008 §D12): ``loss``, ``pred_loss``,
        ``sigreg_loss``, ``grad_norm``, ``lr``.
        """
        self.model.train()
        token_ids = token_ids.to(self.device)
        intent_idx = intent_idx.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        # Mixed precision: only when device supports it (CUDA bf16).
        amp_enabled = self.use_amp and self.device.type == "cuda"
        if amp_enabled:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):  # type: ignore[attr-defined]
                result = self.model(token_ids, intent_idx, attention_mask)
        else:
            result = self.model(token_ids, intent_idx, attention_mask)

        loss = result.loss
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.grad_clip,
        )

        self.optimizer.step()
        self.scheduler.step()
        self._global_step += 1

        grad_norm_value = float(grad_norm.detach().item())
        # Guard against rare NaN/Inf surfaces produced by upstream autograd:
        # we still want to report a finite number to downstream callbacks.
        if not math.isfinite(grad_norm_value):
            grad_norm_value = float("inf")

        return {
            "loss": float(loss.detach().item()),
            "pred_loss": float(result.pred_loss.detach().item()),
            "sigreg_loss": float(result.sigreg_loss.detach().item()),
            "grad_norm": grad_norm_value,
            "lr": self.current_lr,
        }

    @torch.no_grad()
    def validation_step(
        self,
        token_ids: torch.Tensor,
        intent_idx: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Run one validation step (no grad). Returns a metrics dict."""
        self.model.eval()
        token_ids = token_ids.to(self.device)
        intent_idx = intent_idx.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        amp_enabled = self.use_amp and self.device.type == "cuda"
        if amp_enabled:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):  # type: ignore[attr-defined]
                result = self.model(token_ids, intent_idx, attention_mask)
        else:
            result = self.model(token_ids, intent_idx, attention_mask)

        return {
            "loss": float(result.loss.detach().item()),
            "pred_loss": float(result.pred_loss.detach().item()),
            "sigreg_loss": float(result.sigreg_loss.detach().item()),
        }


__all__ = ["TrainingModule"]
