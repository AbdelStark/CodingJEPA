"""Lightning-analog training manager for CodingJEPA (issue #67, RFC-0008).

The :class:`Manager` orchestrates the training loop:

* iterates batches from ``train_loader``,
* calls :meth:`TrainingModule.training_step` on each batch,
* steps the LR scheduler and reports metrics,
* dispatches the configured callbacks (logging, rank diagnostic, loss
  monotonicity, checkpointing),
* periodically calls :meth:`TrainingModule.validation_step` over the
  validation loader (when supplied).

The implementation is intentionally compact: no Lightning dependency, no
distributed training, no fault-tolerance — those are deferred per RFC-0008
§Deferred items.
"""

from __future__ import annotations

import os
import random
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:  # pragma: no cover - import-cycle / type-only imports
    from codingjepa.training.module import TrainingModule


class Manager:
    """Training manager for CodingJEPA (RFC-0008, Lightning analog).

    Parameters
    ----------
    module:
        The :class:`TrainingModule` wrapping the model + optimizer +
        scheduler.
    train_loader:
        DataLoader yielding batches with ``token_ids`` and ``intent_idx``
        tensors.
    val_loader:
        Optional DataLoader yielding validation batches.
    callbacks:
        Optional list of callback objects with a ``__call__`` or
        ``on_step_end``-shaped interface (we treat all callbacks
        polymorphically — see :meth:`_invoke_callbacks`).
    run_dir:
        Root directory for run artifacts. A subdirectory ``runs/<run_id>``
        is created if it does not already exist.
    run_id:
        Optional run identifier. A UUID is generated if not provided.
    log_every_n_steps:
        Cadence at which to print / log step metrics. Default 100.
    val_every_n_steps:
        Cadence at which to run validation. Default 1,000. Ignored when
        ``val_loader is None``.
    max_steps:
        Default maximum number of optimizer steps. Overridable by
        :meth:`fit`.
    seed:
        Global RNG seed for reproducibility (RFC-0008 §D9).
    """

    def __init__(
        self,
        module: TrainingModule,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        val_loader: DataLoader[dict[str, torch.Tensor]] | None = None,
        *,
        callbacks: list[Any] | None = None,
        run_dir: Path = Path("runs"),
        run_id: str | None = None,
        log_every_n_steps: int = 100,
        val_every_n_steps: int = 1_000,
        max_steps: int = 200_000,
        seed: int = 42,
    ) -> None:
        self.module = module
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.callbacks: list[Any] = list(callbacks or [])
        self.log_every_n_steps = log_every_n_steps
        self.val_every_n_steps = val_every_n_steps
        self.default_max_steps = max_steps
        self.seed = seed

        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.run_dir = Path(run_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._last_metrics: dict[str, float] = {}
        self._set_seeds(self.seed)

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def fit(self, max_steps: int | None = None) -> dict[str, float]:
        """Run the training loop. Returns the last observed metrics dict."""
        target_steps = max_steps if max_steps is not None else self.default_max_steps
        if target_steps <= 0:
            return {}

        start_time = time.monotonic()
        step = self.module.global_step
        target = step + target_steps

        # We use an infinite iterator so we can run more steps than
        # ``len(train_loader)`` in tiny-slice tests.
        loader_iter = iter(self.train_loader)
        last_metrics: dict[str, float] = {}

        while self.module.global_step < target:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.train_loader)
                batch = next(loader_iter)

            token_ids, intent_idx, attention_mask = self._unpack_batch(batch)
            metrics = self.module.training_step(token_ids, intent_idx, attention_mask)
            last_metrics = dict(metrics)
            current_step = self.module.global_step

            self._invoke_callbacks(
                step=current_step,
                metrics=metrics,
                phase="train",
            )

            if self.log_every_n_steps > 0 and current_step % self.log_every_n_steps == 0:
                self._log_step(current_step, metrics)

            if (
                self.val_loader is not None
                and self.val_every_n_steps > 0
                and current_step % self.val_every_n_steps == 0
            ):
                val_metrics = self._run_validation()
                last_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                self._invoke_callbacks(
                    step=current_step,
                    metrics=val_metrics,
                    phase="val",
                )

        elapsed = time.monotonic() - start_time
        last_metrics["wall_clock_s"] = float(elapsed)
        self._last_metrics = last_metrics
        return last_metrics

    @property
    def last_metrics(self) -> dict[str, float]:
        """The metrics dict from the most recent :meth:`fit` call."""
        return dict(self._last_metrics)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_seeds(self, seed: int) -> None:
        """Set global RNG seeds per RFC-0008 §D9."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ.setdefault("PYTHONHASHSEED", str(seed))

    def _unpack_batch(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        token_ids = batch["token_ids"]
        intent_idx = batch["intent_idx"]
        attention_mask = batch.get("attention_mask")
        return token_ids, intent_idx, attention_mask

    def _log_step(self, step: int, metrics: dict[str, float]) -> None:
        """Default per-step log: structured stdout for any non-WandB consumers."""
        payload = ", ".join(
            f"{k}={metrics[k]:.4f}" if isinstance(metrics[k], float) else f"{k}={metrics[k]}"
            for k in sorted(metrics)
        )
        print(f"[manager] step={step} {payload}")

    def _run_validation(self) -> dict[str, float]:
        """Iterate the entire val loader once and return averaged metrics."""
        if self.val_loader is None:
            return {}
        sums: dict[str, float] = {}
        n = 0
        for batch in self.val_loader:
            token_ids, intent_idx, attention_mask = self._unpack_batch(batch)
            metrics = self.module.validation_step(token_ids, intent_idx, attention_mask)
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            n += 1
        if n == 0:
            return {}
        return {k: v / n for k, v in sums.items()}

    def _invoke_callbacks(
        self,
        *,
        step: int,
        metrics: dict[str, float],
        phase: str,
    ) -> None:
        """Dispatch a step / val event to any callbacks with a matching hook."""
        for cb in self.callbacks:
            # WandB-style ``log``.
            if hasattr(cb, "log"):
                try:
                    cb.log(metrics, step=step)
                except TypeError:
                    # Some loggers may take a different signature; ignore.
                    pass
            # LossMonotonicity-style ``update``.
            if phase == "train" and "pred_loss" in metrics and hasattr(cb, "update"):
                try:
                    cb.update(step, float(metrics["pred_loss"]))
                except TypeError:
                    pass
            # RankDiagnostic-style ``on_step_end`` (no embeddings here).
            if hasattr(cb, "on_step_end") and not hasattr(cb, "save"):
                try:
                    cb.on_step_end(step)
                except TypeError:
                    pass


__all__ = ["Manager"]
