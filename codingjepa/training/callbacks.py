"""Training callbacks for CodingJEPA (RFC-0008 §D7, §D13).

This module exposes the three callbacks called out in the implementation plan:

* :class:`RankDiagnostic` (issue #70) — effective-rank gate run at the end of
  each epoch (or every 5k steps). Halts training if the projected embedding
  matrix collapses below ``0.9 * embed_dim`` effective dimensions.
* :class:`LossMonotonicity` (issue #71) — first-5k-steps loss-monotonicity
  gate. Tracks a 100-step running average of ``pred_loss`` and asserts that
  it decreases monotonically across consecutive windows.
* :class:`Checkpoint` (issue #72) — saves the model + optimizer state every
  epoch / every-N-steps and keeps the last 3 checkpoints + the best one by
  ``val_retrieval_at_10``.

The callbacks are intentionally plain Python objects (no Lightning Callback
inheritance) so they can be invoked from the bespoke :class:`Manager`
training loop without dragging in the full Lightning dependency tree.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:  # pragma: no cover - import-cycle / type-only imports
    from codingjepa.model import CodingJEPA


# ---------------------------------------------------------------------------
# RankDiagnostic (issue #70, RFC-0008 §D7)
# ---------------------------------------------------------------------------


def _effective_rank(emb: torch.Tensor) -> float:
    """Return the effective rank ``exp(H(σ))`` of an embedding matrix.

    ``σ`` is the normalized singular-value spectrum of ``emb``; ``H`` denotes
    Shannon entropy. The result lies in ``[1, min(N, D)]`` and equals the
    full rank only when the spectrum is perfectly uniform.
    """
    if emb.dim() != 2:
        emb = emb.reshape(-1, emb.shape[-1])
    # Compute singular values; on CPU this falls back to SVD.
    s = torch.linalg.svdvals(emb.float())
    s = s[s > 0]
    if s.numel() == 0:
        return 1.0
    p = s / s.sum()
    # Shannon entropy (in nats).
    eps = 1e-12
    h = -(p * (p + eps).log()).sum()
    return float(h.exp().item())


class RankDiagnostic:
    """Effective-rank gate per RFC-0008 §D7.

    Computes effective rank = ``exp(H(σ))`` where ``σ`` is the normalized
    singular-value spectrum of the embedding matrix.

    Gate: ``effective_rank >= threshold * embed_dim``.

    Parameters
    ----------
    model:
        The CodingJEPA model (kept as a reference so callers can pass it in
        their preferred way; the model is not mutated by this callback).
    embed_dim:
        Embedding dimension. Default 512.
    threshold:
        Fractional gate. Default 0.9 per RFC-0008 §D7.
    every_n_steps:
        Run the diagnostic only on steps where ``step % every_n_steps == 0``.
        Default 5,000.
    """

    def __init__(
        self,
        model: CodingJEPA,
        embed_dim: int = 512,
        threshold: float = 0.9,
        every_n_steps: int = 5_000,
    ) -> None:
        self.model = model
        self.embed_dim = embed_dim
        self.threshold = threshold
        self.every_n_steps = every_n_steps

    def on_step_end(
        self,
        step: int,
        embeddings: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Check the rank gate.

        Parameters
        ----------
        step:
            Current global step (used to decide whether to skip the
            diagnostic on non-boundary steps).
        embeddings:
            ``(N, D)`` projected embedding matrix to test. When ``None``,
            the diagnostic returns ``{'ran': False}``.

        Returns
        -------
        dict
            On a non-boundary step or when ``embeddings is None``:
            ``{'ran': False}``. On a boundary step with embeddings:
            ``{'ran': True, 'effective_rank': float, 'passes': bool}``.
        """
        if self.every_n_steps > 0 and step % self.every_n_steps != 0:
            return {"ran": False}
        if embeddings is None:
            return {"ran": False}
        eff_rank = _effective_rank(embeddings)
        gate = self.threshold * self.embed_dim
        return {
            "ran": True,
            "effective_rank": eff_rank,
            "passes": bool(eff_rank >= gate),
        }


# ---------------------------------------------------------------------------
# LossMonotonicity (issue #71, RFC-0008 §D7)
# ---------------------------------------------------------------------------


class LossMonotonicity:
    """First-5k-steps loss-monotonicity gate per RFC-0008 §D7.

    Tracks a ``window``-step running average of ``pred_loss``. The gate fires
    if the running average ever fails to decrease compared to the previous
    full window during the first ``gate_steps`` steps.

    Parameters
    ----------
    window:
        Number of steps per running-average window. Default 100.
    gate_steps:
        Number of initial steps over which the gate is enforced. Default 5,000.
    """

    def __init__(self, window: int = 100, gate_steps: int = 5_000) -> None:
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        self.window = window
        self.gate_steps = gate_steps
        self._buffer: deque[float] = deque(maxlen=window)
        self._last_full_window_avg: float | None = None
        # ``True`` until a window fails; once ``False`` the gate stays open.
        self._monotonic: bool = True

    def update(self, step: int, pred_loss: float) -> dict[str, bool | float]:
        """Update with a new loss value. Returns gate state.

        Returns ``{'monotonic': bool, 'running_avg': float}``.

        ``monotonic`` is ``True`` while the running average has been
        non-increasing across every complete window seen so far (within the
        first ``gate_steps`` steps); it becomes (and stays) ``False`` the
        first time a window's average exceeds the previous window's average.
        After ``step >= gate_steps`` the gate is permanently considered to
        have passed.
        """
        self._buffer.append(float(pred_loss))
        running_avg = sum(self._buffer) / len(self._buffer)

        # Only check the gate within the initial monitored region.
        if step < self.gate_steps:
            # End of a window: compare against the last full-window average.
            if len(self._buffer) == self.window and (step + 1) % self.window == 0:
                if (
                    self._last_full_window_avg is not None
                    and running_avg > self._last_full_window_avg
                ):
                    self._monotonic = False
                self._last_full_window_avg = running_avg

        return {
            "monotonic": self._monotonic,
            "running_avg": running_avg,
        }


# ---------------------------------------------------------------------------
# Checkpoint (issue #72, RFC-0008 §D13)
# ---------------------------------------------------------------------------


_BEST_METRIC_KEY = "val_retrieval_at_10"


def _checkpoint_paths(checkpoint_dir: Path) -> list[Path]:
    """List all ``step_*.pt`` files in ``checkpoint_dir``, ascending by step."""
    if not checkpoint_dir.exists():
        return []
    items: list[tuple[int, Path]] = []
    for p in checkpoint_dir.glob("step_*.pt"):
        try:
            step = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        items.append((step, p))
    items.sort(key=lambda t: t[0])
    return [p for _, p in items]


class Checkpoint:
    """Checkpoint callback per RFC-0008 §D13.

    Saves the model + optimizer state and metric dict to
    ``checkpoint_dir / step_<N>.pt``. Keeps the last ``keep_last`` files plus
    the single "best" checkpoint as scored by ``val_retrieval_at_10``
    (higher is better).

    Parameters
    ----------
    checkpoint_dir:
        Directory under which to write checkpoint files. Created if absent.
    keep_last:
        Number of most-recent checkpoints to retain. Default 3.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        keep_last: int = 3,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        # Index of best step → (best_metric_value, path).
        self._best: tuple[float, Path] | None = None

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        model: CodingJEPA,
        optimizer: torch.optim.Optimizer,
        step: int,
        metrics: dict[str, float],
    ) -> Path:
        """Save a checkpoint and prune retained files.

        The saved file contains:

        ``model_state``, ``optimizer_state``, ``step``, ``metrics``.
        """
        path = self.checkpoint_dir / f"step_{step}.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": int(step),
                "metrics": dict(metrics),
            },
            path,
        )

        # Track best by val_retrieval_at_10 (higher is better).
        best_value = metrics.get(_BEST_METRIC_KEY)
        if isinstance(best_value, (int, float)) and not math.isnan(float(best_value)):
            if self._best is None or float(best_value) > self._best[0]:
                self._best = (float(best_value), path)

        self._prune()
        return path

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_best(self, model: CodingJEPA) -> int:
        """Load the best checkpoint by ``val_retrieval_at_10`` into ``model``.

        Returns the step number of the loaded checkpoint. If no checkpoint
        has a ``val_retrieval_at_10`` metric, falls back to :meth:`load_latest`.
        """
        if self._best is not None:
            return self._load_path(model, self._best[1])
        best_path, best_step = self._scan_best_path()
        if best_path is None:
            return self.load_latest(model)
        return self._load_path(model, best_path, step_hint=best_step)

    def load_latest(self, model: CodingJEPA) -> int:
        """Load the most recent checkpoint into ``model``.

        Returns the step number of the loaded checkpoint. Raises
        :class:`FileNotFoundError` if no checkpoints exist.
        """
        paths = _checkpoint_paths(self.checkpoint_dir)
        if not paths:
            raise FileNotFoundError(f"no checkpoints found in {self.checkpoint_dir}")
        return self._load_path(model, paths[-1])

    def _load_path(
        self,
        model: CodingJEPA,
        path: Path,
        step_hint: int | None = None,
    ) -> int:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        if "step" in payload:
            return int(payload["step"])
        if step_hint is not None:
            return step_hint
        return int(path.stem.split("_")[1])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _scan_best_path(self) -> tuple[Path | None, int | None]:
        best_path: Path | None = None
        best_step: int | None = None
        best_value: float | None = None
        for p in _checkpoint_paths(self.checkpoint_dir):
            try:
                payload = torch.load(p, map_location="cpu", weights_only=False)
            except Exception:  # noqa: BLE001 — skip unreadable files
                continue
            metrics = payload.get("metrics", {})
            value = metrics.get(_BEST_METRIC_KEY)
            if not isinstance(value, (int, float)):
                continue
            f_value = float(value)
            if best_value is None or f_value > best_value:
                best_value = f_value
                best_path = p
                best_step = int(payload.get("step", p.stem.split("_")[1]))
        return best_path, best_step

    def _prune(self) -> None:
        """Delete checkpoints beyond ``keep_last`` while protecting best."""
        paths = _checkpoint_paths(self.checkpoint_dir)
        if len(paths) <= self.keep_last:
            return
        protected: set[Path] = set(paths[-self.keep_last :])
        if self._best is not None:
            protected.add(self._best[1])
        for p in paths:
            if p not in protected:
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass


def _iter_metric_values(items: Iterable[tuple[int, Path]]) -> Iterable[tuple[int, Path]]:
    """Compat helper used by older callsites; iterates as-is."""
    yield from items


__all__ = ["Checkpoint", "LossMonotonicity", "RankDiagnostic"]
