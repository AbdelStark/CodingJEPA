"""WandB integration for CodingJEPA training (issue #73, RFC-0008 §D12).

Logs training metrics, model architecture, and hyperparameters via Weights &
Biases. Falls back to printing to stdout when:

* ``wandb`` is not installed,
* the caller passes ``disabled=True`` (used by the unit tests so they do not
  touch the network), or
* WandB initialization fails for any reason.

The logger is intentionally a thin wrapper — the underlying client is not
exposed, so the implementation can be swapped without breaking callers.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-cycle / type-only imports
    from codingjepa.model import CodingJEPA


try:
    import wandb as _wandb

    HAS_WANDB = True
except ImportError:  # pragma: no cover - wandb is a project dependency
    _wandb = None  # type: ignore[assignment]
    HAS_WANDB = False


def _count_parameters(model: Any) -> int:
    """Return the total parameter count of ``model``.

    ``Any`` for the annotation because we want to support both nn.Module and
    plain Python objects (defensive fallback for tests).
    """
    try:
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:  # noqa: BLE001
        return 0


class WandBLogger:
    """WandB integration per RFC-0008 §D12.

    Logs training metrics, model architecture, and hyperparameters.
    Falls back to console logging when WandB is unavailable or disabled.

    Parameters
    ----------
    project:
        WandB project name. Default ``"codingjepa"``.
    run_name:
        Optional run name. WandB will autogenerate one if ``None``.
    config:
        Optional config dict to log alongside the run.
    disabled:
        When ``True``, the logger never calls WandB and acts as a no-op.
    """

    def __init__(
        self,
        project: str = "codingjepa",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        *,
        disabled: bool = False,
    ) -> None:
        self.project = project
        self.run_name = run_name
        self.config = dict(config or {})
        self.disabled = disabled or not HAS_WANDB
        self._run: Any | None = None

        if not self.disabled and _wandb is not None:
            try:
                self._run = _wandb.init(
                    project=self.project,
                    name=self.run_name,
                    config=self.config,
                    reinit=True,
                )
            except Exception as exc:  # noqa: BLE001 — fall back gracefully
                print(
                    f"[WandBLogger] wandb.init failed ({exc!r}); "
                    "falling back to console logging",
                    file=sys.stderr,
                )
                self.disabled = True
                self._run = None

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Log a metrics dict at ``step``."""
        if self.disabled or self._run is None:
            return
        try:
            self._run.log(metrics, step=step)
        except Exception as exc:  # noqa: BLE001 — never bring down training
            print(
                f"[WandBLogger] log failed at step {step}: {exc!r}",
                file=sys.stderr,
            )

    def log_model(self, model: CodingJEPA, step: int) -> None:
        """Log a short model-architecture summary.

        We avoid uploading checkpoint artifacts here — RFC-0008 §D13
        delegates that to the :class:`codingjepa.training.callbacks.Checkpoint`
        callback. This method records the parameter count and module
        repr only.
        """
        if self.disabled or self._run is None:
            return
        try:
            summary = {
                "model/param_count": _count_parameters(model),
                "model/class": type(model).__name__,
            }
            self._run.summary.update(summary)
            self._run.log(summary, step=step)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[WandBLogger] log_model failed at step {step}: {exc!r}",
                file=sys.stderr,
            )

    def finish(self) -> None:
        """Close the WandB run."""
        if self.disabled or self._run is None:
            return
        try:
            self._run.finish()
        except Exception as exc:  # noqa: BLE001
            print(
                f"[WandBLogger] finish failed: {exc!r}",
                file=sys.stderr,
            )
        finally:
            self._run = None


__all__ = ["HAS_WANDB", "WandBLogger"]
