"""Preflight checklist enforcer (issue #74, RFC-0008 §D16, SYSTEM-SPEC.md §2).

The preflight runs *before* a long training job is launched. Failing any
critical check raises :class:`PreflightError` (a plain Python exception,
intentionally *not* registered in :data:`codingjepa.errors.CLOSED_TAXONOMY`
because this is a training-time configuration concern, not a runtime
codingjepa-package error).

Checks performed (in order):

1. ``data/manifest.lock.json`` exists, parses as JSON, and — when the file
   matches the manifest schema — has a valid manifest hash.
2. All recorded repo audits pass the hard gates (compile / dedup / secrets /
   license). If the manifest has no ``audits`` block we treat the gate as
   "no information", which is a soft pass.
3. Baseline metrics file exists (BM25, CodeBERT, MLM-encoder) when
   ``require_baseline_metrics=True``.
4. Model is defined and exposes ``parameters()`` (when supplied).
5. GPU memory is sufficient (skipped on CPU-only hosts so the unit tests
   stay portable).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codingjepa.errors import ConfigError

if TYPE_CHECKING:  # pragma: no cover - import-cycle / type-only imports
    from codingjepa.model import CodingJEPA


class PreflightError(Exception):
    """Raised when a preflight check fails.

    Carries a ``failed`` list of check names plus a structured ``context``
    dict so callers (CLI / WandB) can render a precise error report.
    """

    def __init__(
        self,
        message: str = "",
        *,
        failed: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.failed = list(failed or [])
        self.context = dict(context or {})

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"PreflightError(message={self.message!r}, failed={self.failed!r}, "
            f"context={self.context!r})"
        )


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_manifest(path: Path) -> tuple[bool, dict[str, Any]]:
    """Check the manifest exists and has a valid hash where applicable.

    Returns ``(passed, manifest_payload_or_empty)``.
    """
    if not path.exists():
        return False, {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, {}

    # If the manifest looks like a real RFC-0014 manifest, verify the hash.
    if isinstance(payload, dict) and "manifest_hash" in payload and "repos" in payload:
        try:
            from codingjepa.data.manifest import verify_manifest_hash  # noqa: PLC0415
        except ImportError:  # pragma: no cover - module always present
            return True, payload
        if not verify_manifest_hash(payload):
            return False, payload
    return True, payload


def _check_audit_gates(manifest_payload: dict[str, Any]) -> bool:
    """Verify recorded audits, if any, pass the hard gates.

    When the manifest does not include an ``audits`` block, we have no
    information to assert — this is treated as a soft pass so the preflight
    does not block on a missing optional field.
    """
    audits = manifest_payload.get("audits")
    if not audits:
        return True
    # Accept either embedded audit dicts or just a list of audit paths.
    if isinstance(audits, list):
        for entry in audits:
            if isinstance(entry, dict):
                # Inline audit: must declare passes_all_gates=True.
                if not entry.get("passes_all_gates", False):
                    return False
            else:
                # We don't load external files here; surface failure if a
                # caller injected a non-dict, non-string entry.
                if not isinstance(entry, str):
                    return False
        return True
    if isinstance(audits, dict):
        for entry in audits.values():
            if isinstance(entry, dict) and not entry.get("passes_all_gates", False):
                return False
        return True
    return False


def _check_baselines(path: Path) -> bool:
    """Check that the baseline-metrics file exists and parses as JSON."""
    if not path.exists():
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return True


def _check_model(model: CodingJEPA | None) -> bool:
    """Check that a model was supplied and has at least one parameter."""
    if model is None:
        # No model passed — skip rather than fail; caller may not have built one.
        return True
    try:
        n_params = sum(p.numel() for p in model.parameters())
    except Exception:  # noqa: BLE001 — defensive
        return False
    return n_params > 0


def _check_gpu_memory(min_gb: float = 16.0) -> bool:
    """Check that at least one GPU has ``min_gb`` of free memory.

    Skipped (returns ``True``) on CPU-only hosts so this check does not
    spuriously fail in CI.
    """
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return True
        for i in range(torch.cuda.device_count()):
            free, _total = torch.cuda.mem_get_info(i)
            if free >= min_gb * (1024**3):
                return True
        return False
    except Exception:  # noqa: BLE001 — never fail preflight on introspection error
        return True


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def run_preflight(
    *,
    data_manifest_path: Path = Path("data/manifest.lock.json"),
    model: CodingJEPA | None = None,
    require_baseline_metrics: bool = True,
    baseline_metrics_path: Path = Path("data/audit/baselines.json"),
) -> dict[str, bool]:
    """Run all preflight checks before training (RFC-0008 §D16).

    Returns a dict of ``{check_name: bool}``. Raises :class:`PreflightError`
    when any required check fails.

    Parameters
    ----------
    data_manifest_path:
        Path to ``data/manifest.lock.json``.
    model:
        Optional :class:`codingjepa.model.CodingJEPA` instance. When supplied
        the model must have at least one parameter.
    require_baseline_metrics:
        Whether the baseline-metrics file is required (set to ``False`` from
        unit tests).
    baseline_metrics_path:
        Path to the baseline-metrics JSON file.
    """
    data_manifest_path = Path(data_manifest_path)
    baseline_metrics_path = Path(baseline_metrics_path)

    manifest_ok, manifest_payload = _check_manifest(data_manifest_path)
    audits_ok = _check_audit_gates(manifest_payload) if manifest_ok else False
    baselines_ok = _check_baselines(baseline_metrics_path) if require_baseline_metrics else True
    model_ok = _check_model(model)
    gpu_ok = _check_gpu_memory()

    results: dict[str, bool] = {
        "manifest_exists_and_valid": manifest_ok,
        "audit_gates": audits_ok,
        "baseline_metrics": baselines_ok,
        "model_defined": model_ok,
        "gpu_memory": gpu_ok,
    }

    failed = [name for name, ok in results.items() if not ok]
    if failed:
        raise PreflightError(
            f"preflight failed: {', '.join(failed)}",
            failed=failed,
            context={
                "manifest_path": str(data_manifest_path),
                "baseline_metrics_path": str(baseline_metrics_path),
                "results": results,
            },
        )
    return results


def check_baselines_first(
    baselines_dir: Path = Path("data/baselines"),
) -> None:
    """Refuse to start training if any baseline metrics JSON is missing.

    RFC-0005 §D9 requires that BM25, MLM-encoder, and CodeBERT baselines
    are computed and stored *before* the JEPA model is trained, so that
    the first checkpoint is immediately comparable to all three baselines.

    Parameters
    ----------
    baselines_dir:
        Root directory under which the three ``results.json`` files are
        expected.  Defaults to ``data/baselines``.

    Raises
    ------
    ConfigError
        With message ``"baseline missing"`` if any of the three expected
        files is absent from *baselines_dir*.
    """
    baselines_dir = Path(baselines_dir)
    required = [
        baselines_dir / "bm25" / "results.json",
        baselines_dir / "mlm_encoder" / "results.json",
        baselines_dir / "codebert" / "results.json",
    ]
    for path in required:
        if not path.exists():
            raise ConfigError("baseline missing", path=str(path))


__all__ = ["PreflightError", "check_baselines_first", "run_preflight"]
