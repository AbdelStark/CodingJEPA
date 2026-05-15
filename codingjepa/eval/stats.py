"""Bootstrap CI + paired-bootstrap p-value. RFC-0005 §D7 / RFC-0010 §D7.

* `bootstrap_ci(samples, n_boot=1000, level=0.95)` — percentile-method CI on
  the mean of `samples`.
* `paired_bootstrap_p(model_scores, baseline_scores)` — one-sided p-value for
  H0: mean(model) <= mean(baseline) under paired resampling.

The implementation is numpy-only at runtime. The scipy-comparison test
(`tests/eval/test_paired_bootstrap.py`) exercises it against
`scipy.stats.bootstrap` within `1e-3` and is gated by the dev extra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from codingjepa.errors import UsageError

_RNG_DEFAULT_SEED = 0


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    std: float
    ci_low: float
    ci_high: float
    n_seeds: int
    n_boot: int
    level: float


def bootstrap_ci(
    samples: npt.ArrayLike,
    *,
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = _RNG_DEFAULT_SEED,
) -> BootstrapCI:
    """Percentile-method bootstrap CI on the mean of `samples`.

    Replicates `scipy.stats.bootstrap(..., method='percentile')` to within
    Monte-Carlo error for the same `n_boot` budget.
    """

    arr = _as_1d(samples, name="samples")
    if arr.size < 2:
        raise UsageError("bootstrap_ci requires >= 2 samples", n=int(arr.size))
    if not 0.0 < level < 1.0:
        raise UsageError("level must be in (0, 1)", level=level)
    if n_boot < 1:
        raise UsageError("n_boot must be >= 1", n_boot=n_boot)

    rng = np.random.default_rng(seed)
    n = arr.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(boot_means, [alpha, 1.0 - alpha])

    return BootstrapCI(
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        ci_low=float(low),
        ci_high=float(high),
        n_seeds=int(arr.size),
        n_boot=int(n_boot),
        level=float(level),
    )


def paired_bootstrap_p(
    model_scores: npt.ArrayLike,
    baseline_scores: npt.ArrayLike,
    *,
    n_boot: int = 10000,
    seed: int = _RNG_DEFAULT_SEED,
) -> float:
    """Paired bootstrap p-value for H0: mean(model) <= mean(baseline).

    Computes ``p = P(boot_diff_mean <= 0)`` under paired resampling with
    replacement, where ``diff_i = model_i - baseline_i``. The test is
    one-sided in the "is the model strictly better?" direction.
    """

    model = _as_1d(model_scores, name="model_scores")
    baseline = _as_1d(baseline_scores, name="baseline_scores")
    if model.shape != baseline.shape:
        raise UsageError(
            "paired_bootstrap_p requires equal-length paired scores",
            model_shape=model.shape,
            baseline_shape=baseline.shape,
        )
    if model.size < 2:
        raise UsageError("paired_bootstrap_p requires >= 2 pairs", n=int(model.size))
    if n_boot < 1:
        raise UsageError("n_boot must be >= 1", n_boot=n_boot)

    diff = model - baseline
    rng = np.random.default_rng(seed)
    n = diff.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    return float(np.mean(boot_means <= 0.0))


def _as_1d(value: npt.ArrayLike, *, name: str) -> npt.NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise UsageError(f"{name} must be 1-D", name=name, ndim=int(arr.ndim))
    return arr


__all__ = ["BootstrapCI", "bootstrap_ci", "paired_bootstrap_p"]
