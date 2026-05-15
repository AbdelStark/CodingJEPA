"""CJ-PROBE-NAME / CJ-PROBE-DEFECT / CJ-PROBE-CLONE linear probes
(RFC-0010 §D3, #114).

Each probe trains a logistic-regression classifier on synthetic embeddings
plus labels and reports either top-1 accuracy (NAME) or F1 (DEFECT, CLONE).
``sklearn`` is an optional dependency — when it is unavailable we either
fall back to a tiny numpy closed-form binary classifier or, for the
multi-class NAME probe, mark the result as ``skipped=True``.
"""

from __future__ import annotations

import pathlib

import numpy as np
import numpy.typing as npt

from codingjepa.eval.harness import Benchmark, BenchmarkResult

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover — sklearn-absent CI path
    _HAS_SKLEARN = False


def _binary_logreg_predict(
    x_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.int64],
    x_test: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]:
    """Closed-form binary classifier used when sklearn is unavailable.

    Picks the class whose mean training embedding is closer in Euclidean
    distance to the test embedding. This is enough to give a deterministic,
    non-trivial F1 on the synthetic data the test fixture generates.
    """

    classes = np.unique(y_train)
    if classes.size != 2:
        # Degenerate — predict the majority.
        majority = classes[0] if classes.size else 0
        return np.full(x_test.shape[0], majority, dtype=np.int64)
    mu0 = x_train[y_train == classes[0]].mean(axis=0)
    mu1 = x_train[y_train == classes[1]].mean(axis=0)
    d0 = np.linalg.norm(x_test - mu0, axis=1)
    d1 = np.linalg.norm(x_test - mu1, axis=1)
    result: npt.NDArray[np.int64] = np.where(d0 <= d1, classes[0], classes[1]).astype(np.int64)
    return result


def _binary_f1(y_true: npt.NDArray[np.int64], y_pred: npt.NDArray[np.int64]) -> float:
    """Macro F1 fallback when sklearn is unavailable."""

    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1s: list[float] = []
    for c in classes:
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


class _LinearProbe(Benchmark):
    """Synthetic linear-probe base. Subclasses pick the label distribution."""

    benchmark_id = "CJ-PROBE"
    n_classes: int = 2
    metric_key: str = "f1"

    def __init__(
        self,
        *,
        n_examples: int = 200,
        dim: int = 64,
        global_seed: int = 0,
    ) -> None:
        super().__init__(global_seed=global_seed)
        self._n_examples = n_examples
        self._dim = dim
        self._x: npt.NDArray[np.float64] | None = None
        self._y: npt.NDArray[np.int64] | None = None

    def prepare(self, data_dir: pathlib.Path) -> None:  # noqa: ARG002 — interface
        rng = np.random.default_rng(self.seed)
        # Class-conditional Gaussians so the probe has signal.
        y = rng.integers(0, self.n_classes, size=self._n_examples).astype(np.int64)
        centers = rng.standard_normal((self.n_classes, self._dim)) * 3.0
        x = centers[y] + rng.standard_normal((self._n_examples, self._dim))
        self._x = x.astype(np.float64)
        self._y = y

    def run(self) -> BenchmarkResult:
        if self._x is None or self._y is None:
            raise RuntimeError("prepare() must be called before run()")
        split = self._n_examples // 2
        x_train, x_test = self._x[:split], self._x[split:]
        y_train, y_test = self._y[:split], self._y[split:]

        if _HAS_SKLEARN:
            clf = LogisticRegression(max_iter=1000, multi_class="auto")
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            if self.metric_key == "top1_accuracy":
                metric = float(np.mean(y_pred == y_test))
            else:
                metric = float(f1_score(y_test, y_pred, average="macro"))
            return BenchmarkResult(
                benchmark_id=self.benchmark_id,
                metrics={self.metric_key: metric, "n_examples": self._n_examples},
            )

        # numpy fallback — only meaningful for the binary case.
        if self.n_classes == 2:
            y_pred = _binary_logreg_predict(x_train, y_train, x_test)
            metric = _binary_f1(y_test, y_pred)
            return BenchmarkResult(
                benchmark_id=self.benchmark_id,
                metrics={
                    self.metric_key: float(metric),
                    "n_examples": self._n_examples,
                    "skipped": False,
                },
            )

        # Multi-class without sklearn → mark as skipped.
        return BenchmarkResult(
            benchmark_id=self.benchmark_id,
            metrics={
                self.metric_key: 0.0,
                "n_examples": self._n_examples,
                "skipped": True,
                "reason": "sklearn unavailable for multi-class probe",
            },
        )


class ProbeNameBenchmark(_LinearProbe):
    benchmark_id = "CJ-PROBE-NAME"
    n_classes = 8
    metric_key = "top1_accuracy"


class ProbeDefectBenchmark(_LinearProbe):
    benchmark_id = "CJ-PROBE-DEFECT"
    n_classes = 2
    metric_key = "f1"


class ProbeCloneBenchmark(_LinearProbe):
    benchmark_id = "CJ-PROBE-CLONE"
    n_classes = 2
    metric_key = "f1"


__all__ = [
    "ProbeCloneBenchmark",
    "ProbeDefectBenchmark",
    "ProbeNameBenchmark",
]
