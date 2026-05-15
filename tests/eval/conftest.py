"""Eval-namespace pytest conftest.

Suppress the macOS OpenMP duplicate-lib crash that triggers when ``faiss``
loads its bundled libomp before torch initializes its own copy. Setting
``KMP_DUPLICATE_LIB_OK`` here runs **before** any of the eval tests import
``codingjepa.eval.benchmarks`` (which transitively imports faiss), so the
flag is visible to libomp by the time it gets loaded.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
