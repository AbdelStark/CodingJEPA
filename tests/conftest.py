"""Top-level pytest conftest.

Set ``KMP_DUPLICATE_LIB_OK=TRUE`` at session start to suppress the macOS
OpenMP duplicate-lib warning that triggers when ``faiss`` (libomp via
OpenBLAS) and ``torch`` (its own libomp) both load. Several individual
modules (``tests/inference/test_retrieve.py`` et al.) also set this at
import time for safety; centralizing it here means new test modules that
bring in ``codingjepa.eval`` do not need to repeat the boilerplate.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
