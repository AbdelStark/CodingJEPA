"""CodingJEPA retrieval baselines (RFC-0005 §D9).

Re-exports :class:`Metrics` and the ``run`` / ``write_results`` callables
from each individual baseline module so callers can do::

    from codingjepa.baselines import Metrics
    from codingjepa.baselines.bm25 import run as bm25_run
"""

from codingjepa.baselines.bm25 import Metrics
from codingjepa.baselines.bm25 import run as bm25_run
from codingjepa.baselines.bm25 import write_results as bm25_write
from codingjepa.baselines.codebert import (
    CodeBERTBaseline,
    write_lock_file,
)
from codingjepa.baselines.mlm_encoder import (
    MLMEncoder,
    mask_tokens,
)
from codingjepa.baselines.mlm_encoder import (
    run as mlm_run,
)
from codingjepa.baselines.mlm_encoder import (
    write_results as mlm_write,
)

__all__ = [
    "CodeBERTBaseline",
    "MLMEncoder",
    "Metrics",
    "bm25_run",
    "bm25_write",
    "mask_tokens",
    "mlm_run",
    "mlm_write",
    "write_lock_file",
]
