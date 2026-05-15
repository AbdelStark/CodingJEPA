"""Top-level package; re-exports the public symbols documented in spec/02-public-api.md."""

from codingjepa.eval import run_all
from codingjepa.inference import infer
from codingjepa.intents import INTENT_NONE, INTENTS
from codingjepa.model import CodingJEPA

__version__ = "0.0.0"

__all__ = [
    "CodingJEPA",
    "INTENT_NONE",
    "INTENTS",
    "__version__",
    "infer",
    "run_all",
]
