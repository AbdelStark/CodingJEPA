"""Model modules (encoder/projector/predictor/intent/sigreg). Placeholder re-exports.

Public per spec/02 §codingjepa.modules; real implementations land in #58–#63.
"""

from __future__ import annotations

from codingjepa.modules.ar_predictor import ARPredictor
from codingjepa.modules.encoder import Encoder
from codingjepa.modules.intent_embedder import IntentEmbedder
from codingjepa.modules.pred_proj import PredProj
from codingjepa.modules.projector import Projector
from codingjepa.modules.sigreg import SIGReg

__all__ = [
    "ARPredictor",
    "Encoder",
    "IntentEmbedder",
    "PredProj",
    "Projector",
    "SIGReg",
]
