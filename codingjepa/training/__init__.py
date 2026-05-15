"""Training pipeline for CodingJEPA (RFC-0008).

Public surface (re-exports):

* :class:`Manager` — Lightning-analog training loop (issue #67).
* :class:`TrainingModule` — LeWM-style forward + loss wrapper (issue #66).
* :func:`build_optimizer`, :func:`build_scheduler` — AdamW + warmup-cosine
  (issue #69).
* :class:`ChunkSequenceDataset`, :class:`RefactorPairDataset`,
  :func:`build_pretrain_dataloader`, :func:`build_finetune_dataloader`,
  :func:`seed_worker` — data loading (issue #68).
* :class:`RankDiagnostic`, :class:`LossMonotonicity`, :class:`Checkpoint` —
  training callbacks (issues #70, #71, #72).
* :func:`run_preflight`, :class:`PreflightError` — pre-launch checklist
  (issue #74).
* :class:`WandBLogger` — Weights & Biases integration (issue #73).
"""

from __future__ import annotations

from codingjepa.training.callbacks import Checkpoint, LossMonotonicity, RankDiagnostic
from codingjepa.training.dataloader import (
    ChunkSequenceDataset,
    RefactorPairDataset,
    build_finetune_dataloader,
    build_pretrain_dataloader,
    seed_worker,
)
from codingjepa.training.logging import WandBLogger
from codingjepa.training.manager import Manager
from codingjepa.training.module import TrainingModule
from codingjepa.training.optimizer import build_optimizer, build_scheduler
from codingjepa.training.preflight import PreflightError, run_preflight

__all__ = [
    "Checkpoint",
    "ChunkSequenceDataset",
    "LossMonotonicity",
    "Manager",
    "PreflightError",
    "RankDiagnostic",
    "RefactorPairDataset",
    "TrainingModule",
    "WandBLogger",
    "build_finetune_dataloader",
    "build_optimizer",
    "build_pretrain_dataloader",
    "build_scheduler",
    "run_preflight",
    "seed_worker",
]
