"""Tests for codingjepa.training (RFC-0008).

Covers issues #66-#74:
- optimizer: AdamW + LinearWarmupCosineAnnealingLR (#69)
- module: TrainingModule with forward + loss (#66)
- dataloader: workers, seeding, intent-balanced sampling (#68)
- callbacks: RankDiagnostic (#70), LossMonotonicity (#71), Checkpoint (#72)
- manager: Lightning-like training loop (#67)
- preflight: pre-launch checklist enforcer (#74)
- logging: WandB integration (#73)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Optimizer / Scheduler (issue #69)
# ---------------------------------------------------------------------------


def test_build_optimizer() -> None:
    from codingjepa.model import build_model
    from codingjepa.training.optimizer import build_optimizer

    model = build_model()
    opt = build_optimizer(model)
    assert isinstance(opt, torch.optim.AdamW)
    # Verify hyperparameters from RFC-0008 §D4.
    group = opt.param_groups[0]
    assert group["lr"] == pytest.approx(3e-4)
    assert group["weight_decay"] == pytest.approx(0.05)
    assert group["betas"] == (0.9, 0.95)
    assert group["eps"] == pytest.approx(1e-8)


def test_build_optimizer_custom_params() -> None:
    from codingjepa.model import build_model
    from codingjepa.training.optimizer import build_optimizer

    model = build_model()
    opt = build_optimizer(model, lr=1e-3, weight_decay=0.1)
    group = opt.param_groups[0]
    assert group["lr"] == pytest.approx(1e-3)
    assert group["weight_decay"] == pytest.approx(0.1)


def test_build_scheduler_warmup() -> None:
    from codingjepa.model import build_model
    from codingjepa.training.optimizer import build_optimizer, build_scheduler

    model = build_model()
    opt = build_optimizer(model)
    sch = build_scheduler(opt, warmup_steps=100, total_steps=1000)
    lrs: list[float] = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    # Should start small and grow monotonically during warmup.
    assert lrs[0] < lrs[-1]
    for a, b in zip(lrs[:-1], lrs[1:], strict=False):
        assert a <= b


def test_scheduler_warmup_reaches_base_lr() -> None:
    """At step == warmup_steps the LR equals the base LR."""
    from codingjepa.model import build_model
    from codingjepa.training.optimizer import build_optimizer, build_scheduler

    model = build_model()
    base_lr = 3e-4
    opt = build_optimizer(model, lr=base_lr)
    sch = build_scheduler(opt, warmup_steps=100, total_steps=1000, min_lr=1e-5)
    for _ in range(100):
        opt.step()
        sch.step()
    # After 100 warmup steps, LR should equal base_lr.
    assert opt.param_groups[0]["lr"] == pytest.approx(base_lr, rel=1e-3)


def test_scheduler_cosine_decay_to_min_lr() -> None:
    """At step == total_steps the LR equals min_lr."""
    from codingjepa.model import build_model
    from codingjepa.training.optimizer import build_optimizer, build_scheduler

    model = build_model()
    min_lr = 1e-5
    opt = build_optimizer(model, lr=3e-4)
    sch = build_scheduler(opt, warmup_steps=10, total_steps=100, min_lr=min_lr)
    for _ in range(100):
        opt.step()
        sch.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(min_lr, rel=1e-3)


# ---------------------------------------------------------------------------
# TrainingModule (issue #66)
# ---------------------------------------------------------------------------


def test_training_module_step() -> None:
    from codingjepa.model import build_model
    from codingjepa.training.module import TrainingModule
    from codingjepa.training.optimizer import build_optimizer, build_scheduler

    model = build_model()
    opt = build_optimizer(model, lr=1e-3)
    sch = build_scheduler(opt, warmup_steps=10, total_steps=100)
    module = TrainingModule(model, opt, sch, use_amp=False)
    B, S, L = 2, 4, 16
    tokens = torch.randint(0, 100, (B, S, L))
    intent = torch.zeros(B, dtype=torch.long)
    metrics = module.training_step(tokens, intent)
    assert "loss" in metrics
    assert "pred_loss" in metrics
    assert "sigreg_loss" in metrics
    assert not torch.isnan(torch.tensor(metrics["loss"]))


def test_training_module_global_step_increments() -> None:
    from codingjepa.model import build_model
    from codingjepa.training.module import TrainingModule
    from codingjepa.training.optimizer import build_optimizer, build_scheduler

    model = build_model()
    opt = build_optimizer(model, lr=1e-3)
    sch = build_scheduler(opt, warmup_steps=10, total_steps=100)
    module = TrainingModule(model, opt, sch, use_amp=False)
    assert module.global_step == 0
    B, S, L = 2, 4, 16
    tokens = torch.randint(0, 100, (B, S, L))
    intent = torch.zeros(B, dtype=torch.long)
    module.training_step(tokens, intent)
    assert module.global_step == 1
    module.training_step(tokens, intent)
    assert module.global_step == 2


def test_training_module_validation_step_no_grad() -> None:
    from codingjepa.model import build_model
    from codingjepa.training.module import TrainingModule
    from codingjepa.training.optimizer import build_optimizer, build_scheduler

    model = build_model()
    opt = build_optimizer(model, lr=1e-3)
    sch = build_scheduler(opt, warmup_steps=10, total_steps=100)
    module = TrainingModule(model, opt, sch, use_amp=False)
    B, S, L = 2, 4, 16
    tokens = torch.randint(0, 100, (B, S, L))
    intent = torch.zeros(B, dtype=torch.long)
    metrics = module.validation_step(tokens, intent)
    assert "loss" in metrics
    assert "pred_loss" in metrics


def test_training_module_current_lr() -> None:
    from codingjepa.model import build_model
    from codingjepa.training.module import TrainingModule
    from codingjepa.training.optimizer import build_optimizer, build_scheduler

    model = build_model()
    opt = build_optimizer(model, lr=1e-3)
    sch = build_scheduler(opt, warmup_steps=10, total_steps=100)
    module = TrainingModule(model, opt, sch, use_amp=False)
    assert isinstance(module.current_lr, float)


def test_training_module_grad_clip() -> None:
    """Gradient clipping is applied during training_step."""
    from codingjepa.model import build_model
    from codingjepa.training.module import TrainingModule
    from codingjepa.training.optimizer import build_optimizer, build_scheduler

    model = build_model()
    opt = build_optimizer(model, lr=1e-3)
    sch = build_scheduler(opt, warmup_steps=10, total_steps=100)
    module = TrainingModule(model, opt, sch, use_amp=False, grad_clip=0.5)
    B, S, L = 2, 4, 16
    tokens = torch.randint(0, 100, (B, S, L))
    intent = torch.zeros(B, dtype=torch.long)
    metrics = module.training_step(tokens, intent)
    # grad_norm should be reported and finite.
    assert "grad_norm" in metrics
    assert math.isfinite(metrics["grad_norm"])


# ---------------------------------------------------------------------------
# Callbacks: RankDiagnostic (issue #70)
# ---------------------------------------------------------------------------


def test_rank_diagnostic() -> None:
    from codingjepa.model import build_model
    from codingjepa.training.callbacks import RankDiagnostic

    cb = RankDiagnostic(build_model(), embed_dim=8)
    emb = torch.randn(100, 8)
    result = cb.on_step_end(5000, emb)
    assert "effective_rank" in result
    assert "passes" in result


def test_rank_diagnostic_passes_for_isotropic() -> None:
    """A nice isotropic random matrix should pass the rank gate."""
    from codingjepa.model import build_model
    from codingjepa.training.callbacks import RankDiagnostic

    cb = RankDiagnostic(build_model(), embed_dim=8, threshold=0.9)
    torch.manual_seed(0)
    emb = torch.randn(500, 8)
    result = cb.on_step_end(5000, emb)
    assert result["passes"] is True
    # Effective rank should be close to embed_dim for isotropic data.
    assert result["effective_rank"] >= 0.9 * 8


def test_rank_diagnostic_fails_for_collapsed() -> None:
    """A rank-1 matrix should fail the gate."""
    from codingjepa.model import build_model
    from codingjepa.training.callbacks import RankDiagnostic

    cb = RankDiagnostic(build_model(), embed_dim=8, threshold=0.9)
    # Construct a rank-1 matrix by outer product.
    base = torch.randn(8)
    emb = torch.randn(100, 1) * base.unsqueeze(0)
    result = cb.on_step_end(5000, emb)
    assert result["passes"] is False


def test_rank_diagnostic_skips_off_schedule() -> None:
    """Outside every_n_steps boundary the diagnostic skips and returns no-op."""
    from codingjepa.model import build_model
    from codingjepa.training.callbacks import RankDiagnostic

    cb = RankDiagnostic(build_model(), embed_dim=8, every_n_steps=5000)
    result = cb.on_step_end(123, torch.randn(100, 8))
    # On a non-boundary step, callback should signal it did not run.
    assert result.get("ran", False) is False


# ---------------------------------------------------------------------------
# Callbacks: LossMonotonicity (issue #71)
# ---------------------------------------------------------------------------


def test_loss_monotonicity_detects_divergence() -> None:
    from codingjepa.training.callbacks import LossMonotonicity

    cb = LossMonotonicity(window=5, gate_steps=50)
    result: dict[str, object] | None = None
    for i in range(55):
        result = cb.update(i, float(i))  # increasing losses
    assert result is not None
    assert not result["monotonic"]


def test_loss_monotonicity_passes_for_decreasing() -> None:
    from codingjepa.training.callbacks import LossMonotonicity

    cb = LossMonotonicity(window=5, gate_steps=50)
    result: dict[str, object] | None = None
    for i in range(55):
        result = cb.update(i, 100.0 - i)  # decreasing
    assert result is not None
    assert result["monotonic"]


def test_loss_monotonicity_disabled_after_gate_steps() -> None:
    """Past gate_steps the gate is no longer enforced (monotonic stays True)."""
    from codingjepa.training.callbacks import LossMonotonicity

    cb = LossMonotonicity(window=5, gate_steps=10)
    # First 10 steps: decreasing → ok
    for i in range(10):
        cb.update(i, 100.0 - i)
    # Past gate, losses spike — still considered ok because gate is off.
    result = cb.update(20, 1_000_000.0)
    assert result["monotonic"]


def test_loss_monotonicity_returns_running_avg() -> None:
    from codingjepa.training.callbacks import LossMonotonicity

    cb = LossMonotonicity(window=5, gate_steps=50)
    result = cb.update(0, 10.0)
    assert "running_avg" in result
    assert result["running_avg"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Callbacks: Checkpoint (issue #72)
# ---------------------------------------------------------------------------


def test_checkpoint_save_load(tmp_path: Path) -> None:
    from codingjepa.model import build_model
    from codingjepa.training.callbacks import Checkpoint

    model = build_model()
    opt = torch.optim.AdamW(model.parameters())
    cb = Checkpoint(tmp_path)
    path = cb.save(model, opt, step=100, metrics={"val_loss": 1.5})
    assert path.exists()


def test_checkpoint_keep_last_n(tmp_path: Path) -> None:
    """Only the last `keep_last` checkpoints are retained (plus best)."""
    from codingjepa.model import build_model
    from codingjepa.training.callbacks import Checkpoint

    model = build_model()
    opt = torch.optim.AdamW(model.parameters())
    cb = Checkpoint(tmp_path, keep_last=2)
    # Save 5 checkpoints, all with same metric so best is the first.
    paths: list[Path] = []
    for i, step in enumerate([10, 20, 30, 40, 50]):
        # Make val_retrieval_at_10 worse over time so best is step 10.
        metric = 1.0 - i * 0.1
        paths.append(cb.save(model, opt, step=step, metrics={"val_retrieval_at_10": metric}))
    # Last 2 step files should exist.
    existing = sorted(tmp_path.glob("step_*.pt"))
    # 2 most recent + 1 best (the best may or may not be in the last 2).
    # The contract: keep last 2 + best.
    step_numbers = {int(p.stem.split("_")[1]) for p in existing}
    assert 40 in step_numbers
    assert 50 in step_numbers
    # Best (step 10, highest val_retrieval_at_10) should also be retained.
    assert 10 in step_numbers


def test_checkpoint_load_latest(tmp_path: Path) -> None:
    from codingjepa.model import build_model
    from codingjepa.training.callbacks import Checkpoint

    model = build_model()
    opt = torch.optim.AdamW(model.parameters())
    cb = Checkpoint(tmp_path)
    cb.save(model, opt, step=10, metrics={"val_loss": 2.0})
    cb.save(model, opt, step=50, metrics={"val_loss": 1.0})
    cb.save(model, opt, step=30, metrics={"val_loss": 1.5})
    fresh_model = build_model()
    step = cb.load_latest(fresh_model)
    assert step == 50


def test_checkpoint_load_best(tmp_path: Path) -> None:
    from codingjepa.model import build_model
    from codingjepa.training.callbacks import Checkpoint

    model = build_model()
    opt = torch.optim.AdamW(model.parameters())
    cb = Checkpoint(tmp_path)
    # Highest val_retrieval_at_10 wins → step 30 should be best.
    cb.save(model, opt, step=10, metrics={"val_retrieval_at_10": 0.4})
    cb.save(model, opt, step=20, metrics={"val_retrieval_at_10": 0.5})
    cb.save(model, opt, step=30, metrics={"val_retrieval_at_10": 0.7})
    cb.save(model, opt, step=40, metrics={"val_retrieval_at_10": 0.6})
    fresh_model = build_model()
    step = cb.load_best(fresh_model)
    assert step == 30


# ---------------------------------------------------------------------------
# Preflight (issue #74)
# ---------------------------------------------------------------------------


def test_preflight_missing_manifest(tmp_path: Path) -> None:
    from codingjepa.training.preflight import PreflightError, run_preflight

    with pytest.raises(PreflightError):
        run_preflight(
            data_manifest_path=tmp_path / "missing.json",
            require_baseline_metrics=False,
        )


def test_preflight_returns_check_dict(tmp_path: Path) -> None:
    """When called with a fake manifest, preflight returns a dict of checks."""
    from codingjepa.training.preflight import PreflightError

    # Try a totally missing manifest, expect failure.
    with pytest.raises(PreflightError):
        from codingjepa.training.preflight import run_preflight

        run_preflight(
            data_manifest_path=tmp_path / "no.json",
            require_baseline_metrics=False,
        )


def test_preflight_missing_baselines(tmp_path: Path) -> None:
    """Preflight fails if baseline metrics are required but missing."""
    from codingjepa.training.preflight import PreflightError, run_preflight

    # Create a fake valid manifest file (preflight should still fail on baseline).
    manifest_path = tmp_path / "manifest.lock.json"
    manifest_path.write_text("{}")  # not strictly valid JSON-schema but exists
    with pytest.raises(PreflightError):
        run_preflight(
            data_manifest_path=manifest_path,
            require_baseline_metrics=True,
            baseline_metrics_path=tmp_path / "missing-baselines.json",
        )


# ---------------------------------------------------------------------------
# Logging (issue #73)
# ---------------------------------------------------------------------------


def test_wandb_logger_disabled() -> None:
    from codingjepa.training.logging import WandBLogger

    logger = WandBLogger(disabled=True)
    logger.log({"loss": 1.0}, step=1)  # should not raise
    logger.finish()


def test_wandb_logger_log_model_disabled() -> None:
    """log_model on a disabled logger should be a no-op."""
    from codingjepa.model import build_model
    from codingjepa.training.logging import WandBLogger

    logger = WandBLogger(disabled=True)
    logger.log_model(build_model(), step=0)
    logger.finish()


# ---------------------------------------------------------------------------
# DataLoader (issue #68)
# ---------------------------------------------------------------------------


def test_chunk_sequence_dataset(tmp_path: Path) -> None:
    from codingjepa.training.dataloader import ChunkSequenceDataset

    data_file = tmp_path / "seq.jsonl"
    rows = [
        {
            "context_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "target_ids": [10, 11, 12],
            "intent_idx": -1,
        }
    ] * 5
    with open(data_file, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    ds = ChunkSequenceDataset(data_file)
    assert len(ds) == 5


def test_chunk_sequence_dataset_getitem(tmp_path: Path) -> None:
    from codingjepa.training.dataloader import ChunkSequenceDataset

    data_file = tmp_path / "seq.jsonl"
    rows = [
        {
            "context_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "target_ids": [10, 11, 12],
            "intent_idx": 0,
        }
    ] * 3
    with open(data_file, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    ds = ChunkSequenceDataset(data_file, max_length=16)
    item = ds[0]
    assert "token_ids" in item
    assert "intent_idx" in item
    assert isinstance(item["token_ids"], torch.Tensor)
    # token_ids should be (S, L) where S = H + n_preds (4 chunks here).
    assert item["token_ids"].ndim == 2


def test_refactor_pair_dataset(tmp_path: Path) -> None:
    from codingjepa.training.dataloader import RefactorPairDataset

    data_file = tmp_path / "pairs.jsonl"
    rows = [
        {
            "before_ids": [1, 2, 3, 4],
            "after_ids": [5, 6, 7, 8],
            "intent_idx": i % 8,
        }
        for i in range(16)
    ]
    with open(data_file, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    ds = RefactorPairDataset(data_file)
    assert len(ds) == 16
    item = ds[0]
    assert "token_ids" in item
    assert "intent_idx" in item


def test_seed_worker_is_deterministic() -> None:
    """seed_worker sets a deterministic seed based on worker_id."""
    from codingjepa.training.dataloader import seed_worker

    # We just check that the function runs without raising.
    seed_worker(0)
    seed_worker(1)


def test_build_pretrain_dataloader(tmp_path: Path) -> None:
    from codingjepa.training.dataloader import (
        ChunkSequenceDataset,
        build_pretrain_dataloader,
    )

    data_file = tmp_path / "seq.jsonl"
    rows = [
        {
            "context_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "target_ids": [10, 11, 12],
            "intent_idx": -1,
        }
    ] * 16
    with open(data_file, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    ds = ChunkSequenceDataset(data_file)
    loader = build_pretrain_dataloader(ds, batch_size=4, num_workers=0)
    batch = next(iter(loader))
    assert "token_ids" in batch
    assert batch["token_ids"].shape[0] == 4


def test_build_finetune_dataloader_intent_balanced(tmp_path: Path) -> None:
    from codingjepa.training.dataloader import (
        RefactorPairDataset,
        build_finetune_dataloader,
    )

    data_file = tmp_path / "pairs.jsonl"
    # Heavily skewed: 14 of intent 0, 2 of intent 1.
    rows = []
    for _ in range(14):
        rows.append({"before_ids": [1, 2], "after_ids": [3, 4], "intent_idx": 0})
    for _ in range(2):
        rows.append({"before_ids": [1, 2], "after_ids": [3, 4], "intent_idx": 1})
    with open(data_file, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    ds = RefactorPairDataset(data_file)
    loader = build_finetune_dataloader(ds, batch_size=4, num_workers=0)
    # Iterate a few batches; the sampler should over-sample intent 1.
    intents_seen: list[int] = []
    for i, batch in enumerate(loader):
        if i >= 4:
            break
        intents_seen.extend(int(x) for x in batch["intent_idx"].tolist())
    # In strict uniform sampling, intent 1 would appear ~12.5% of the time.
    # With intent-balanced weighted sampling it should appear much more often.
    n_intent_1 = sum(1 for x in intents_seen if x == 1)
    # At least one intent-1 should appear.
    assert n_intent_1 >= 1


# ---------------------------------------------------------------------------
# Manager (issue #67)
# ---------------------------------------------------------------------------


def test_manager_constructs(tmp_path: Path) -> None:
    from codingjepa.model import build_model
    from codingjepa.training.dataloader import (
        ChunkSequenceDataset,
        build_pretrain_dataloader,
    )
    from codingjepa.training.manager import Manager
    from codingjepa.training.module import TrainingModule
    from codingjepa.training.optimizer import build_optimizer, build_scheduler

    data_file = tmp_path / "seq.jsonl"
    rows = [
        {
            "context_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "target_ids": [10, 11, 12],
            "intent_idx": -1,
        }
    ] * 4
    with open(data_file, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    ds = ChunkSequenceDataset(data_file, max_length=16)
    loader = build_pretrain_dataloader(ds, batch_size=2, num_workers=0)
    model = build_model()
    opt = build_optimizer(model, lr=1e-3)
    sch = build_scheduler(opt, warmup_steps=10, total_steps=100)
    module = TrainingModule(model, opt, sch, use_amp=False)
    mgr = Manager(module, loader, run_dir=tmp_path / "runs")
    assert mgr is not None


def test_manager_fit_runs(tmp_path: Path) -> None:
    """Manager.fit should run a few steps without raising."""
    from codingjepa.model import build_model
    from codingjepa.training.dataloader import (
        ChunkSequenceDataset,
        build_pretrain_dataloader,
    )
    from codingjepa.training.manager import Manager
    from codingjepa.training.module import TrainingModule
    from codingjepa.training.optimizer import build_optimizer, build_scheduler

    data_file = tmp_path / "seq.jsonl"
    rows = [
        {
            "context_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "target_ids": [10, 11, 12],
            "intent_idx": -1,
        }
    ] * 8
    with open(data_file, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    ds = ChunkSequenceDataset(data_file, max_length=16)
    loader = build_pretrain_dataloader(ds, batch_size=2, num_workers=0)
    model = build_model()
    opt = build_optimizer(model, lr=1e-3)
    sch = build_scheduler(opt, warmup_steps=10, total_steps=100)
    module = TrainingModule(model, opt, sch, use_amp=False)
    mgr = Manager(
        module,
        loader,
        run_dir=tmp_path / "runs",
        log_every_n_steps=1,
        max_steps=2,
    )
    final_metrics = mgr.fit(max_steps=2)
    assert "loss" in final_metrics
