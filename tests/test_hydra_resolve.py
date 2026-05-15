"""Hydra config resolution tests (issue #17, RFC-0008 §D11).

Verifies that the config tree under ``config/`` composes cleanly via
``hydra.initialize_config_dir`` and exposes the load-bearing fields used by
the training pipeline (optimizer LR, model hidden size, loss lambda, etc.).

These tests are deliberately schema-light: they assert that every config
resolves without error, that defaults compose correctly, and that the
RFC-0008 values flow through to the merged config.
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterator

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"


@pytest.fixture(autouse=True)
def _clear_hydra_state() -> Iterator[None]:
    """Hydra's global state is process-wide; clear it around each test.

    ``initialize_config_dir`` complains if a previous test left state
    initialized, so we explicitly tear it down before and after each test.
    """
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    yield
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


def _compose(config_name: str, overrides: list[str] | None = None) -> DictConfig:
    """Compose a Hydra config from ``config/`` and return the merged ``DictConfig``."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    assert isinstance(cfg, DictConfig)
    return cfg


# ---------------------------------------------------------------------------
# Top-level configs
# ---------------------------------------------------------------------------


def test_pretrain_resolves() -> None:
    """Stage A pretrain config composes and exposes RFC-0008 §D2/§D4 values."""
    cfg = _compose("train/pretrain")
    # Optimizer (RFC-0008 §D4).
    assert pytest.approx(cfg.optimizer.lr) == 3e-4
    assert pytest.approx(cfg.optimizer.weight_decay) == 0.05
    # Model (RFC-0003 §D1, RFC-0008 default).
    assert cfg.model.hidden_size == 512
    # Stage A schedule (RFC-0008 §D2 / §D4).
    assert cfg.train.total_steps == 200_000
    assert cfg.train.warmup_steps == 5_000
    assert cfg.train.batch_size == 64
    assert cfg.train.grad_accum == 16
    # Loss (RFC-0008 §D3).
    assert cfg.loss.enabled is True
    assert pytest.approx(cfg.loss.lambda_) == 0.05


def test_finetune_resolves() -> None:
    """Stage B finetune config composes and exposes RFC-0008 §D2/§D5 values."""
    cfg = _compose("train/finetune")
    assert pytest.approx(cfg.optimizer.lr) == 3e-4
    assert cfg.model.hidden_size == 512
    # Stage B schedule (RFC-0008 §D2).
    assert cfg.train.total_steps == 50_000
    assert cfg.train.batch_size == 32
    # Intent-balanced sampling (RFC-0008 §D5).
    assert cfg.data.intent_balanced is True


# ---------------------------------------------------------------------------
# Ablations
# ---------------------------------------------------------------------------


def test_ablation_no_sigreg_resolves() -> None:
    """``no-sigreg`` ablation turns off the SIGReg loss."""
    cfg = _compose("train/ablations/no-sigreg")
    assert cfg.loss.enabled is False


def test_ablation_no_intent_resolves() -> None:
    """``no-intent`` ablation disables intent conditioning."""
    cfg = _compose("train/ablations/no-intent")
    assert cfg.model.intent is False


def test_ablation_no_projector_resolves() -> None:
    """``no-projector`` ablation removes the projector MLP."""
    cfg = _compose("train/ablations/no-projector")
    assert cfg.model.projector is False


def test_ablation_lambda_sweep_resolves() -> None:
    """λ sweep covers RFC-0008 §D3 alternates ``{0.005, 0.05, 0.5}``."""
    cfg_low = _compose("train/ablations/lambda-sweep/0p005")
    cfg_mid = _compose("train/ablations/lambda-sweep/0p05")
    cfg_high = _compose("train/ablations/lambda-sweep/0p5")
    assert pytest.approx(cfg_low.loss.lambda_) == 0.005
    assert pytest.approx(cfg_mid.loss.lambda_) == 0.05
    assert pytest.approx(cfg_high.loss.lambda_) == 0.5


def test_ablation_history_resolves() -> None:
    """History sweep covers ``{1, 3, 6}`` context steps."""
    cfg_h1 = _compose("train/ablations/history/H1")
    cfg_h3 = _compose("train/ablations/history/H3")
    cfg_h6 = _compose("train/ablations/history/H6")
    assert cfg_h1.model.history_len == 1
    assert cfg_h3.model.history_len == 3
    assert cfg_h6.model.history_len == 6


def test_ablation_size_resolves() -> None:
    """Size sweep covers ``small`` (256), ``base`` (512), ``large`` (768)."""
    cfg_small = _compose("train/ablations/size/small")
    cfg_base = _compose("train/ablations/size/base")
    cfg_large = _compose("train/ablations/size/large")
    assert cfg_small.model.hidden_size == 256
    assert cfg_base.model.hidden_size == 512
    assert cfg_large.model.hidden_size == 768


# ---------------------------------------------------------------------------
# Sub-configs are self-contained
# ---------------------------------------------------------------------------


def test_model_subconfig_has_target() -> None:
    """The model sub-config carries a Hydra ``_target_`` so ``instantiate`` can build it."""
    cfg = _compose("train/pretrain")
    assert "_target_" in cfg.model
    target: str = cfg.model._target_
    assert target.startswith("codingjepa.")


def test_optimizer_subconfig_has_target() -> None:
    """The optimizer sub-config carries a Hydra ``_target_`` for ``hydra.utils.instantiate``."""
    cfg = _compose("train/pretrain")
    assert "_target_" in cfg.optimizer
    target: str = cfg.optimizer._target_
    assert target.endswith("build_optimizer")


def test_pretrain_has_no_unresolved_interpolations() -> None:
    """Every interpolation in the merged pretrain config resolves to a concrete value."""
    cfg = _compose("train/pretrain")
    # ``OmegaConf.to_container(..., resolve=True)`` raises on unresolved refs.
    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert isinstance(resolved, dict)
