"""Tests for codingjepa.modules.ar_predictor (issue #60, RFC-0003 §D3)."""

from __future__ import annotations

import torch

from codingjepa.modules.ar_predictor import ARPredictor


def test_ar_predictor_output_shape() -> None:
    pred = ARPredictor()
    ctx = torch.randn(2, 3, 512)  # B=2, H=3, D=512
    act = torch.randn(2, 3, 512)
    out = pred(ctx, act)
    assert out.shape == (2, 1, 512)  # n_preds=1


def test_ar_predictor_parameter_count() -> None:
    pred = ARPredictor()
    n = sum(p.numel() for p in pred.parameters())
    assert 8_000_000 <= n <= 15_000_000  # ~10-12M


def test_ar_predictor_no_nan() -> None:
    pred = ARPredictor()
    ctx = torch.randn(4, 3, 512)
    act = torch.randn(4, 3, 512)
    out = pred(ctx, act)
    assert not torch.isnan(out).any()
