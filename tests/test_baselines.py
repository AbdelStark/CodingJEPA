"""Tests for Phase 4 baselines: BM25, MLM encoder, CodeBERT, and preflight check.

Issues #78–#81.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from codingjepa.baselines.bm25 import Metrics
from codingjepa.baselines.bm25 import run as bm25_run
from codingjepa.baselines.bm25 import write_results as bm25_write
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
from codingjepa.errors import ConfigError
from codingjepa.training.preflight import check_baselines_first

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_corpus(n: int = 10) -> list[str]:
    """Return *n* distinct, easily-tokenizable strings."""
    return [f"def function_{i}(x): return x + {i}" for i in range(n)]


# ---------------------------------------------------------------------------
# BM25 tests (issue #78)
# ---------------------------------------------------------------------------


def test_bm25_run_perfect_retrieval() -> None:
    """With distinct corpus entries as queries, BM25 should achieve R@1 = 1.0."""
    corpus = _make_corpus(10)
    # Use every 2nd corpus entry as a query — targets are the exact corpus strings.
    queries = corpus[:5]
    targets = corpus[:5]

    metrics = bm25_run(corpus, queries, targets)

    assert isinstance(metrics, Metrics)
    assert metrics.retrieval_at_1 == pytest.approx(1.0)
    assert metrics.retrieval_at_5 == pytest.approx(1.0)
    assert metrics.retrieval_at_10 == pytest.approx(1.0)
    assert metrics.mrr == pytest.approx(1.0)


def test_bm25_run_returns_float_metrics() -> None:
    """Metrics fields are floats in [0, 1]."""
    corpus = _make_corpus(10)
    queries = [corpus[0], corpus[2], corpus[4]]
    targets = [corpus[0], corpus[2], corpus[4]]

    metrics = bm25_run(corpus, queries, targets)

    for field in (
        metrics.retrieval_at_1,
        metrics.retrieval_at_5,
        metrics.retrieval_at_10,
        metrics.mrr,
    ):
        assert 0.0 <= field <= 1.0


def test_bm25_write_results(tmp_path: Path) -> None:
    """write_results writes a valid JSON file with all required keys."""
    metrics = Metrics(
        retrieval_at_1=0.8,
        retrieval_at_5=0.9,
        retrieval_at_10=1.0,
        mrr=0.85,
    )
    out = tmp_path / "bm25" / "results.json"
    bm25_write(metrics, out, n_queries=5)

    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["baseline"] == "bm25"
    assert payload["n_queries"] == 5
    assert payload["retrieval_at_1"] == pytest.approx(0.8)
    assert payload["retrieval_at_5"] == pytest.approx(0.9)
    assert payload["retrieval_at_10"] == pytest.approx(1.0)
    assert payload["mrr"] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# MLM encoder tests (issue #79)
# ---------------------------------------------------------------------------


def test_mlm_mask_tokens_rate() -> None:
    """mask_tokens changes approximately 15% of tokens; unmasked labels are -100."""
    torch.manual_seed(42)
    B, L = 4, 200
    input_ids = torch.randint(2, 32000, (B, L))
    mask_token_id = 99

    masked_ids, labels = mask_tokens(input_ids, mask_token_id, mask_prob=0.15)

    # Labels should be -100 where nothing was masked.
    unmasked = labels == -100
    masked = labels != -100
    assert masked.any(), "At least some tokens should be masked"
    # Original IDs preserved in labels at masked positions.
    assert torch.all(labels[masked] == input_ids[masked])
    # Unmasked positions in masked_ids equal input_ids.
    assert torch.all(masked_ids[unmasked] == input_ids[unmasked])
    # Masked positions in masked_ids become mask_token_id.
    assert torch.all(masked_ids[masked] == mask_token_id)

    # Check the mask rate is roughly right (allow ±5 pp for B*L=800 samples).
    actual_rate = masked.float().mean().item()
    assert 0.10 <= actual_rate <= 0.20, f"mask rate {actual_rate:.3f} outside expected range"


def test_mlm_forward_shapes() -> None:
    """MLMEncoder.forward returns (logits, loss) with correct shapes."""
    model = MLMEncoder()
    model.eval()
    B, L = 2, 16
    input_ids = torch.randint(0, 32015, (B, L))

    with torch.no_grad():
        logits, loss = model(input_ids)

    assert logits.shape == (B, L, 32000), f"unexpected logits shape {logits.shape}"
    assert loss.shape == (), "loss must be a scalar"


def test_mlm_embed_l2_normalized() -> None:
    """MLMEncoder.embed returns L2-normalised CLS embeddings."""
    model = MLMEncoder()
    model.eval()
    B, L = 3, 16
    input_ids = torch.randint(0, 32015, (B, L))

    with torch.no_grad():
        embs = model.embed(input_ids)

    assert embs.shape == (B, 512)
    norms = torch.linalg.norm(embs, dim=-1)
    assert torch.allclose(norms, torch.ones(B), atol=1e-5), f"embeddings are not unit-norm: {norms}"


def test_mlm_run_returns_metrics() -> None:
    """mlm_encoder.run returns Metrics with floats in [0, 1]."""
    corpus = _make_corpus(5)
    queries = corpus[:3]
    targets = corpus[:3]

    metrics = mlm_run(corpus, queries, targets)

    assert isinstance(metrics, Metrics)
    for field in (
        metrics.retrieval_at_1,
        metrics.retrieval_at_5,
        metrics.retrieval_at_10,
        metrics.mrr,
    ):
        assert isinstance(field, float)
        assert 0.0 <= field <= 1.0


def test_mlm_write_results(tmp_path: Path) -> None:
    """mlm_encoder.write_results writes a valid JSON file."""
    metrics = Metrics(
        retrieval_at_1=0.6,
        retrieval_at_5=0.8,
        retrieval_at_10=0.9,
        mrr=0.7,
    )
    out = tmp_path / "mlm_encoder" / "results.json"
    mlm_write(metrics, out, n_queries=3)

    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["baseline"] == "mlm_encoder"
    assert payload["n_queries"] == 3
    assert payload["retrieval_at_1"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# CodeBERT tests (issue #80)
# ---------------------------------------------------------------------------


def test_codebert_write_lock_file(tmp_path: Path) -> None:
    """write_lock_file creates a JSON with model and revision keys."""
    from codingjepa.baselines.codebert import (
        CODEBERT_MODEL,
        CODEBERT_REVISION,
        write_lock_file,
    )

    lock_path = tmp_path / "codebert.lock.json"
    write_lock_file(lock_path)

    assert lock_path.exists()
    payload = json.loads(lock_path.read_text())
    assert payload["model"] == CODEBERT_MODEL
    assert payload["revision"] == CODEBERT_REVISION


def test_codebert_baseline_mocked(tmp_path: Path) -> None:
    """CodeBERTBaseline initialises and embeds texts using mocked transformers."""
    hidden_size = 768
    seq_len = 4

    # Build mock tokenizer output
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.zeros(2, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(2, seq_len, dtype=torch.long),
    }

    # Build mock model output
    mock_model_output = MagicMock()
    mock_model_output.last_hidden_state = torch.randn(2, seq_len, hidden_size)
    mock_model = MagicMock()
    mock_model.return_value = mock_model_output
    mock_model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("transformers.AutoModel.from_pretrained", return_value=mock_model),
    ):
        from codingjepa.baselines.codebert import CodeBERTBaseline  # noqa: PLC0415

        baseline = CodeBERTBaseline(device="cpu")
        texts = ["def foo(): pass", "def bar(): pass"]
        embs = baseline.embed(texts)

    assert embs.shape == (2, hidden_size)
    # Each row should be L2-normalised.
    norms = np.linalg.norm(embs, axis=1)
    np.testing.assert_allclose(norms, np.ones(2), atol=1e-5)


# ---------------------------------------------------------------------------
# Preflight check_baselines_first tests (issue #81)
# ---------------------------------------------------------------------------


def test_check_baselines_first_missing(tmp_path: Path) -> None:
    """check_baselines_first raises ConfigError when results.json files are absent."""
    with pytest.raises(ConfigError, match="baseline missing"):
        check_baselines_first(baselines_dir=tmp_path)


def test_check_baselines_first_ok(tmp_path: Path) -> None:
    """check_baselines_first does not raise when all three results.json exist."""
    for sub in ("bm25", "mlm_encoder", "codebert"):
        d = tmp_path / sub
        d.mkdir()
        (d / "results.json").write_text(json.dumps({"baseline": sub, "mrr": 0.5}), encoding="utf-8")

    # Should complete without raising.
    check_baselines_first(baselines_dir=tmp_path)


def test_check_baselines_first_partial(tmp_path: Path) -> None:
    """check_baselines_first raises if only some baselines are present."""
    # Create only BM25 and MLM; CodeBERT missing.
    for sub in ("bm25", "mlm_encoder"):
        d = tmp_path / sub
        d.mkdir()
        (d / "results.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ConfigError, match="baseline missing"):
        check_baselines_first(baselines_dir=tmp_path)
