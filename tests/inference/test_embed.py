"""Tests for codingjepa.inference.embed (#82)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F

from codingjepa.inference.embed import embed_chunk


def _make_mock_tokenizer(token_ids: list[int]) -> MagicMock:
    """Build a tokenizer stub that always returns *token_ids* from encode()."""
    tok = MagicMock()
    tok.encode.return_value = token_ids
    return tok


def _make_mock_model(embed_dim: int = 16) -> MagicMock:
    """Build a model stub whose embed() returns a random L2-normalized (1, D) tensor."""
    model = MagicMock()
    raw = torch.randn(1, embed_dim)
    normed = F.normalize(raw, dim=-1)
    model.embed.return_value = normed
    return model


class TestEmbedHappyPath:
    def test_returns_tensor(self) -> None:
        tokenizer = _make_mock_tokenizer(list(range(10)))
        model = _make_mock_model(embed_dim=16)
        result = embed_chunk("def f(): pass", model, tokenizer)
        assert result is not None
        assert isinstance(result, torch.Tensor)

    def test_shape_is_1d(self) -> None:
        tokenizer = _make_mock_tokenizer(list(range(10)))
        model = _make_mock_model(embed_dim=16)
        result = embed_chunk("def f(): pass", model, tokenizer)
        assert result is not None
        assert result.dim() == 1

    def test_l2_norm_approximately_one(self) -> None:
        tokenizer = _make_mock_tokenizer(list(range(10)))
        model = _make_mock_model(embed_dim=32)
        result = embed_chunk("x = 1\n", model, tokenizer)
        assert result is not None
        norm = result.norm().item()
        assert abs(norm - 1.0) < 1e-5

    def test_model_embed_called_with_tensor(self) -> None:
        token_ids = [2, 5, 6, 3]  # [CLS], <DOC>, [I_0], [SEP]
        tokenizer = _make_mock_tokenizer(token_ids)
        model = _make_mock_model(embed_dim=8)
        embed_chunk("x = 1\n", model, tokenizer)
        model.embed.assert_called_once()
        call_arg = model.embed.call_args[0][0]
        assert isinstance(call_arg, torch.Tensor)
        assert call_arg.shape[0] == 1  # batch dim


class TestEmbedParseFail:
    def test_invalid_syntax_returns_none(self) -> None:
        tokenizer = _make_mock_tokenizer([1, 2, 3])
        model = _make_mock_model()
        # normalize_chunk will fail to parse/compile this
        result = embed_chunk("def (broken: syntax @@@", model, tokenizer)
        assert result is None

    def test_normalize_chunk_none_propagates(self) -> None:
        """If normalize_chunk returns None, embed_chunk should return None."""
        tokenizer = _make_mock_tokenizer([1, 2, 3])
        model = _make_mock_model()
        with patch("codingjepa.inference.embed.normalize_chunk", return_value=None):
            result = embed_chunk("any source", model, tokenizer)
        assert result is None
        model.embed.assert_not_called()


class TestEmbedOverCap:
    def test_over_512_tokens_returns_none(self) -> None:
        # 600 token IDs - exceeds MAX_TOKEN_LENGTH=512
        tokenizer = _make_mock_tokenizer(list(range(600)))
        model = _make_mock_model()
        # Use valid Python so normalize_chunk passes
        result = embed_chunk("x = 1\n", model, tokenizer)
        assert result is None
        model.embed.assert_not_called()

    def test_exactly_512_tokens_passes(self) -> None:
        tokenizer = _make_mock_tokenizer(list(range(512)))
        model = _make_mock_model(embed_dim=8)
        result = embed_chunk("x = 1\n", model, tokenizer)
        assert result is not None

    def test_513_tokens_returns_none(self) -> None:
        tokenizer = _make_mock_tokenizer(list(range(513)))
        model = _make_mock_model()
        result = embed_chunk("x = 1\n", model, tokenizer)
        assert result is None
