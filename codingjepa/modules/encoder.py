"""Transformer encoder with RoPE positional encoding (RFC-0003 §D1).

6-layer pre-norm Transformer encoder that maps BPE token sequences to a single
[CLS] embedding. Architecture mirrors LeWorldModel's ViT encoder but uses Rotary
Positional Encoding instead of learned absolute positions.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn


def _build_rope_freqs(head_dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """Compute the (max_seq_len, head_dim // 2) inverse-frequency table for RoPE.

    Returns a tensor `freqs` where `freqs[pos, i] = pos / base^(2i / head_dim)`.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    half = head_dim // 2
    # inv_freq[i] = 1 / base^(2i / head_dim) for i = 0..half-1.
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    # (max_seq_len, half)
    freqs = torch.outer(positions, inv_freq)
    return freqs


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split last dim in half and rotate: (x1, x2) -> (-x2, x1)."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional encoding to x.

    Args:
        x: (B, n_heads, L, head_dim)
        cos: (L, head_dim) — cosine of position frequencies (duplicated halves).
        sin: (L, head_dim) — sine of position frequencies (duplicated halves).

    Returns:
        Rotated tensor of same shape as x.
    """
    # Broadcast cos/sin from (L, head_dim) to (1, 1, L, head_dim).
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (_rotate_half(x) * sin)


class RoPEAttention(nn.Module):
    """Multi-head attention with Rotary Positional Encoding (RoPE)."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})")
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len

        # Fused QKV projection.
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

        # Pre-compute RoPE cos/sin tables for positions 0..max_seq_len-1.
        # freqs has shape (max_seq_len, head_dim // 2).
        # We need a (max_seq_len, head_dim) cos/sin that "duplicates" each freq
        # across the two halves of the head_dim, matching `_rotate_half` layout.
        freqs = _build_rope_freqs(self.head_dim, max_seq_len, base=rope_base)
        # Duplicate freqs across the two halves of head_dim so cos/sin align
        # with the (x1, x2) split used by _rotate_half.
        # emb[..., :half] = emb[..., half:] = freqs.
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Self-attention with RoPE.

        Args:
            x: (B, L, D)
            attn_mask: optional additive attention mask, broadcastable to
                (B, n_heads, L, L). Use -inf to mask.
            key_padding_mask: optional (B, L) bool mask where True means "padded
                token, do not attend".

        Returns:
            (B, L, D)
        """
        bsz, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        qkv = self.qkv_proj(x)  # (B, L, 3D)
        qkv = qkv.view(bsz, seq_len, 3, self.n_heads, self.head_dim)
        # (3, B, n_heads, L, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        cos = cast(torch.Tensor, self.cos_cached)[:seq_len].to(q.dtype)
        sin = cast(torch.Tensor, self.sin_cached)[:seq_len].to(q.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Build the combined attention mask if needed.
        # scaled_dot_product_attention takes an additive mask of shape
        # broadcastable to (B, n_heads, L, L) OR a boolean mask where True=keep.
        # We use the additive convention.
        mask: torch.Tensor | None = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, L) bool, True=mask out.
            # Expand to (B, 1, 1, L) and convert to additive.
            kp = key_padding_mask.to(torch.bool)
            additive = torch.zeros(bsz, 1, 1, seq_len, dtype=q.dtype, device=q.device)
            additive = additive.masked_fill(kp[:, None, None, :], float("-inf"))
            mask = additive
        if attn_mask is not None:
            if mask is None:
                mask = attn_mask
            else:
                mask = mask + attn_mask

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        # (B, n_heads, L, head_dim) -> (B, L, D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_dim)
        return cast(torch.Tensor, self.out_proj(attn_out))


class EncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer with RoPE attention."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = RoPEAttention(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm attention.
        h = self.norm1(x)
        h = self.attn(h, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.dropout1(h)
        # Pre-norm FFN.
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.dropout2(h)
        return x


class Encoder(nn.Module):
    """6-layer Transformer encoder that maps BPE token sequences to a CLS embedding.

    Vocab size is 32k + 15 special tokens = 32015. The [CLS] token is at position 0.
    Max sequence length 512 tokens.

    Parameter count: ~30M.
    """

    def __init__(
        self,
        vocab_size: int = 32_015,
        hidden_dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_dim=hidden_dim,
                    n_heads=n_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters with truncated normal / xavier conventions."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return CLS embedding of shape (B, hidden_dim).

        Args:
            token_ids: (B, L) int64 token ids. Position 0 should be the [CLS] id.
            attention_mask: optional (B, L) float mask, 1=attend, 0=mask. May also
                be passed as a bool tensor.

        Returns:
            (B, hidden_dim) hidden state at the [CLS] position.
        """
        x = self.token_embedding(token_ids)  # (B, L, D)
        x = self.embedding_dropout(x)

        # Convert attention_mask (1=attend, 0=mask) to key_padding_mask
        # (True=padded/mask) for the attention layers.
        key_padding_mask: torch.Tensor | None = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        x = self.final_norm(x)
        # Return CLS token at position 0.
        return cast(torch.Tensor, x[:, 0])


__all__ = ["Encoder", "EncoderLayer", "RoPEAttention"]
