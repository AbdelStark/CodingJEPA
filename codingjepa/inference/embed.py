"""Embedding pipeline (RFC-0009 §D2)."""

from __future__ import annotations

import torch

from codingjepa.data.normalize import normalize_chunk
from codingjepa.data.tokenizer import MAX_TOKEN_LENGTH, Tokenizer
from codingjepa.model import CodingJEPA


def embed_chunk(
    source: str,
    model: CodingJEPA,
    tokenizer: Tokenizer,
) -> torch.Tensor | None:
    """Embed source: normalize → tokenize → encode → project → L2-norm.

    Returns None on:
    - parse/compile failure (normalize_chunk returns None)
    - over-cap input (> 512 tokens)

    Parameters
    ----------
    source:
        Raw Python source code for a single chunk.
    model:
        CodingJEPA model (eval mode; model.embed handles that internally).
    tokenizer:
        Fitted tokenizer used to encode the normalized source.

    Returns
    -------
    torch.Tensor | None
        Shape ``(D,)`` L2-normalized embedding, or None if the chunk is
        unparseable or exceeds the token cap.
    """
    # Step 1: normalize — returns None on parse/compile failure
    normalized = normalize_chunk(source)
    if normalized is None:
        return None

    # Step 2: tokenize and check length cap
    token_ids = tokenizer.encode(normalized)
    if len(token_ids) > MAX_TOKEN_LENGTH:
        return None

    # Step 3: encode, project, L2-normalize via model.embed
    ids_tensor = torch.tensor([token_ids], dtype=torch.long)
    emb = model.embed(ids_tensor)  # (1, D)

    # Step 4: squeeze to (D,)
    return emb.squeeze(0)


__all__ = ["embed_chunk"]
