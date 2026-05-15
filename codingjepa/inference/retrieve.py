"""FAISS retrieval with predictor-history expansion (RFC-0009 §D4)."""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
import torch
import torch.nn.functional as F

from codingjepa.model import CodingJEPA

from .index import IndexMeta


@dataclass
class RetrievalResult:
    """Raw FAISS retrieval result before reranking."""

    indices: list[int]
    cosines: list[float]
    meta: IndexMeta


def retrieve(
    source_emb: torch.Tensor,
    intent_idx: int,
    model: CodingJEPA,
    index: faiss.Index,
    index_meta: IndexMeta,
    *,
    top_m: int = 100,
) -> RetrievalResult:
    """Expand source embedding with predictor, query FAISS top-M.

    Parameters
    ----------
    source_emb:
        ``(D,)`` L2-normalized embedding of the source chunk.
    intent_idx:
        Intent index (0–7) or -1 for NONE.
    model:
        CodingJEPA model used for predictor-history expansion.
    index:
        Loaded FAISS IndexFlatIP.
    index_meta:
        Sidecar metadata for the FAISS index.
    top_m:
        Number of nearest neighbours to retrieve.

    Returns
    -------
    RetrievalResult
        FAISS indices (into the meta), cosine similarities, and the meta.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            # Expand the single source embedding to fill H history slots.
            # history_len is stored on the model as model.history_len.
            H = model.history_len
            # (D,) -> (1, H, D)
            ctx = source_emb.unsqueeze(0).unsqueeze(0).expand(1, H, -1).contiguous()

            # Obtain action / intent embedding: (1, 1, D)
            if intent_idx == -1:
                # Unconditional: use the NONE embedding
                act_emb_1d = model.action_encoder.none_embedding(1)  # (1, D)
            else:
                intent_tensor = torch.tensor([intent_idx], dtype=torch.long)
                act_emb_1d = model.action_encoder(intent_tensor)  # (1, D)
            # Broadcast to (1, H, D) to match ctx shape
            act_emb = act_emb_1d.unsqueeze(1).expand(1, H, -1).contiguous()

            # AR predictor: (1, H, D) x (1, H, D) -> (1, n_preds, D)
            pred_raw = model.predictor(ctx, act_emb)

            # Apply pred_proj to the last predicted position: (1, D)
            B, P, D = pred_raw.shape
            pred_flat = pred_raw.reshape(B * P, D)
            pred_proj_out = model.pred_proj(pred_flat).reshape(B, P, D)

            # L2-normalize the last predicted position
            pred_emb = F.normalize(pred_proj_out[:, -1, :], dim=-1)  # (1, D)
    finally:
        if was_training:
            model.train()

    # Query FAISS
    pred_np = pred_emb.cpu().numpy().astype(np.float32)  # (1, D)
    distances, faiss_indices = index.search(pred_np, top_m)  # (1, top_m) each

    # Convert to flat lists, filtering out -1 (padding when index < top_m)
    raw_indices = faiss_indices[0].tolist()
    raw_distances = distances[0].tolist()

    valid_pairs = [
        (int(i), float(d)) for i, d in zip(raw_indices, raw_distances, strict=True) if i >= 0
    ]

    if valid_pairs:
        result_indices, result_cosines = zip(*valid_pairs, strict=False)
    else:
        result_indices, result_cosines = [], []  # type: ignore[assignment]

    return RetrievalResult(
        indices=list(result_indices),
        cosines=list(result_cosines),
        meta=index_meta,
    )


__all__ = ["RetrievalResult", "retrieve"]
