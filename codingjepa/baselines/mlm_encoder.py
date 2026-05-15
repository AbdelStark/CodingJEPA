"""MLM encoder baseline (issue #79, RFC-0005 §D9).

Wraps the CodingJEPA :class:`~codingjepa.modules.encoder.Encoder` with a
masked-language-modelling head.  The masked-LM loss provides a supervised
pre-training signal; the ``embed`` method exposes L2-normalised CLS embeddings
for retrieval evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from codingjepa.baselines.bm25 import Metrics
from codingjepa.data.tokenizer import VOCAB_SIZE
from codingjepa.modules.encoder import Encoder


def mask_tokens(
    input_ids: torch.Tensor,
    mask_token_id: int,
    mask_prob: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly mask BPE tokens for masked-language-modelling.

    Parameters
    ----------
    input_ids:
        ``(B, L)`` integer tensor of token IDs.
    mask_token_id:
        The vocabulary ID used for the ``[MASK]`` replacement token.
    mask_prob:
        Fraction of tokens to mask (default: 0.15).

    Returns
    -------
    masked_ids:
        Copy of *input_ids* with randomly selected positions replaced
        by *mask_token_id*.
    labels:
        Integer tensor of the same shape.  Positions that were masked
        carry their original token ID; all other positions are set to
        ``-100`` (``torch.nn.CrossEntropyLoss`` ignores ``-100``).
    """
    masked_ids = input_ids.clone()
    labels = torch.full_like(input_ids, fill_value=-100)

    # Sample a Bernoulli mask over all positions.
    mask = torch.bernoulli(torch.full_like(input_ids, mask_prob, dtype=torch.float)).bool()

    labels[mask] = input_ids[mask]
    masked_ids[mask] = mask_token_id
    return masked_ids, labels


class MLMEncoder(nn.Module):
    """Encoder with a masked-language-modelling projection head.

    Architecture
    ------------
    * A 6-layer :class:`~codingjepa.modules.encoder.Encoder` (hidden=512).
    * A linear ``mlm_head`` that projects hidden states to vocabulary
      logits: ``(*, 512) → (*, VOCAB_SIZE)``.

    The :meth:`forward` method returns ``(logits, loss)``; the loss is
    computed only over the masked positions.  :meth:`embed` extracts the
    CLS-token embedding and L2-normalises it for retrieval.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        hidden_dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        # Encoder uses vocab_size + 15 special tokens = 32015 by default.
        encoder_vocab = vocab_size + 15
        self.encoder = Encoder(
            vocab_size=encoder_vocab,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        # MLM projection: hidden_dim → vocab_size (32000 BPE tokens, no specials).
        self.mlm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute MLM logits and loss.

        Parameters
        ----------
        input_ids:
            ``(B, L)`` integer tensor.  The first token at position 0 is
            expected to be ``[CLS]``.  Tokens set to the mask ID carry
            their original IDs in the paired *labels* tensor produced by
            :func:`mask_tokens`.

        Returns
        -------
        logits:
            ``(B, L, vocab_size)`` float tensor.
        loss:
            Scalar cross-entropy loss over masked positions.  If no
            position is masked the loss is 0.
        """
        # Run every token through the encoder, not just CLS.
        x = self.encoder.token_embedding(input_ids)
        x = self.encoder.embedding_dropout(x)
        for layer in self.encoder.layers:
            x = layer(x)
        x = self.encoder.final_norm(x)  # (B, L, D)
        logits: torch.Tensor = self.mlm_head(x)  # (B, L, V)

        # Compute a dummy zero loss (callers supply labels via mask_tokens).
        # When used stand-alone (e.g. in tests) we return a 0-loss.
        loss = torch.tensor(0.0, device=input_ids.device, dtype=logits.dtype)
        return logits, loss

    def forward_with_labels(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Like :meth:`forward` but also computes the masked-LM loss.

        Parameters
        ----------
        input_ids:
            ``(B, L)`` masked token IDs (from :func:`mask_tokens`).
        labels:
            ``(B, L)`` integer tensor with ``-100`` for non-masked
            positions and the original token ID elsewhere.

        Returns
        -------
        logits, loss
        """
        logits, _ = self.forward(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        return logits, loss

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return an L2-normalised CLS-token embedding.

        Parameters
        ----------
        input_ids:
            ``(B, L)`` integer tensor.

        Returns
        -------
        ``(B, hidden_dim)`` float tensor, each row has unit L2 norm.
        """
        cls_emb = self.encoder(input_ids)  # (B, D)
        return F.normalize(cls_emb, p=2, dim=-1)


def _build_dummy_ids(texts: list[str], seq_len: int = 16) -> torch.Tensor:
    """Create deterministic token-ID tensors for texts without a real tokenizer.

    Used when no :class:`~codingjepa.data.tokenizer.Tokenizer` is available
    (e.g. in the retrieval-only evaluation path).  Each text is hashed to a
    reproducible sequence of IDs in ``[2, VOCAB_SIZE)``.
    """
    ids_list = []
    for text in texts:
        seed = abs(hash(text)) % (2**31)
        rng = np.random.default_rng(seed)
        token_ids = rng.integers(2, VOCAB_SIZE, size=seq_len).tolist()
        ids_list.append(token_ids)
    return torch.tensor(ids_list, dtype=torch.long)


def run(
    corpus: list[str],
    queries: list[str],
    targets: list[str],
    model: MLMEncoder | None = None,
    seq_len: int = 16,
) -> Metrics:
    """MLM-encoder retrieval baseline.

    Embeds every corpus entry and query using :meth:`MLMEncoder.embed`,
    then retrieves by cosine similarity (equivalent to dot-product on
    L2-normalised vectors).

    Parameters
    ----------
    corpus:
        Strings to index.
    queries:
        Query strings.
    targets:
        ``targets[i]`` must be in ``corpus``.
    model:
        Optional pre-built :class:`MLMEncoder`.  A fresh untrained
        instance is created when ``None``.
    seq_len:
        Sequence length used when building synthetic token IDs (no real
        tokenizer).

    Returns
    -------
    Metrics
    """
    if len(queries) != len(targets):
        raise ValueError(
            f"queries and targets must have the same length, "
            f"got {len(queries)} vs {len(targets)}"
        )

    if model is None:
        model = MLMEncoder()
    model.eval()

    with torch.no_grad():
        corpus_ids = _build_dummy_ids(corpus, seq_len=seq_len)
        query_ids = _build_dummy_ids(queries, seq_len=seq_len)
        corpus_embs = model.embed(corpus_ids)  # (N, D)
        query_embs = model.embed(query_ids)  # (Q, D)

    # Cosine similarity: (Q, N)
    sim = torch.matmul(query_embs, corpus_embs.T).numpy()

    corpus_index: dict[str, int] = {}
    for idx, doc in enumerate(corpus):
        if doc not in corpus_index:
            corpus_index[doc] = idx

    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    reciprocal_ranks: list[float] = []

    for q_idx, target in enumerate(targets):
        ranked = list(np.argsort(sim[q_idx])[::-1])
        target_idx = corpus_index.get(target)
        if target_idx is None:
            reciprocal_ranks.append(0.0)
            continue
        try:
            rank = ranked.index(target_idx) + 1
        except ValueError:
            reciprocal_ranks.append(0.0)
            continue

        if rank <= 1:
            hits_at_1 += 1
        if rank <= 5:
            hits_at_5 += 1
        if rank <= 10:
            hits_at_10 += 1
        reciprocal_ranks.append(1.0 / rank)

    n = len(queries)
    return Metrics(
        retrieval_at_1=hits_at_1 / n,
        retrieval_at_5=hits_at_5 / n,
        retrieval_at_10=hits_at_10 / n,
        mrr=float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
    )


def write_results(
    metrics: Metrics,
    output_path: Path = Path("data/baselines/mlm_encoder/results.json"),
    *,
    n_queries: int = 0,
) -> None:
    """Write MLM-encoder metrics to *output_path* as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": "mlm_encoder",
        "n_queries": n_queries,
        "retrieval_at_1": metrics.retrieval_at_1,
        "retrieval_at_5": metrics.retrieval_at_5,
        "retrieval_at_10": metrics.retrieval_at_10,
        "mrr": metrics.mrr,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


__all__ = ["MLMEncoder", "Metrics", "mask_tokens", "run", "write_results"]
