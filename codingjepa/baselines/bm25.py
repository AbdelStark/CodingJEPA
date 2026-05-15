"""BM25 retrieval baseline over BPE tokens (issue #78, RFC-0005 §D9).

Tokenizes corpus and query strings using the CodingJEPA SentencePiece BPE
tokenizer, converts token IDs to strings for the BM25 corpus, and evaluates
Retrieval@k and MRR against the provided ground-truth targets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from codingjepa.data.tokenizer import Tokenizer


@dataclass
class Metrics:
    """Retrieval metrics for a baseline run."""

    retrieval_at_1: float
    retrieval_at_5: float
    retrieval_at_10: float
    mrr: float


def _tokenize_for_bm25(texts: list[str], tokenizer: Tokenizer) -> list[list[str]]:
    """Encode each text with the BPE tokenizer and return token-ID strings."""
    result: list[list[str]] = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        result.append([str(t) for t in ids])
    return result


def run(
    corpus: list[str],
    queries: list[str],
    targets: list[str],
    tokenizer: Tokenizer | None = None,
) -> Metrics:
    """BM25 baseline over BPE tokens.

    Parameters
    ----------
    corpus:
        All ``chunk_after`` strings used to build the BM25 index.
    queries:
        ``chunk_before`` strings — one per query.
    targets:
        ``targets[i]`` is the correct answer for ``queries[i]`` and
        must be a member of ``corpus``.
    tokenizer:
        Optional :class:`~codingjepa.data.tokenizer.Tokenizer` instance.
        When ``None``, strings are split on whitespace (useful in unit
        tests that do not have a trained SPM model on disk).

    Returns
    -------
    Metrics
        Retrieval@1/5/10 and MRR computed over all queries.
    """
    if len(queries) != len(targets):
        raise ValueError(
            f"queries and targets must have the same length, "
            f"got {len(queries)} vs {len(targets)}"
        )

    # Tokenize corpus and queries.
    if tokenizer is not None:
        tokenized_corpus = _tokenize_for_bm25(corpus, tokenizer)
        tokenized_queries = _tokenize_for_bm25(queries, tokenizer)
    else:
        tokenized_corpus = [text.split() for text in corpus]
        tokenized_queries = [q.split() for q in queries]

    bm25 = BM25Okapi(tokenized_corpus)

    # Build a lookup from corpus string → index (first occurrence wins).
    corpus_index: dict[str, int] = {}
    for idx, doc in enumerate(corpus):
        if doc not in corpus_index:
            corpus_index[doc] = idx

    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    reciprocal_ranks: list[float] = []

    for tq, target in zip(tokenized_queries, targets, strict=True):
        scores = bm25.get_scores(tq)
        ranked = list(np.argsort(scores)[::-1])

        target_idx = corpus_index.get(target)
        if target_idx is None:
            reciprocal_ranks.append(0.0)
            continue

        try:
            rank = ranked.index(target_idx) + 1  # 1-based
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
    output_path: Path = Path("data/baselines/bm25/results.json"),
    *,
    n_queries: int = 0,
) -> None:
    """Write BM25 metrics to *output_path* as JSON.

    The JSON schema is:

    .. code-block:: json

        {
            "baseline": "bm25",
            "n_queries": <int>,
            "retrieval_at_1": <float>,
            "retrieval_at_5": <float>,
            "retrieval_at_10": <float>,
            "mrr": <float>
        }
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": "bm25",
        "n_queries": n_queries,
        "retrieval_at_1": metrics.retrieval_at_1,
        "retrieval_at_5": metrics.retrieval_at_5,
        "retrieval_at_10": metrics.retrieval_at_10,
        "mrr": metrics.mrr,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


__all__ = ["Metrics", "run", "write_results"]
