"""Frozen CodeBERT retrieval baseline (issue #80, RFC-0005 §D9).

Uses ``microsoft/codebert-base`` at a pinned revision so results are
reproducible across runs.  The model is loaded via the ``transformers``
library and kept frozen (no gradient updates).

Retrieval is performed by mean-pooling the last hidden state, L2-normalising
the result, and ranking by cosine similarity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from codingjepa.baselines.bm25 import Metrics

CODEBERT_MODEL = "microsoft/codebert-base"
CODEBERT_REVISION = "3b6e86c"


class CodeBERTBaseline:
    """Frozen CodeBERT embedding model for retrieval evaluation.

    Parameters
    ----------
    device:
        PyTorch device string (``"cpu"`` or ``"cuda"``).
    """

    def __init__(self, device: str = "cpu") -> None:
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            CODEBERT_MODEL,
            revision=CODEBERT_REVISION,
        )
        self.model = AutoModel.from_pretrained(
            CODEBERT_MODEL,
            revision=CODEBERT_REVISION,
        )
        self.model.eval()
        self.model.to(device)
        # Freeze all parameters — this is a fixed baseline, not a trained model.
        for param in self.model.parameters():
            param.requires_grad_(False)

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Mean-pool last hidden state and L2-normalise.

        Parameters
        ----------
        texts:
            Strings to embed.
        batch_size:
            Number of strings to encode per forward pass.

        Returns
        -------
        ``(N, hidden_size)`` float32 numpy array with unit L2 rows.
        """
        import torch  # noqa: PLC0415

        all_embs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
            # Mean-pool over the sequence dimension using the attention mask.
            last_hidden = out.last_hidden_state  # (B, L, H)
            attn_mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
            summed = (last_hidden * attn_mask).sum(dim=1)  # (B, H)
            counts = attn_mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
            mean_pooled = (summed / counts).cpu().numpy()  # (B, H)
            # L2 normalise each row.
            norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-9)
            all_embs.append(mean_pooled / norms)

        return np.concatenate(all_embs, axis=0)

    def run(
        self,
        corpus: list[str],
        queries: list[str],
        targets: list[str],
    ) -> Metrics:
        """CodeBERT retrieval baseline.

        Parameters
        ----------
        corpus:
            Strings to index.
        queries:
            Query strings.
        targets:
            ``targets[i]`` must be in ``corpus``.

        Returns
        -------
        Metrics
        """
        if len(queries) != len(targets):
            raise ValueError(
                f"queries and targets must have the same length, "
                f"got {len(queries)} vs {len(targets)}"
            )

        corpus_embs = self.embed(corpus)  # (N, H)
        query_embs = self.embed(queries)  # (Q, H)
        sim = query_embs @ corpus_embs.T  # (Q, N)

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
        self,
        metrics: Metrics,
        output_path: Path = Path("data/baselines/codebert/results.json"),
        *,
        n_queries: int = 0,
    ) -> None:
        """Write CodeBERT metrics to *output_path* as JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "baseline": "codebert",
            "model": CODEBERT_MODEL,
            "revision": CODEBERT_REVISION,
            "n_queries": n_queries,
            "retrieval_at_1": metrics.retrieval_at_1,
            "retrieval_at_5": metrics.retrieval_at_5,
            "retrieval_at_10": metrics.retrieval_at_10,
            "mrr": metrics.mrr,
        }
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_lock_file(
    output_path: Path = Path("data/baselines/codebert.lock.json"),
) -> None:
    """Write the CodeBERT model + revision lock file.

    The lock file records the pinned HuggingFace model and revision so
    that the experiment is reproducible without a network connection.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": CODEBERT_MODEL,
        "revision": CODEBERT_REVISION,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "CODEBERT_MODEL",
    "CODEBERT_REVISION",
    "CodeBERTBaseline",
    "Metrics",
    "write_lock_file",
]
