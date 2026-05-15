"""Top-level CodingJEPA model (RFC-0003 §D7, §D10). Implements issue #64.

Mirrors LeWorldModel's ``JEPA`` structure: a single network composed of
encoder + projector + ARPredictor + pred_proj + intent embedder + SIGReg.
There is no EMA, no teacher / student split, and no separate target encoder.

Forward pass (RFC-0003 §D7)::

    emb_all  = encoder(token_ids)              for each of S chunks → (B, S, D)
    ctx_emb  = projector(emb_all[:, :H])        (B, H, D)
    tgt_emb  = projector(emb_all[:, -n_preds:]).detach()  stop-grad
    act_emb  = intent_embedder(intent_idx)      (B, H, D)  broadcast if needed
    pred_emb = pred_proj(predictor(ctx_emb, act_emb))
    pred_loss   = MSE(pred_emb, tgt_emb)
    sigreg_loss = sigreg(projector(emb_all.view(-1, D)))
    loss = pred_loss + λ * sigreg_loss

See ``docs/spec/02-public-api.md`` for the public API surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:  # pragma: no cover - import-cycle / type-only imports
    from codingjepa.modules.ar_predictor import ARPredictor
    from codingjepa.modules.encoder import Encoder
    from codingjepa.modules.intent_embedder import IntentEmbedder
    from codingjepa.modules.pred_proj import PredProj
    from codingjepa.modules.projector import Projector
    from codingjepa.modules.sigreg import SIGReg


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ForwardResult:
    """Result of a :class:`CodingJEPA` forward pass.

    Attributes
    ----------
    pred_loss:
        Mean-squared error between predicted and target embeddings.
        ``MSE(pred_emb, tgt_emb.detach())``. Scalar tensor.
    sigreg_loss:
        SIGReg loss on the projector output over all chunk embeddings.
        Scalar tensor.
    loss:
        ``pred_loss + sigreg_lambda * sigreg_loss``. Scalar tensor.
    pred_emb:
        Predicted embeddings of shape ``(B, n_preds, D)``.
    tgt_emb:
        Target embeddings of shape ``(B, n_preds, D)`` (stop-grad).
    ctx_emb:
        Context embeddings (post-projector) of shape ``(B, H, D)``.
    """

    pred_loss: torch.Tensor
    sigreg_loss: torch.Tensor
    loss: torch.Tensor
    pred_emb: torch.Tensor
    tgt_emb: torch.Tensor
    ctx_emb: torch.Tensor


# ---------------------------------------------------------------------------
# Internal default implementations
#
# The companion modules in ``codingjepa.modules`` are being implemented in
# parallel (issues #58–#63). The classes below provide concrete defaults that
# satisfy the RFC-0003 contract so that :func:`build_model` produces a working
# model today. When the parallel implementations land, :func:`build_model`
# should be updated to use them directly; the :class:`CodingJEPA` __init__
# already accepts any module that conforms to the expected forward signatures.
# ---------------------------------------------------------------------------


class _DefaultEncoder(nn.Module):
    """Default Python-token encoder used when no encoder is supplied.

    RFC-0003 §D1: 6-layer pre-norm transformer, hidden 512, 8 heads, FFN 2048,
    GELU, dropout 0.1. Returns the hidden state at position 0 (analogous to
    LeWM's ``output.last_hidden_state[:, 0]``).

    This is a minimal stand-in: it uses learned absolute positional embeddings
    instead of RoPE. The parameter count lands around ~30M as targeted in §D9.
    """

    def __init__(
        self,
        vocab_size: int = 32_015,
        embed_dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``(B, D)`` CLS embedding for ``(B, L)`` token IDs."""
        B, L = token_ids.shape
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, L)
        x = self.token_emb(token_ids) + self.pos_emb(positions)
        # ``TransformerEncoder`` expects True for positions to MASK OUT.
        key_padding_mask: torch.Tensor | None = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        x = cast(torch.Tensor, self.transformer(x, src_key_padding_mask=key_padding_mask))
        # Hidden state at position 0 (analogous to [CLS]).
        return x[:, 0, :]


class _DefaultProjector(nn.Module):
    """RFC-0003 §D2 / §D4 projector.

    ``Linear(D, 2048) → BatchNorm1d → ReLU → Linear(2048, D)``. Used both for
    the main projector and ``pred_proj`` (same shape, distinct parameters).
    """

    def __init__(self, embed_dim: int = 512, hidden_dim: int = 2048) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``(..., D)`` → ``(..., D)`` through the MLP.

        ``BatchNorm1d`` expects ``(N, C)``; we flatten leading dims and restore
        them on the way out so the projector can be applied to ``(B, D)``,
        ``(B, S, D)``, or any other tensor whose last dim is the channel.
        """
        orig_shape = x.shape
        flat = x.reshape(-1, orig_shape[-1])
        h = self.fc1(flat)
        h = self.bn(h)
        h = self.relu(h)
        out = self.fc2(h)
        return cast(torch.Tensor, out.reshape(orig_shape))


class _DefaultIntentEmbedder(nn.Module):
    """RFC-0003 §D5 intent embedder.

    ``nn.Embedding(9, embed_dim)`` over the 8 named intents plus ``[I_NONE]``
    at index 8.
    """

    def __init__(self, n_intents: int = 9, embed_dim: int = 512) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_intents, embed_dim)

    def forward(self, intent_idx: torch.Tensor) -> torch.Tensor:
        """Embed intent indices. Accepts ``(B,)`` or ``(B, H)``."""
        return cast(torch.Tensor, self.embedding(intent_idx))


class _DefaultSIGReg(nn.Module):
    """Default Sliced Isotropic Gaussian Regularizer (RFC-0003 §D6).

    Projects the batch of embeddings onto ``K`` random unit directions; for
    each direction, penalizes deviation from ``N(0, 1/d)``. Slices are
    refreshed at every call.

    The implementation here is intentionally a minimal stand-in: it computes
    the squared deviation of the per-slice mean from 0 and the per-slice
    variance from ``1/d``. Issue #63 will replace this with the full SIGReg
    once it lands.
    """

    def __init__(self, embed_dim: int = 512, n_slices: int = 256) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_slices = n_slices

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """Return a scalar regularization loss given ``(N, D)`` embeddings."""
        if emb.dim() != 2:
            emb = emb.reshape(-1, emb.shape[-1])
        N, D = emb.shape
        # Sample K random unit-norm directions.
        slices = torch.randn(D, self.n_slices, device=emb.device, dtype=emb.dtype)
        slices = slices / (slices.norm(dim=0, keepdim=True) + 1e-8)
        # (N, K) projections; for unit-norm slices over N(0, 1/d) batched
        # embeddings, the target distribution is N(0, 1/d).
        projections = emb @ slices
        target_var = 1.0 / D
        mean = projections.mean(dim=0)
        var = projections.var(dim=0, unbiased=False)
        return cast(torch.Tensor, mean.pow(2).mean() + (var - target_var).pow(2).mean())


class _DefaultPredProj(_DefaultProjector):
    """Same shape as :class:`_DefaultProjector`; distinct parameters (§D4)."""


# ---------------------------------------------------------------------------
# Public model
# ---------------------------------------------------------------------------


class CodingJEPA(nn.Module):
    """Top-level CodingJEPA model (RFC-0003 §D7, §D10).

    Mirrors LeWM's ``JEPA`` structure: a single network with encoder +
    projector + ARPredictor + pred_proj + intent embedder + SIGReg. No EMA,
    no teacher / student split.

    Parameters
    ----------
    encoder:
        Token → CLS encoder, ``(B, L) → (B, D)``. RFC-0003 §D1.
    predictor:
        Autoregressive transformer predictor, ``(B, H, D), (B, H, D) →
        (B, n_preds, D)``. RFC-0003 §D3.
    action_encoder:
        Intent / action embedder. RFC-0003 §D5.
    projector:
        Shared projector applied to context + target embeddings. RFC-0003 §D2.
    pred_proj:
        Predictor-output projector. RFC-0003 §D4.
    sigreg:
        Sliced Isotropic Gaussian Regularizer. RFC-0003 §D6.
    history_len:
        Predictor history length ``H``. Default ``3``.
    n_preds:
        Number of predicted steps. Default ``1``.
    sigreg_lambda:
        Weight ``λ`` on the SIGReg loss in the total. Default ``0.05``.
    """

    def __init__(
        self,
        encoder: Encoder,
        predictor: ARPredictor,
        action_encoder: IntentEmbedder,
        projector: Projector,
        pred_proj: PredProj,
        sigreg: SIGReg,
        *,
        history_len: int = 3,
        n_preds: int = 1,
        sigreg_lambda: float = 0.05,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector
        self.pred_proj = pred_proj
        self.sigreg = sigreg

        self.history_len = history_len
        self.n_preds = n_preds
        self.sigreg_lambda = sigreg_lambda

    # ------------------------------------------------------------------
    # Public surface (RFC-0003 §D10)
    # ------------------------------------------------------------------

    def encode(self, chunk_tokens: torch.Tensor) -> torch.Tensor:
        """Encode a single chunk ``(B, L) → (B, D)`` CLS embedding."""
        return cast(torch.Tensor, self.encoder(chunk_tokens))

    def predict(self, ctx_emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """Predict next embedding. ``(B, H, D), (B, H, D) → (B, n_preds, D)``."""
        return cast(torch.Tensor, self.predictor(ctx_emb, act_emb))

    def embed(self, chunk_tokens: torch.Tensor) -> torch.Tensor:
        """Encode + project + L2-normalize for retrieval. ``(B, L) → (B, D)``.

        Used by the FAISS index in the retrieval-and-rerank decode path
        (RFC-0003 §D8).
        """
        # eval-mode to avoid BatchNorm1d issues with N=1 batches and to keep
        # the embedding deterministic for retrieval.
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                emb = self.encoder(chunk_tokens)
                projected = self.projector(emb)
                return F.normalize(projected, dim=-1)
        finally:
            if was_training:
                self.train()

    # ------------------------------------------------------------------
    # Full LeWM-style forward
    # ------------------------------------------------------------------

    def forward(
        self,
        token_ids: torch.Tensor,
        intent_idx: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> ForwardResult:
        """Full JEPA forward pass.

        Parameters
        ----------
        token_ids:
            ``(B, S, L)`` token IDs for ``S = H + n_preds`` chunks.
        intent_idx:
            ``(B,)`` or ``(B, H)`` intent indices. ``(B,)`` is broadcast across
            the history dimension.
        attention_mask:
            Optional ``(B, S, L)`` attention mask. ``1`` for real tokens, ``0``
            for padding.
        """
        if token_ids.dim() != 3:
            raise ValueError(f"token_ids must be (B, S, L); got shape {tuple(token_ids.shape)}")
        B, S, _L = token_ids.shape
        H = self.history_len
        P = self.n_preds
        if S < H + P:
            raise ValueError(f"Need at least history_len + n_preds = {H + P} chunks, got S = {S}")

        # 1. Encode each chunk separately so any padding mask can be applied
        #    per-chunk. Some encoders may not accept a ``mask`` kwarg, so we
        #    only forward it when explicitly provided.
        emb_list: list[torch.Tensor] = []
        for s in range(S):
            chunk_ids = token_ids[:, s, :]
            if attention_mask is not None:
                chunk_emb = self.encoder(chunk_ids, attention_mask[:, s, :])
            else:
                chunk_emb = self.encoder(chunk_ids)
            emb_list.append(chunk_emb)
        emb_all = torch.stack(emb_list, dim=1)  # (B, S, D)

        # 2. Project context + targets. The projector is shared (RFC-0003 §D2).
        #    BatchNorm1d requires 2-D input, so we flatten/unflatten around H/P dims.
        D = emb_all.shape[-1]
        ctx_emb_raw = emb_all[:, :H]  # (B, H, D)
        tgt_emb_raw = emb_all[:, -P:]  # (B, n_preds, D)
        ctx_emb = self.projector(ctx_emb_raw.reshape(B * H, D)).reshape(B, H, D)
        tgt_emb = self.projector(tgt_emb_raw.reshape(B * P, D)).reshape(B, P, D).detach()

        # 3. Action / intent embeddings, broadcast across the H positions.
        act_emb = self._compute_action_embeddings(intent_idx, B, H)

        # 4. Predictor + pred_proj.
        pred_raw = self.predictor(ctx_emb, act_emb)  # (B, n_preds, D)
        # pred_proj also needs 2-D input
        pred_emb = self.pred_proj(pred_raw.reshape(B * P, D)).reshape(B, P, D)

        # 5. Losses.
        pred_loss = F.mse_loss(pred_emb, tgt_emb)
        # SIGReg over all S embeddings (flatten B*S first, then project)
        sigreg_input = self.projector(emb_all.reshape(B * S, D))
        sigreg_loss = self.sigreg(sigreg_input)
        loss = pred_loss + self.sigreg_lambda * sigreg_loss

        return ForwardResult(
            pred_loss=pred_loss,
            sigreg_loss=sigreg_loss,
            loss=loss,
            pred_emb=pred_emb,
            tgt_emb=tgt_emb,
            ctx_emb=ctx_emb,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_action_embeddings(
        self,
        intent_idx: torch.Tensor,
        batch_size: int,
        history_len: int,
    ) -> torch.Tensor:
        """Return ``(B, H, D)`` action embeddings.

        Accepts ``intent_idx`` of shape ``(B,)``, ``(B, H)``, or
        ``(B, history_len)``. ``(B,)`` is broadcast to ``(B, H)`` by repeating
        the per-sample index across the history dimension.
        """
        if intent_idx.dim() == 1:
            if intent_idx.shape[0] != batch_size:
                raise ValueError(
                    f"intent_idx batch dim {intent_idx.shape[0]} != token batch {batch_size}"
                )
            intent_h = intent_idx.unsqueeze(1).expand(batch_size, history_len)
        elif intent_idx.dim() == 2:
            if intent_idx.shape != (batch_size, history_len):
                raise ValueError(
                    "intent_idx must be (B,) or (B, history_len); got shape "
                    f"{tuple(intent_idx.shape)}"
                )
            intent_h = intent_idx
        else:
            raise ValueError(f"intent_idx must have 1 or 2 dims; got {intent_idx.dim()}")
        return cast(torch.Tensor, self.action_encoder(intent_h))  # (B, H, D)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_model(
    *,
    vocab_size: int = 32_015,
    embed_dim: int = 512,
    history_len: int = 3,
    n_preds: int = 1,
    sigreg_lambda: float = 0.05,
) -> CodingJEPA:
    """Build a :class:`CodingJEPA` with the RFC-0003 default hyperparameters.

    This factory wires the public RFC-0003 architecture using the local
    default implementations defined in this module. As the parallel modules
    (issues #58–#63) land, this function can be swapped to use them.
    """
    from codingjepa.modules.ar_predictor import ARPredictor as _ARPredictor
    from codingjepa.modules.encoder import Encoder as _Encoder
    from codingjepa.modules.intent_embedder import IntentEmbedder as _IntentEmbedder
    from codingjepa.modules.pred_proj import PredProj as _PredProj
    from codingjepa.modules.projector import Projector as _Projector
    from codingjepa.modules.sigreg import SIGReg as _SIGReg

    encoder = _Encoder(vocab_size=vocab_size, hidden_dim=embed_dim)
    projector = _Projector(input_dim=embed_dim, output_dim=embed_dim)
    pred_proj = _PredProj(input_dim=embed_dim, output_dim=embed_dim)
    action_encoder = _IntentEmbedder(embed_dim=embed_dim)
    sigreg = _SIGReg(embed_dim=embed_dim)
    predictor = _ARPredictor(
        embed_dim=embed_dim,
        history_len=history_len,
        n_preds=n_preds,
    )

    return CodingJEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
        sigreg=sigreg,
        history_len=history_len,
        n_preds=n_preds,
        sigreg_lambda=sigreg_lambda,
    )


__all__ = ["CodingJEPA", "ForwardResult", "build_model"]
