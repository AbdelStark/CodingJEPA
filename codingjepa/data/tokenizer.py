"""SentencePiece BPE tokenizer for CodingJEPA (RFC-0012 §D7, D8, D10).

The tokenizer wraps a SentencePiece BPE model with:

* a 32k vocabulary (RFC-0012 §D7);
* a fixed set of special tokens (``[PAD]``, ``[CLS]``, intent markers, …)
  whose IDs are pinned in :data:`SPECIAL_TOKENS`;
* byte-fallback so any UTF-8 character is at worst byte-segmented
  (keeps the UNK rate below the §D7 threshold of 0.001);
* identity normalization, which preserves whitespace and indentation so
  Python code round-trips through ``encode``/``decode`` without losing
  block structure.

The on-disk layout produced by :meth:`Tokenizer.train` is

::

    tokenizer/v1/
        tokenizer.model        # sentencepiece BPE model
        tokenizer.vocab        # sentencepiece vocab (text)
        special_tokens.json    # SPECIAL_TOKENS map, for the manifest

so :meth:`Tokenizer.load` can rebuild the same object by reading the
model file. The ``special_tokens.json`` exists as an explicit witness
that this tokenizer commits to the special-token contract; downstream
manifest writers (RFC-0014 §D4) hash both files.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import sentencepiece as spm

# --------------------------------------------------------------------------- #
# constants                                                                   #
# --------------------------------------------------------------------------- #

# Special token definitions (RFC-0012 §D7). The numeric values here are
# load-bearing: the model card, the dataset manifest, and the trainer
# all encode `[CLS]==2`, etc.
SPECIAL_TOKENS: dict[str, int] = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "[CHUNK]": 4,
    "<DOC>": 5,
    # Intent tokens (RFC-0012 §D8): one slot per supervised label plus
    # an explicit "no label" sentinel.
    "[I_0]": 6,
    "[I_1]": 7,
    "[I_2]": 8,
    "[I_3]": 9,
    "[I_4]": 10,
    "[I_5]": 11,
    "[I_6]": 12,
    "[I_7]": 13,
    "[I_NONE]": 14,
}

VOCAB_SIZE = 32_000
MAX_TOKEN_LENGTH = 512  # RFC-0012 §D2


# --------------------------------------------------------------------------- #
# Tokenizer                                                                   #
# --------------------------------------------------------------------------- #


class Tokenizer:
    """A SentencePiece BPE tokenizer with CodingJEPA special tokens.

    Use :meth:`train` to fit a fresh tokenizer on a corpus and
    :meth:`load` to reopen one from disk. The instance is immutable —
    there is no ``update_vocab`` path because retraining the tokenizer
    invalidates every downstream artefact (RFC-0014 §D4).
    """

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(str(self.model_path))
        # Resolve special token IDs once. SentencePiece returns the
        # special tokens' IDs via PieceToId; if the tokenizer was
        # trained with the right user_defined_symbols, these match
        # SPECIAL_TOKENS exactly.
        self._special_ids: dict[str, int] = {
            piece: self._sp.PieceToId(piece) for piece in SPECIAL_TOKENS
        }

    # ----- factories ----- #

    @classmethod
    def load(cls, tokenizer_dir: Path) -> Tokenizer:
        """Load a previously-trained tokenizer from ``tokenizer/v1/``.

        Expects ``tokenizer.model`` (and, for the manifest, but not
        required to load, ``special_tokens.json``) inside ``tokenizer_dir``.
        """

        tokenizer_dir = Path(tokenizer_dir)
        model_path = tokenizer_dir / "tokenizer.model"
        if not model_path.exists():
            raise FileNotFoundError(f"tokenizer model not found: {model_path}")
        return cls(model_path)

    @classmethod
    def train(
        cls,
        corpus_files: list[Path] | None = None,
        corpus_text: str | None = None,
        output_dir: Path = Path("tokenizer/v1"),
        vocab_size: int = VOCAB_SIZE,
    ) -> Tokenizer:
        """Train a SentencePiece BPE tokenizer on ``corpus_files`` or ``corpus_text``.

        Writes ``tokenizer.model``, ``tokenizer.vocab``, and
        ``special_tokens.json`` into ``output_dir`` and returns a
        :class:`Tokenizer` already loaded from the trained model.
        """

        if corpus_files is None and corpus_text is None:
            raise ValueError("must pass corpus_files or corpus_text to train()")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # SentencePieceTrainer accepts a comma-separated list of input
        # paths or a single path. When given inline text, materialise
        # it as a temporary file so sentencepiece can ingest it.
        with tempfile.TemporaryDirectory() as tmp:
            input_paths: list[Path] = list(corpus_files or [])
            if corpus_text is not None:
                tmp_corpus = Path(tmp) / "corpus.txt"
                tmp_corpus.write_text(corpus_text, encoding="utf-8")
                input_paths.append(tmp_corpus)

            model_prefix = output_dir / "tokenizer"
            # SentencePiece always reserves ids 0/1 for pad/unk (and -1
            # for bos/eos when disabled). We rename them to ``[PAD]``
            # and ``[UNK]`` via ``pad_piece``/``unk_piece`` so that the
            # first two slots match SPECIAL_TOKENS exactly, and then
            # list the rest as ``user_defined_symbols`` so they land
            # in slots 2..len(SPECIAL_TOKENS)-1.
            extra_specials = [t for t in SPECIAL_TOKENS if t not in ("[PAD]", "[UNK]")]
            spm.SentencePieceTrainer.Train(
                input=",".join(str(p) for p in input_paths),
                model_prefix=str(model_prefix),
                vocab_size=vocab_size,
                model_type="bpe",
                character_coverage=0.9995,
                byte_fallback=True,
                pad_id=SPECIAL_TOKENS["[PAD]"],
                unk_id=SPECIAL_TOKENS["[UNK]"],
                pad_piece="[PAD]",
                unk_piece="[UNK]",
                # We use [CLS]/[SEP] instead of sentencepiece's own
                # bos/eos, so disable those slots.
                bos_id=-1,
                eos_id=-1,
                user_defined_symbols=extra_specials,
                # Preserve whitespace/indentation for code: identity
                # normalization is the only mode that doesn't fold
                # repeated spaces.
                normalization_rule_name="identity",
            )

        # Side-car JSON: a manifest-friendly view of which special
        # tokens this model commits to.
        (output_dir / "special_tokens.json").write_text(
            json.dumps(SPECIAL_TOKENS, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        return cls.load(output_dir)

    # ----- inference ----- #

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Encode ``text`` to a list of token IDs.

        With ``add_special_tokens=True`` (the default), the output is:

        * length ``MAX_TOKEN_LENGTH``,
        * ``[CLS]`` at position 0,
        * BPE pieces in the middle (truncated to fit),
        * ``[SEP]`` immediately after the last BPE piece,
        * ``[PAD]`` for the remainder.

        With ``add_special_tokens=False``, returns the raw BPE id
        sequence without padding or boundary tokens.
        """

        ids: list[int] = self._sp.EncodeAsIds(text)
        if not add_special_tokens:
            return ids

        cls_id = SPECIAL_TOKENS["[CLS]"]
        sep_id = SPECIAL_TOKENS["[SEP]"]
        pad_id = SPECIAL_TOKENS["[PAD]"]

        # Reserve two positions for [CLS] and [SEP].
        max_body = MAX_TOKEN_LENGTH - 2
        body = ids[:max_body]
        out: list[int] = [cls_id, *body, sep_id]
        # Pad up to MAX_TOKEN_LENGTH.
        if len(out) < MAX_TOKEN_LENGTH:
            out = out + [pad_id] * (MAX_TOKEN_LENGTH - len(out))
        return out

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back to text.

        Pad tokens are dropped before passing to sentencepiece so the
        decoder doesn't render trailing ``[PAD]`` markers.
        """

        pad_id = SPECIAL_TOKENS["[PAD]"]
        sep_id = SPECIAL_TOKENS["[SEP]"]
        cls_id = SPECIAL_TOKENS["[CLS]"]
        filtered = [i for i in ids if i not in (pad_id, sep_id, cls_id)]
        return str(self._sp.DecodeIds(filtered))

    def tokenize(self, text: str) -> list[str]:
        """Return SentencePiece pieces (strings) for ``text`` — useful for debugging."""

        return list(self._sp.EncodeAsPieces(text))

    # ----- diagnostics ----- #

    def unk_rate(self, texts: list[str]) -> float:
        """Fraction of tokens that come out as ``[UNK]`` across ``texts``.

        RFC-0012 §D7 requires this to stay under 0.001 on the held-out
        coverage audit set.
        """

        unk_id = SPECIAL_TOKENS["[UNK]"]
        total = 0
        unk = 0
        for text in texts:
            ids = self._sp.EncodeAsIds(text)
            total += len(ids)
            unk += sum(1 for i in ids if i == unk_id)
        if total == 0:
            return 0.0
        return unk / total

    # ----- introspection ----- #

    def special_token_id(self, token: str) -> int:
        """Return the integer id for a special token like ``'[CLS]'`` or ``'[I_0]'``."""

        if token not in SPECIAL_TOKENS:
            raise KeyError(f"unknown special token: {token!r}")
        # Prefer the actual model's id (in case sentencepiece moved a
        # piece) over the static SPECIAL_TOKENS table.
        return self._special_ids.get(token, SPECIAL_TOKENS[token])


# --------------------------------------------------------------------------- #
# audits                                                                      #
# --------------------------------------------------------------------------- #


def audit_coverage(
    tokenizer: Tokenizer, texts: list[str], threshold: float = 0.999
) -> dict[str, Any]:
    """Run the §D7 coverage audit and return a structured report.

    ``threshold`` is the *coverage* threshold (fraction of tokens that
    are NOT ``[UNK]``); ``passes`` is ``True`` when coverage >= threshold.
    """

    unk_rate = tokenizer.unk_rate(texts)
    coverage = 1.0 - unk_rate
    return {
        "unk_rate": unk_rate,
        "coverage": coverage,
        "threshold": threshold,
        "passes": coverage >= threshold,
    }


def tokenize_chunks(
    chunks: list[Any], tokenizer: Tokenizer, max_length: int = MAX_TOKEN_LENGTH
) -> list[Any]:
    """Fill ``chunk.token_ids`` for each chunk in ``chunks``.

    The BPE token count is checked *before* special tokens are added:
    if a chunk's raw BPE sequence is longer than ``max_length``
    (including the two reserved slots for ``[CLS]``/``[SEP]``), the
    chunk is marked as oversized by leaving ``token_ids`` empty. This
    is how downstream stages spot RFC-0012 §D2 violations without
    losing the chunk metadata.
    """

    out = []
    max_body = max_length - 2  # [CLS] + BPE pieces + [SEP]
    for chunk in chunks:
        raw_ids = tokenizer._sp.EncodeAsIds(chunk.source_normalized)  # noqa: SLF001
        if len(raw_ids) > max_body:
            chunk.token_ids = []
        else:
            chunk.token_ids = tokenizer.encode(chunk.source_normalized)
            # Truncate / pad to *requested* max_length, not the module
            # default. encode() always pads to MAX_TOKEN_LENGTH, so
            # slice down if the caller asked for a smaller window.
            if max_length != MAX_TOKEN_LENGTH:
                chunk.token_ids = chunk.token_ids[:max_length]
        out.append(chunk)
    return out


__all__ = [
    "MAX_TOKEN_LENGTH",
    "SPECIAL_TOKENS",
    "VOCAB_SIZE",
    "Tokenizer",
    "audit_coverage",
    "tokenize_chunks",
]
