"""Synthetic mismatch → specific HashMismatch error; CLI maps to exit 4. RFC-0010 §D1."""

from __future__ import annotations

import pathlib
import textwrap

import pytest

from codingjepa.errors import (
    CheckpointHashMismatch,
    IndexHashMismatch,
    ManifestHashMismatch,
    UsageError,
)
from codingjepa.eval.harness import (
    HashTriple,
    check_hashes,
    format_hash_diff,
    hash_triple_from_model_card,
    parse_model_card_front_matter,
)

_GOOD = HashTriple(
    manifest_hash="a" * 64,
    checkpoint_hash="b" * 64,
    index_id="aaaaaaaa-bbbbbbbb",
)


def test_matching_hashes_pass() -> None:
    check_hashes(_GOOD, _GOOD)


def test_manifest_drift_raises_first() -> None:
    runtime = HashTriple(
        manifest_hash="c" * 64,
        checkpoint_hash="b" * 64,
        index_id="aaaaaaaa-bbbbbbbb",
    )
    with pytest.raises(ManifestHashMismatch) as exc:
        check_hashes(_GOOD, runtime)
    assert exc.value.context["expected"] == "a" * 64
    assert exc.value.context["actual"] == "c" * 64
    assert exc.value.code == "E_MANIFEST_HASH_MISMATCH"


def test_checkpoint_drift_raises() -> None:
    runtime = HashTriple(
        manifest_hash="a" * 64,
        checkpoint_hash="d" * 64,
        index_id="aaaaaaaa-bbbbbbbb",
    )
    with pytest.raises(CheckpointHashMismatch) as exc:
        check_hashes(_GOOD, runtime)
    assert exc.value.code == "E_CHECKPOINT_HASH_MISMATCH"


def test_index_drift_raises() -> None:
    runtime = HashTriple(
        manifest_hash="a" * 64,
        checkpoint_hash="b" * 64,
        index_id="ffffffff-eeeeeeee",
    )
    with pytest.raises(IndexHashMismatch) as exc:
        check_hashes(_GOOD, runtime)
    assert exc.value.code == "E_INDEX_HASH_MISMATCH"


def test_format_hash_diff_shape() -> None:
    runtime = HashTriple(
        manifest_hash="c" * 64,
        checkpoint_hash="b" * 64,
        index_id="aaaaaaaa-bbbbbbbb",
    )
    try:
        check_hashes(_GOOD, runtime)
    except ManifestHashMismatch as exc:
        line = format_hash_diff(exc)
    assert "E_MANIFEST_HASH_MISMATCH" in line
    assert "expected=" in line
    assert "actual=" in line


# ---- MODEL_CARD.md front-matter parser -------------------------------------


_GOOD_CARD = textwrap.dedent("""\
    ---
    schema_version: v1
    license: apache-2.0
    checkpoint_hash: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    manifest_hash: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    tokenizer_hash: cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    index_id: aaaaaaaa-bbbbbbbb
    training_compute_h100_hours: 240
    seeds_reported: 3
    ---

    # Model card body markdown follows...
    """)


def test_parse_front_matter_extracts_kv(tmp_path: pathlib.Path) -> None:
    card = tmp_path / "MODEL_CARD.md"
    card.write_text(_GOOD_CARD, encoding="utf-8")
    fm = parse_model_card_front_matter(card)
    assert fm["schema_version"] == "v1"
    assert fm["index_id"] == "aaaaaaaa-bbbbbbbb"
    assert fm["seeds_reported"] == "3"


def test_quoted_values_are_unquoted(tmp_path: pathlib.Path) -> None:
    card = tmp_path / "MODEL_CARD.md"
    card.write_text(
        textwrap.dedent("""\
            ---
            license: "apache-2.0"
            index_id: 'aaaaaaaa-bbbbbbbb'
            ---
            """),
        encoding="utf-8",
    )
    fm = parse_model_card_front_matter(card)
    assert fm["license"] == "apache-2.0"
    assert fm["index_id"] == "aaaaaaaa-bbbbbbbb"


def test_hash_triple_from_card_round_trip(tmp_path: pathlib.Path) -> None:
    card = tmp_path / "MODEL_CARD.md"
    card.write_text(_GOOD_CARD, encoding="utf-8")
    triple = hash_triple_from_model_card(card)
    assert triple.manifest_hash == "a" * 64
    assert triple.checkpoint_hash == "b" * 64
    assert triple.index_id == "aaaaaaaa-bbbbbbbb"


def test_missing_front_matter_fails(tmp_path: pathlib.Path) -> None:
    card = tmp_path / "MODEL_CARD.md"
    card.write_text("# Just a heading, no YAML\n", encoding="utf-8")
    with pytest.raises(UsageError):
        parse_model_card_front_matter(card)


def test_missing_required_hashes_fails(tmp_path: pathlib.Path) -> None:
    card = tmp_path / "MODEL_CARD.md"
    card.write_text(
        textwrap.dedent("""\
            ---
            schema_version: v1
            ---
            """),
        encoding="utf-8",
    )
    with pytest.raises(UsageError) as exc:
        hash_triple_from_model_card(card)
    missing = exc.value.context["missing"]
    assert "manifest_hash" in missing
    assert "checkpoint_hash" in missing
    assert "index_id" in missing


def test_nested_yaml_rejected(tmp_path: pathlib.Path) -> None:
    """Lines that are not flat `key: value` (lists / nested) raise UsageError."""

    card = tmp_path / "MODEL_CARD.md"
    card.write_text(
        textwrap.dedent("""\
            ---
            schema_version: v1
              - this is a list item, not flat key:value
            ---
            """),
        encoding="utf-8",
    )
    with pytest.raises(UsageError):
        parse_model_card_front_matter(card)
