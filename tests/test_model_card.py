"""MODEL_CARD.md validates against data/schemas/model_card.schema.json; tools round-trip."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

from codingjepa._jsonschema import load_schema, validate_record
from codingjepa.eval.harness import parse_model_card_front_matter

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_CARD = REPO_ROOT / "MODEL_CARD.md"
UPDATE_SCRIPT = REPO_ROOT / "tools" / "model_card_update.py"


def _front_matter_as_typed_dict(card: pathlib.Path) -> dict[str, object]:
    """Promote integer-shaped strings to int for schema validation."""

    raw = parse_model_card_front_matter(card)
    out: dict[str, object] = {}
    for k, v in raw.items():
        if k in {"training_compute_h100_hours", "seeds_reported"}:
            out[k] = int(v)
        else:
            out[k] = v
    return out


def test_committed_model_card_validates() -> None:
    """The placeholder MODEL_CARD.md at repo root passes the schema (#138)."""

    schema = load_schema("model_card")
    payload = _front_matter_as_typed_dict(MODEL_CARD)
    validate_record(payload, schema)


def test_update_script_round_trips(tmp_path: pathlib.Path) -> None:
    """Run tools/model_card_update.py against synthetic artifacts; resulting card validates."""

    # Synthetic checkpoint: a small file whose sha256 is the "checkpoint_hash".
    ckpt = tmp_path / "ckpt.safetensors"
    ckpt.write_bytes(b"fake-weights")

    # Synthetic manifest with manifest_hash placeholder.
    manifest = tmp_path / "manifest.lock.json"
    manifest_payload = {
        "schema_version": "v1",
        "manifest_hash": "0" * 64,  # placeholder; the script recomputes
        "generated_at": "2026-05-15T00:00:00.000Z",
        "chunker_version": "0.1.0",
        "tokenizer_hash": "1" * 64,
        "secrets_scanner_version": "0.1.0",
        "splits_path": "data/splits/v1.lock.json",
        "repos": [],
    }
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")

    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"fake-tokenizer-bytes")

    card_copy = tmp_path / "MODEL_CARD.md"
    card_copy.write_text(MODEL_CARD.read_text(encoding="utf-8"), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(UPDATE_SCRIPT),
            str(ckpt),
            str(manifest),
            "--tokenizer",
            str(tokenizer),
            "--training-compute-h100-hours",
            "240",
            "--seeds-reported",
            "3",
            "--model-card",
            str(card_copy),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    # Reload + validate
    schema = load_schema("model_card")
    payload = _front_matter_as_typed_dict(card_copy)
    validate_record(payload, schema)
    assert payload["training_compute_h100_hours"] == 240
    assert payload["seeds_reported"] == 3
    assert payload["checkpoint_hash"] != "0" * 64  # was updated
    # index_id is checkpoint_hash[:8] + "-" + manifest_hash[:8]
    index_id = payload["index_id"]
    assert isinstance(index_id, str) and "-" in index_id


def test_update_script_rejects_missing_card(tmp_path: pathlib.Path) -> None:
    ckpt = tmp_path / "ckpt.safetensors"
    ckpt.write_bytes(b"x")
    manifest = tmp_path / "manifest.lock.json"
    manifest.write_text("{}", encoding="utf-8")
    not_a_card = tmp_path / "DOES_NOT_EXIST.md"
    result = subprocess.run(
        [
            sys.executable,
            str(UPDATE_SCRIPT),
            str(ckpt),
            str(manifest),
            "--model-card",
            str(not_a_card),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0


@pytest.mark.parametrize(
    "field",
    ["schema_version", "license", "checkpoint_hash", "manifest_hash", "tokenizer_hash", "index_id"],
)
def test_required_field_present(field: str) -> None:
    payload = _front_matter_as_typed_dict(MODEL_CARD)
    assert field in payload, f"MODEL_CARD.md missing required field: {field}"
