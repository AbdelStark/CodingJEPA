"""Populate MODEL_CARD.md front-matter from a fresh checkpoint + manifest.

Usage:
    python tools/model_card_update.py <checkpoint_path> <manifest_path> \\
        [--training-compute-h100-hours INT] [--seeds-reported INT]

The script does not load PyTorch — it computes the sha256 of the bytes on disk.
For a safetensors checkpoint the bytes hash IS the model identifier; for a
.ckpt file we still hash bytes (the eval harness compares the same hash).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_CARD = REPO_ROOT / "MODEL_CARD.md"
FRONT_MATTER_RE = re.compile(r"\A(---\s*\n)(.*?)(\n---\s*\n)", re.DOTALL)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_hash(manifest_path: pathlib.Path) -> str:
    """Read a manifest.lock.json and recompute the canonicalized hash spec/03 prescribes."""

    with manifest_path.open(encoding="utf-8") as fp:
        manifest = json.load(fp)
    payload = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def update_front_matter(card_text: str, updates: dict[str, str]) -> str:
    match = FRONT_MATTER_RE.match(card_text)
    if match is None:
        raise SystemExit("MODEL_CARD.md is missing YAML front-matter")
    open_marker, body, close_marker = match.group(1), match.group(2), match.group(3)
    rebuilt: list[str] = []
    seen: set[str] = set()
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            rebuilt.append(line)
            continue
        if ":" not in stripped:
            rebuilt.append(line)
            continue
        key = stripped.split(":", 1)[0].strip()
        if key in updates:
            rebuilt.append(f"{key}: {updates[key]}")
            seen.add(key)
        else:
            rebuilt.append(line)
    missing = set(updates) - seen
    if missing:
        raise SystemExit(f"MODEL_CARD.md front-matter missing keys: {sorted(missing)}")
    new_body = "\n".join(rebuilt)
    return open_marker + new_body + close_marker + card_text[match.end() :]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("--tokenizer", type=pathlib.Path, default=None)
    parser.add_argument("--training-compute-h100-hours", type=int, default=None)
    parser.add_argument("--seeds-reported", type=int, default=None)
    parser.add_argument(
        "--model-card",
        type=pathlib.Path,
        default=MODEL_CARD,
        help="Path to MODEL_CARD.md to update in place.",
    )
    args = parser.parse_args(argv)

    ckpt_hash = sha256_file(args.checkpoint)
    mani_hash = manifest_hash(args.manifest)
    tokenizer_hash = sha256_file(args.tokenizer) if args.tokenizer else None
    index_id = f"{ckpt_hash[:8]}-{mani_hash[:8]}"

    updates: dict[str, str] = {
        "checkpoint_hash": ckpt_hash,
        "manifest_hash": mani_hash,
        "index_id": index_id,
    }
    if tokenizer_hash is not None:
        updates["tokenizer_hash"] = tokenizer_hash
    if args.training_compute_h100_hours is not None:
        updates["training_compute_h100_hours"] = str(args.training_compute_h100_hours)
    if args.seeds_reported is not None:
        updates["seeds_reported"] = str(args.seeds_reported)

    card_text = args.model_card.read_text(encoding="utf-8")
    new_text = update_front_matter(card_text, updates)
    args.model_card.write_text(new_text, encoding="utf-8")
    print(f"Updated {args.model_card}: {sorted(updates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
