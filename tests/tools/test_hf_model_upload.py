"""Tests for ``tools/release/hf_model_upload.py`` — HF Hub upload script (#126).

Covers dry-run behavior, missing-file handling, content-hash idempotency, and
the secret-handling contract (``HF_TOKEN`` is never echoed).

We never hit the network: the live upload code path is only invoked when
``--dry-run`` is absent, and the tests below always run with ``--dry-run``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "release" / "hf_model_upload.py"


def _make_checkpoint(tmp_path: Path) -> Path:
    """Build a fake checkpoint directory with the four required safetensors files."""

    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()
    for name in ("encoder", "projector", "predictor", "pred_proj"):
        (ckpt / f"{name}.safetensors").write_bytes(b"dummy-" + name.encode())
    return ckpt


def _make_repo_root(tmp_path: Path) -> Path:
    """Build a fake repository root containing MODEL_CARD.md, LICENSES/, tokenizer/."""

    root = tmp_path / "repo"
    root.mkdir()
    (root / "MODEL_CARD.md").write_text("# Fake card\n", encoding="utf-8")
    (root / "LICENSES").mkdir()
    (root / "LICENSES" / "Apache-2.0.txt").write_text("Apache-2.0\n", encoding="utf-8")
    (root / "tokenizer").mkdir()
    (root / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer" / "tokenizer.model").write_bytes(b"\x00\x01")
    return root


def _run(*args: str | Path, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *[str(a) for a in args]],
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_script_exists() -> None:
    """The script must exist at the documented path."""

    assert SCRIPT.exists(), f"Expected script at {SCRIPT}"


def test_dry_run_lists_all_files(tmp_path: Path) -> None:
    """``--dry-run`` enumerates every local file and its target HF path."""

    repo = _make_repo_root(tmp_path)
    ckpt = _make_checkpoint(repo)

    result = _run(
        "--checkpoint",
        ckpt,
        "--repo-id",
        "CodingJEPA/coding-jepa-v1",
        "--dry-run",
        cwd=repo,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    out = result.stdout
    # Each safetensors file uploaded to the repo root.
    for name in ("encoder", "projector", "predictor", "pred_proj"):
        assert f"{name}.safetensors" in out, f"Missing {name} in dry-run output:\n{out}"
    # Tokenizer files mapped under tokenizer/.
    assert "tokenizer/tokenizer.json" in out
    assert "tokenizer/tokenizer.model" in out
    # Licenses mapped under LICENSES/.
    assert "LICENSES/Apache-2.0.txt" in out
    # MODEL_CARD.md renamed to README.md for HF Hub.
    assert "README.md" in out
    # Every dry-run line is tagged.
    assert "[DRY-RUN]" in out


def test_dry_run_default_repo_id(tmp_path: Path) -> None:
    """The default ``--repo-id`` is ``CodingJEPA/coding-jepa-v1``."""

    repo = _make_repo_root(tmp_path)
    ckpt = _make_checkpoint(repo)

    result = _run("--checkpoint", ckpt, "--dry-run", cwd=repo)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CodingJEPA/coding-jepa-v1" in result.stdout


def test_dry_run_does_not_leak_token(tmp_path: Path, monkeypatch) -> None:
    """``HF_TOKEN`` is never echoed in stdout/stderr, even in dry-run mode."""

    repo = _make_repo_root(tmp_path)
    ckpt = _make_checkpoint(repo)

    secret = "hf_NEVER_LOG_THIS_TOKEN_1234567890"
    env = {**__import__("os").environ, "HF_TOKEN": secret}

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--checkpoint", str(ckpt), "--dry-run"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert secret not in proc.stdout
    assert secret not in proc.stderr


def test_missing_required_files_exits_nonzero(tmp_path: Path) -> None:
    """Missing a required safetensors file is exit 1 in real-upload mode."""

    repo = _make_repo_root(tmp_path)
    ckpt = repo / "broken"
    ckpt.mkdir()
    # Only one of four required files present.
    (ckpt / "encoder.safetensors").write_bytes(b"e")

    # Without --dry-run, the script must refuse to proceed.
    result = _run("--checkpoint", ckpt, cwd=repo)
    assert result.returncode == 1, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "missing" in combined.lower() or "required" in combined.lower()


def test_missing_required_files_ok_in_dry_run(tmp_path: Path) -> None:
    """``--dry-run`` still lists what *would* be uploaded even with missing files."""

    repo = _make_repo_root(tmp_path)
    ckpt = repo / "partial"
    ckpt.mkdir()
    (ckpt / "encoder.safetensors").write_bytes(b"e")

    result = _run("--checkpoint", ckpt, "--dry-run", cwd=repo)
    # Dry-run should *not* crash on missing files — it surfaces them in the
    # plan with a clear marker instead.
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MISSING" in result.stdout.upper() or "missing" in result.stdout.lower()


def test_content_hash_helper_is_deterministic(tmp_path: Path) -> None:
    """The exposed content-hash helper produces a stable SHA-256 hex digest."""

    sys.path.insert(0, str(REPO_ROOT / "tools" / "release"))
    try:
        import hf_model_upload as mod
    finally:
        sys.path.pop(0)

    f = tmp_path / "f.bin"
    f.write_bytes(b"hello world")
    h1 = mod.file_sha256(f)
    h2 = mod.file_sha256(f)
    assert h1 == h2
    assert len(h1) == 64
    assert h1 == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_plan_collects_all_artifacts(tmp_path: Path) -> None:
    """``build_upload_plan`` returns the expected (source, target) pairs."""

    sys.path.insert(0, str(REPO_ROOT / "tools" / "release"))
    try:
        import hf_model_upload as mod
    finally:
        sys.path.pop(0)

    repo = _make_repo_root(tmp_path)
    ckpt = _make_checkpoint(repo)

    plan = mod.build_upload_plan(checkpoint=ckpt, repo_root=repo)
    targets = {entry.target for entry in plan}

    assert "encoder.safetensors" in targets
    assert "projector.safetensors" in targets
    assert "predictor.safetensors" in targets
    assert "pred_proj.safetensors" in targets
    assert "tokenizer/tokenizer.json" in targets
    assert "tokenizer/tokenizer.model" in targets
    assert "LICENSES/Apache-2.0.txt" in targets
    assert "README.md" in targets  # MODEL_CARD.md → README.md
