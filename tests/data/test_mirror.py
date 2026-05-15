"""Tests for codingjepa.data.mirror — RFC-0002 §D1 and docs/data/CANDIDATE_REPOS.md."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from codingjepa.data import mirror


def test_repo_registry_has_10_repos() -> None:
    """The locked v1 corpus is exactly 10 repositories (RFC-0002 §D1)."""

    assert len(mirror.REPO_REGISTRY) == 10


def test_repo_registry_has_required_metadata() -> None:
    """Each entry carries name, url, commit_sha, license_spdx, subset_paths, split."""

    required_keys = {"name", "url", "commit_sha", "license_spdx", "subset_paths", "split"}
    for entry in mirror.REPO_REGISTRY:
        missing = required_keys - set(entry.keys())
        assert not missing, f"{entry.get('name', '?')} missing keys: {missing}"
        assert entry["split"] in {"train", "val", "test", "ood"}, entry["split"]
        # 40-char hex commit sha
        assert len(entry["commit_sha"]) == 40, entry["commit_sha"]
        assert all(c in "0123456789abcdef" for c in entry["commit_sha"]), entry["commit_sha"]


def test_repo_registry_includes_cpython_with_lib_only() -> None:
    """cpython is restricted to Lib/ per docs/data/CANDIDATE_REPOS.md."""

    by_name = {r["name"]: r for r in mirror.REPO_REGISTRY}
    assert "python/cpython" in by_name
    assert by_name["python/cpython"]["subset_paths"] == ["Lib/"]


def test_repo_registry_split_distribution() -> None:
    """6 train, 2 val, 2 test — RFC-0002 §D1 / docs/data/CANDIDATE_REPOS.md §Splits."""

    splits = [r["split"] for r in mirror.REPO_REGISTRY]
    assert splits.count("train") == 6
    assert splits.count("val") == 2
    assert splits.count("test") == 2


def _make_fake_repo(tmp_path: Path) -> Path:
    """Build a fake repo tree exercising every path filter."""

    repo = tmp_path / "fake-repo"
    repo.mkdir()
    (repo / ".git").mkdir()  # marker so the dir looks like a repo

    # In-scope files.
    (repo / "pkg").mkdir()
    (repo / "pkg" / "core.py").write_text("def f(): pass\n")
    (repo / "pkg" / "util.py").write_text("def g(): pass\n")
    (repo / "top.py").write_text("x = 1\n")

    # Out-of-scope: vendored.
    (repo / "vendor").mkdir()
    (repo / "vendor" / "lib.py").write_text("# vendored\n")
    (repo / "_vendor").mkdir()
    (repo / "_vendor" / "lib.py").write_text("# vendored\n")
    (repo / "pkg" / "third_party").mkdir()
    (repo / "pkg" / "third_party" / "x.py").write_text("# vendored\n")

    # Out-of-scope: generated.
    (repo / "pkg" / "service_pb2.py").write_text("# generated\n")
    (repo / "pkg" / "auto.py").write_text("# DO NOT EDIT\n\ndef h(): pass\n")

    # Out-of-scope: not .py.
    (repo / "pkg" / "core.pyi").write_text("def f() -> None: ...\n")
    (repo / "README.md").write_text("# readme\n")

    return repo


def test_list_py_files_excludes_vendor(tmp_path: Path) -> None:
    """list_py_files filters vendor/, third_party/, _vendor/, *_pb2.py, DO NOT EDIT, non-py."""

    repo = _make_fake_repo(tmp_path)
    config = {"name": "owner/repo", "subset_paths": []}

    files = mirror.list_py_files(repo, config)
    relpaths = sorted(p.relative_to(repo).as_posix() for p in files)

    assert relpaths == ["pkg/core.py", "pkg/util.py", "top.py"]


def test_list_py_files_honors_subset_paths(tmp_path: Path) -> None:
    """When subset_paths is set (e.g. cpython Lib/), only those subtrees are listed."""

    repo = tmp_path / "fake-cpython"
    repo.mkdir()
    (repo / "Lib").mkdir()
    (repo / "Lib" / "os.py").write_text("def x(): pass\n")
    (repo / "Tools").mkdir()
    (repo / "Tools" / "build.py").write_text("def y(): pass\n")
    (repo / "Modules").mkdir()
    (repo / "Modules" / "_io.py").write_text("def z(): pass\n")

    config = {"name": "python/cpython", "subset_paths": ["Lib/"]}
    files = mirror.list_py_files(repo, config)
    relpaths = sorted(p.relative_to(repo).as_posix() for p in files)

    assert relpaths == ["Lib/os.py"]


class _FakeRun:
    """Track subprocess.run calls so tests can assert no real git was invoked.

    Maintains a small per-dest SHA table so that ``rev-parse HEAD`` after a
    clone reports the SHA the clone was asked to materialize.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self._head_by_dest: dict[str, str] = {}

    def __call__(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(tuple(args))
        # `git clone --filter=blob:none --depth=1 <url> <dest>`
        if args[:2] == ["git", "clone"]:
            dest = Path(args[-1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / ".git").mkdir(exist_ok=True)
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        # `git -C <dest> checkout <sha>` — record HEAD for later rev-parse.
        if len(args) >= 5 and args[0] == "git" and args[1] == "-C" and args[3] == "checkout":
            self._head_by_dest[args[2]] = args[4]
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        # `git -C <dest> rev-parse HEAD` — return whatever we last checked out.
        if (
            len(args) >= 5
            and args[0] == "git"
            and args[1] == "-C"
            and args[3] == "rev-parse"
            and args[4] == "HEAD"
        ):
            head = self._head_by_dest.get(args[2], "")
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout=head + "\n", stderr=""
            )
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")


def test_mirror_creates_manifest_if_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First-time mirror writes a fresh data/manifest.lock.json from REPO_REGISTRY."""

    output_dir = tmp_path / "repos"
    manifest_path = tmp_path / "manifest.lock.json"
    fake = _FakeRun()
    monkeypatch.setattr(mirror, "_run", fake)

    repo_dirs = mirror.mirror(output_dir=output_dir, manifest_path=manifest_path)

    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text())
    assert payload["schema_version"] == "v1"
    assert payload["chunker_version"] == "v1"
    assert payload["secrets_scanner_version"] == "v1"
    assert payload["splits_path"] == "data/splits/v1.lock.json"
    assert len(payload["repos"]) == 10
    assert len(payload["manifest_hash"]) == 64

    # Each repo dir was created on disk under owner/name/.
    assert len(repo_dirs) == 10
    for entry, repo_dir in zip(payload["repos"], repo_dirs, strict=True):
        owner, name = entry["name"].split("/")
        assert repo_dir == output_dir / owner / name
        assert repo_dir.exists()


def test_mirror_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling mirror twice does not re-clone repos that already exist at the pinned SHA."""

    output_dir = tmp_path / "repos"
    manifest_path = tmp_path / "manifest.lock.json"
    fake = _FakeRun()
    monkeypatch.setattr(mirror, "_run", fake)

    mirror.mirror(output_dir=output_dir, manifest_path=manifest_path)
    clone_calls_first = sum(1 for call in fake.calls if call[:2] == ("git", "clone"))
    assert clone_calls_first == 10

    # Second pass: no new clones, no new manifest content.
    before = manifest_path.read_text()
    mirror.mirror(output_dir=output_dir, manifest_path=manifest_path)
    clone_calls_second = sum(1 for call in fake.calls if call[:2] == ("git", "clone"))
    assert clone_calls_second == clone_calls_first, "mirror re-cloned an existing repo"
    assert manifest_path.read_text() == before, "mirror rewrote stable manifest"


def test_mirror_preserves_existing_manifest_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing manifest with downstream counts (chunks_emitted, etc.) is preserved on re-run."""

    output_dir = tmp_path / "repos"
    manifest_path = tmp_path / "manifest.lock.json"
    fake = _FakeRun()
    monkeypatch.setattr(mirror, "_run", fake)

    mirror.mirror(output_dir=output_dir, manifest_path=manifest_path)
    payload = json.loads(manifest_path.read_text())
    payload["repos"][0]["chunks_emitted"] = 42
    payload["repos"][0]["py_files_in_scope"] = 17
    manifest_path.write_text(json.dumps(payload))

    mirror.mirror(output_dir=output_dir, manifest_path=manifest_path)
    refreshed = json.loads(manifest_path.read_text())
    assert refreshed["repos"][0]["chunks_emitted"] == 42
    assert refreshed["repos"][0]["py_files_in_scope"] == 17
