"""Tests for codingjepa.data.pairs — RFC-0002 §D3-D4 commit walker."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import libcst
import pytest

from codingjepa.data import pairs


def test_extract_top_level_nodes_simple() -> None:
    """Top-level FunctionDef, AsyncFunctionDef, ClassDef nodes are keyed by qualname."""

    src = (
        "def foo():\n"
        "    return 1\n"
        "\n"
        "async def bar():\n"
        "    return 2\n"
        "\n"
        "class Baz:\n"
        "    def method(self):\n"
        "        return 3\n"
        "\n"
        "x = 1\n"
    )
    tree = libcst.parse_module(src)
    nodes = pairs.extract_top_level_nodes(tree)

    assert set(nodes.keys()) == {"foo", "bar", "Baz"}
    # Methods inside the class are *not* their own top-level nodes (RFC-0012 §D3).
    assert "Baz.method" not in nodes


def test_extract_top_level_nodes_empty_module() -> None:
    """A module with no definitions yields an empty mapping."""

    tree = libcst.parse_module("x = 1\ny = 2\n")
    assert pairs.extract_top_level_nodes(tree) == {}


def test_extract_top_level_nodes_ignores_nested() -> None:
    """Nested defs are not surfaced as top-level (only their enclosing def is)."""

    src = "def outer():\n    def inner():\n        return 1\n    return inner\n"
    tree = libcst.parse_module(src)
    nodes = pairs.extract_top_level_nodes(tree)

    assert set(nodes.keys()) == {"outer"}


@dataclass
class _StubModifiedFile:
    """Stand-in for pydriller's ModifiedFile."""

    new_path: str | None
    old_path: str | None
    source_code_before: str | None
    source_code: str | None
    filename: str = ""

    def __post_init__(self) -> None:
        if not self.filename:
            self.filename = (self.new_path or self.old_path or "").split("/")[-1]


_PRE_CUTOFF_DATE = datetime(2023, 6, 1, tzinfo=UTC)


@dataclass
class _StubCommit:
    """Stand-in for pydriller's Commit."""

    hash: str
    msg: str
    author_name: str = "alice"
    author_email: str = "alice@example.com"
    merge: bool = False
    parents: list[str] | None = None
    modified_files: list[_StubModifiedFile] | None = None
    author_date: datetime = _PRE_CUTOFF_DATE

    def __post_init__(self) -> None:
        if self.parents is None:
            self.parents = ["00" * 20]
        if self.modified_files is None:
            self.modified_files = []


class _StubRepository:
    """Stand-in for pydriller.Repository — iterates over a fixed commit list."""

    last_kwargs: dict[str, Any] | None = None

    def __init__(self, path: str, **kwargs: Any) -> None:
        self.path = path
        _StubRepository.last_kwargs = dict(kwargs)

    def traverse_commits(self) -> list[_StubCommit]:
        return _STUB_COMMITS


_STUB_COMMITS: list[_StubCommit] = []


@pytest.fixture
def stub_pydriller(monkeypatch: pytest.MonkeyPatch) -> type[_StubRepository]:
    """Swap pydriller.Repository for an in-memory stub driven by _STUB_COMMITS."""

    _STUB_COMMITS.clear()
    _StubRepository.last_kwargs = None
    monkeypatch.setattr(pairs, "Repository", _StubRepository)
    return _StubRepository


def test_walk_repo_filters_merge_commits(
    tmp_path: Path, stub_pydriller: type[_StubRepository]
) -> None:
    """Merge commits never produce pairs (RFC-0002 §D4)."""

    _STUB_COMMITS.extend(
        [
            _StubCommit(
                hash="a" * 40,
                msg="merge branch x",
                merge=True,
                parents=["b" * 40, "c" * 40],
                modified_files=[
                    _StubModifiedFile(
                        new_path="m.py",
                        old_path="m.py",
                        source_code_before="def foo():\n    return 1\n",
                        source_code="def foo():\n    return 2\n",
                    )
                ],
            )
        ]
    )

    result = list(pairs.walk_repo(tmp_path, "owner/repo"))
    assert result == []


@pytest.mark.parametrize(
    "msg",
    [
        "wip: refactor",
        "WIP wiring",
        "temp commit",
        "Revert: previous change",
        "formatting only",
        "lint only",
        "Style only sweep",
        "typo fix",
    ],
)
def test_walk_repo_filters_wip_commits(
    tmp_path: Path,
    stub_pydriller: type[_StubRepository],
    msg: str,
) -> None:
    """Commit messages matching the drop-list are skipped (case-insensitive, RFC-0002 §D4)."""

    _STUB_COMMITS.append(
        _StubCommit(
            hash="d" * 40,
            msg=msg,
            modified_files=[
                _StubModifiedFile(
                    new_path="m.py",
                    old_path="m.py",
                    source_code_before="def foo():\n    return 1\n",
                    source_code="def foo():\n    return 2\n",
                )
            ],
        )
    )

    assert list(pairs.walk_repo(tmp_path, "owner/repo")) == []


def test_walk_repo_filters_bot_authors(
    tmp_path: Path, stub_pydriller: type[_StubRepository]
) -> None:
    """Bot-authored commits are dropped (dependabot, pre-commit-ci, etc.)."""

    _STUB_COMMITS.append(
        _StubCommit(
            hash="e" * 40,
            msg="chore: bump deps",
            author_name="dependabot[bot]",
            modified_files=[
                _StubModifiedFile(
                    new_path="m.py",
                    old_path="m.py",
                    source_code_before="def foo():\n    return 1\n",
                    source_code="def foo():\n    return 2\n",
                )
            ],
        )
    )

    assert list(pairs.walk_repo(tmp_path, "owner/repo")) == []


def test_walk_repo_skips_whitespace_only_diff(
    tmp_path: Path, stub_pydriller: type[_StubRepository]
) -> None:
    """A diff that normalizes to the same chunk text is dropped (RFC-0002 §D4)."""

    before = "def foo():\n    return 1\n"
    after = "def foo():\n    return 1\n\n\n"  # extra trailing whitespace only
    _STUB_COMMITS.append(
        _StubCommit(
            hash="f" * 40,
            msg="cleanup trailing newlines",
            modified_files=[
                _StubModifiedFile(
                    new_path="m.py",
                    old_path="m.py",
                    source_code_before=before,
                    source_code=after,
                )
            ],
        )
    )

    assert list(pairs.walk_repo(tmp_path, "owner/repo")) == []


def test_walk_repo_emits_pair_for_real_change(
    tmp_path: Path, stub_pydriller: type[_StubRepository]
) -> None:
    """A genuine body change in a top-level function emits one RawPair."""

    before = "def foo(x):\n    return x + 1\n"
    after = "def foo(x):\n    y = x + 1\n    return y\n"
    _STUB_COMMITS.append(
        _StubCommit(
            hash="9" * 40,
            msg="refactor: extract local for clarity",
            parents=["1" * 40],
            modified_files=[
                _StubModifiedFile(
                    new_path="m.py",
                    old_path="m.py",
                    source_code_before=before,
                    source_code=after,
                )
            ],
        )
    )

    out = list(pairs.walk_repo(tmp_path, "owner/repo"))
    assert len(out) == 1
    pair = out[0]
    assert isinstance(pair, pairs.RawPair)
    assert pair.repo == "owner/repo"
    assert pair.commit_sha_after == "9" * 40
    assert pair.commit_sha_before == "1" * 40
    assert pair.file_path == "m.py"
    assert pair.node_qualname == "foo"
    assert pair.commit_message == "refactor: extract local for clarity"
    assert "y = x + 1" in pair.source_after
    assert "y = x + 1" not in pair.source_before


def test_walk_repo_skips_renamed_nodes(
    tmp_path: Path, stub_pydriller: type[_StubRepository]
) -> None:
    """A node whose qualname does not appear in both before and after is dropped."""

    before = "def foo():\n    return 1\n"
    after = "def renamed_foo():\n    return 1\n"
    _STUB_COMMITS.append(
        _StubCommit(
            hash="2" * 40,
            msg="rename foo",
            modified_files=[
                _StubModifiedFile(
                    new_path="m.py",
                    old_path="m.py",
                    source_code_before=before,
                    source_code=after,
                )
            ],
        )
    )

    assert list(pairs.walk_repo(tmp_path, "owner/repo")) == []


def test_walk_repo_only_processes_python_files(
    tmp_path: Path, stub_pydriller: type[_StubRepository]
) -> None:
    """Non-.py modifications never yield pairs."""

    _STUB_COMMITS.append(
        _StubCommit(
            hash="3" * 40,
            msg="docs: update readme",
            modified_files=[
                _StubModifiedFile(
                    new_path="README.md",
                    old_path="README.md",
                    source_code_before="# old\n",
                    source_code="# new\n",
                )
            ],
        )
    )

    assert list(pairs.walk_repo(tmp_path, "owner/repo")) == []


def test_walk_repo_skips_pyi_and_pxd(tmp_path: Path, stub_pydriller: type[_StubRepository]) -> None:
    """Stub and Cython files are not Python sources for pair mining."""

    _STUB_COMMITS.append(
        _StubCommit(
            hash="4" * 40,
            msg="refactor: tweak stub",
            modified_files=[
                _StubModifiedFile(
                    new_path="m.pyi",
                    old_path="m.pyi",
                    source_code_before="def foo() -> None: ...\n",
                    source_code="def foo() -> int: ...\n",
                )
            ],
        )
    )

    assert list(pairs.walk_repo(tmp_path, "owner/repo")) == []


def test_walk_repo_respects_max_pairs(
    tmp_path: Path, stub_pydriller: type[_StubRepository]
) -> None:
    """The max_pairs cap stops iteration once reached."""

    for i in range(5):
        _STUB_COMMITS.append(
            _StubCommit(
                hash=f"{i:040x}",
                msg=f"refactor commit {i}",
                modified_files=[
                    _StubModifiedFile(
                        new_path="m.py",
                        old_path="m.py",
                        source_code_before=f"def foo():\n    return {i}\n",
                        source_code=f"def foo():\n    return {i + 100}\n",
                    )
                ],
            )
        )

    out = list(pairs.walk_repo(tmp_path, "owner/repo", max_pairs=3))
    assert len(out) == 3


def test_walk_repo_forwards_from_and_to_commit(
    tmp_path: Path, stub_pydriller: type[_StubRepository]
) -> None:
    """from_commit / to_commit kwargs are passed through to pydriller.Repository."""

    list(
        pairs.walk_repo(
            tmp_path,
            "owner/repo",
            from_commit="abc",
            to_commit="def",
        )
    )

    assert stub_pydriller.last_kwargs is not None
    assert stub_pydriller.last_kwargs.get("from_commit") == "abc"
    assert stub_pydriller.last_kwargs.get("to_commit") == "def"


def test_walk_repo_skips_unparseable_source(
    tmp_path: Path, stub_pydriller: type[_StubRepository]
) -> None:
    """If libcst cannot parse before or after, the file is skipped without raising."""

    _STUB_COMMITS.append(
        _StubCommit(
            hash="5" * 40,
            msg="refactor: syntax fix in progress",
            modified_files=[
                _StubModifiedFile(
                    new_path="m.py",
                    old_path="m.py",
                    source_code_before="def foo(:\n    return 1\n",  # invalid
                    source_code="def foo():\n    return 2\n",
                )
            ],
        )
    )

    # Must not raise; just yields nothing.
    assert list(pairs.walk_repo(tmp_path, "owner/repo")) == []


# ---------------------------------------------------------------------------
# Commit cutoff (#174) — RFC-0002 contamination control.
#
# Commits authored on or after ``COMMIT_CUTOFF`` (2024-01-01 UTC) are excluded
# from pair extraction so the corpus stays anchored to a pre-2024 snapshot.
# ---------------------------------------------------------------------------


def test_commit_cutoff_constant_is_2024_jan_1() -> None:
    """The exclusive upper bound is 2024-01-01 UTC (anything earlier is included)."""

    assert pairs.COMMIT_CUTOFF == datetime(2024, 1, 1, tzinfo=UTC)


def test_extract_pairs_skips_after_cutoff(
    tmp_path: Path, stub_pydriller: type[_StubRepository]
) -> None:
    """A commit dated on or after the cutoff yields no pairs."""

    _STUB_COMMITS.append(
        _StubCommit(
            hash="6" * 40,
            msg="refactor: after the cutoff",
            author_date=datetime(2024, 1, 2, tzinfo=UTC),
            modified_files=[
                _StubModifiedFile(
                    new_path="m.py",
                    old_path="m.py",
                    source_code_before="def foo():\n    return 1\n",
                    source_code="def foo():\n    return 2\n",
                )
            ],
        )
    )

    assert list(pairs.walk_repo(tmp_path, "owner/repo")) == []


def test_extract_pairs_skips_commit_at_cutoff_boundary(
    tmp_path: Path, stub_pydriller: type[_StubRepository]
) -> None:
    """A commit dated *exactly* at the cutoff is excluded (cutoff is exclusive)."""

    _STUB_COMMITS.append(
        _StubCommit(
            hash="7" * 40,
            msg="refactor: on the boundary",
            author_date=datetime(2024, 1, 1, tzinfo=UTC),
            modified_files=[
                _StubModifiedFile(
                    new_path="m.py",
                    old_path="m.py",
                    source_code_before="def foo():\n    return 1\n",
                    source_code="def foo():\n    return 2\n",
                )
            ],
        )
    )

    assert list(pairs.walk_repo(tmp_path, "owner/repo")) == []


def test_extract_pairs_keeps_pre_cutoff_commits(
    tmp_path: Path, stub_pydriller: type[_StubRepository]
) -> None:
    """A commit dated before the cutoff is preserved (sanity check)."""

    _STUB_COMMITS.append(
        _StubCommit(
            hash="8" * 40,
            msg="refactor: well before the cutoff",
            author_date=datetime(2023, 12, 31, 23, 59, 59, tzinfo=UTC),
            modified_files=[
                _StubModifiedFile(
                    new_path="m.py",
                    old_path="m.py",
                    source_code_before="def foo():\n    return 1\n",
                    source_code="def foo():\n    return 2\n",
                )
            ],
        )
    )

    assert len(list(pairs.walk_repo(tmp_path, "owner/repo"))) == 1


def test_extract_pairs_treats_naive_datetime_as_utc(
    tmp_path: Path, stub_pydriller: type[_StubRepository]
) -> None:
    """A naive (tz-less) author_date is interpreted as UTC, then compared to cutoff."""

    _STUB_COMMITS.append(
        _StubCommit(
            hash="9" * 40,
            msg="refactor: naive datetime after cutoff",
            # tz-less but after 2024-01-01 — must still be filtered out.
            author_date=datetime(2024, 6, 15, 12, 0, 0),
            modified_files=[
                _StubModifiedFile(
                    new_path="m.py",
                    old_path="m.py",
                    source_code_before="def foo():\n    return 1\n",
                    source_code="def foo():\n    return 2\n",
                )
            ],
        )
    )

    assert list(pairs.walk_repo(tmp_path, "owner/repo")) == []


def test_walk_repo_respects_custom_cutoff_kwarg(
    tmp_path: Path, stub_pydriller: type[_StubRepository]
) -> None:
    """Callers may override the module cutoff via the ``cutoff`` kwarg."""

    _STUB_COMMITS.append(
        _StubCommit(
            hash="a" * 40,
            msg="refactor: dated 2022 but caller cutoff is 2021",
            author_date=datetime(2022, 5, 1, tzinfo=UTC),
            modified_files=[
                _StubModifiedFile(
                    new_path="m.py",
                    old_path="m.py",
                    source_code_before="def foo():\n    return 1\n",
                    source_code="def foo():\n    return 2\n",
                )
            ],
        )
    )

    custom_cutoff = datetime(2021, 1, 1, tzinfo=UTC)
    assert list(pairs.walk_repo(tmp_path, "owner/repo", cutoff=custom_cutoff)) == []
