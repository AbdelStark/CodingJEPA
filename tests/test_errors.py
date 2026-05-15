"""Closure and shape checks for codingjepa.errors. See docs/spec/04-error-model.md."""

from __future__ import annotations

import pytest

from codingjepa import errors


def test_taxonomy_is_closed() -> None:
    """Every CodingJEPAError subclass reachable at runtime is in CLOSED_TAXONOMY.

    Adding a new exception class without registering it in CLOSED_TAXONOMY is an
    RFC-amendment-only change per spec/04 §Error taxonomy; this test enforces it.
    """

    declared = set(errors.CLOSED_TAXONOMY)
    reachable = errors._all_subclasses(errors.CodingJEPAError) | {errors.CodingJEPAError}
    extra = reachable - declared
    extra_names = sorted(c.__name__ for c in extra)
    assert not extra, f"unregistered CodingJEPAError subclass(es): {extra_names}"


@pytest.mark.parametrize("cls", errors.CLOSED_TAXONOMY)
def test_each_class_exposes_code_and_context(cls: type[errors.CodingJEPAError]) -> None:
    """Every class in the taxonomy carries a stable `E_*` code and a context dict."""

    assert errors._CODE_PATTERN.match(cls.code), f"{cls.__name__}.code={cls.code!r} not E_*"

    instance = cls("boom", path="/tmp/x", hash="abc123")
    assert instance.code == cls.code
    assert instance.context == {"path": "/tmp/x", "hash": "abc123"}
    assert instance.message == "boom"
    assert str(instance) == "boom"


@pytest.mark.parametrize("cls", errors.CLOSED_TAXONOMY)
def test_repr_preserves_context(cls: type[errors.CodingJEPAError]) -> None:
    """repr(cls(...)) is stable and round-trips the context dict."""

    instance = cls("a message", request_id="uuid7", batch_idx=42)
    r = repr(instance)
    assert cls.__name__ in r
    assert "'a message'" in r
    assert "'request_id': 'uuid7'" in r
    assert "'batch_idx': 42" in r


def test_codes_are_unique() -> None:
    """No two classes share an `E_*` code; codes are correlation keys for log records."""

    codes = [cls.code for cls in errors.CLOSED_TAXONOMY]
    assert len(codes) == len(set(codes)), f"duplicate codes: {codes}"


def test_hierarchy_is_intact() -> None:
    """A handful of inheritance invariants from spec/04 §Error taxonomy."""

    assert issubclass(errors.ManifestHashMismatch, errors.ArtifactError)
    assert issubclass(errors.CheckpointHashMismatch, errors.ArtifactError)
    assert issubclass(errors.IndexHashMismatch, errors.ArtifactError)
    assert issubclass(errors.SchemaVersionMismatch, errors.DataError)
    assert issubclass(errors.EmbeddingCollapse, errors.ModelError)
    assert issubclass(errors.SandboxTimeout, errors.SandboxError)
    assert issubclass(errors.SandboxNetworkAttempted, errors.SandboxError)
    assert issubclass(errors.ArtifactError, errors.CodingJEPAError)


def test_caught_as_codingjepaerror() -> None:
    """Every class is catchable as the root CodingJEPAError."""

    for cls in errors.CLOSED_TAXONOMY:
        try:
            raise cls("test")
        except errors.CodingJEPAError as e:
            assert isinstance(e, cls)
