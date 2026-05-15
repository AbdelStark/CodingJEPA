"""Tests for codingjepa.data.normalize. See RFC-0012 §D5 and RFC-0014 §D5."""

from __future__ import annotations

from codingjepa.data.normalize import normalize_chunk, normalize_source


def test_normalize_black_formatting() -> None:
    """black is applied: line-length=100, consistent spacing/quotes."""

    src = "def f(x,y ,z):\n  return x+y+z\n"
    out = normalize_chunk(src)
    assert out is not None
    # black uses double-quotes, normalizes spaces around commas.
    assert "def f(x, y, z):" in out
    # black uses 4-space indent.
    assert "    return x + y + z" in out


def test_normalize_docstring_sentinel_function() -> None:
    """A function docstring is replaced with the <DOC> sentinel."""

    src = '''
def f(x):
    """This is a long docstring
    spanning multiple lines.
    """
    return x
'''
    out = normalize_chunk(src)
    assert out is not None
    assert "<DOC>" in out
    assert "spanning multiple lines" not in out
    assert "This is a long docstring" not in out


def test_normalize_docstring_sentinel_class() -> None:
    """A class docstring is replaced with the <DOC> sentinel."""

    src = '''
class C:
    """Class doc here."""

    def m(self):
        """Method doc here."""
        return 1
'''
    out = normalize_chunk(src)
    assert out is not None
    assert "<DOC>" in out
    assert "Class doc here" not in out
    assert "Method doc here" not in out


def test_normalize_module_docstring_sentinel() -> None:
    """Module-level docstring is replaced too."""

    src = '''"""Module docstring."""

x = 1
'''
    out = normalize_source(src)
    assert out is not None
    assert "<DOC>" in out
    assert "Module docstring" not in out


def test_normalize_type_ignore_stripped() -> None:
    """`# type: ignore` annotations are stripped."""

    src = "x = foo()  # type: ignore[no-untyped-call]\n"
    out = normalize_chunk(src)
    assert out is not None
    assert "type: ignore" not in out
    assert "x = foo()" in out


def test_normalize_type_ignore_line_only_removed() -> None:
    """A line that is only `# type: ignore` becomes blank/removed."""

    src = "def f(x):\n    # type: ignore\n    return x\n"
    out = normalize_chunk(src)
    assert out is not None
    assert "type: ignore" not in out


def test_normalize_noqa_stripped() -> None:
    """`# noqa` editor pragmas are stripped."""

    src = "import os  # noqa: F401\n"
    out = normalize_chunk(src)
    assert out is not None
    assert "noqa" not in out


def test_normalize_fmt_off_stripped() -> None:
    """`# fmt: off` editor pragmas are stripped."""

    src = "x = 1  # fmt: off\ny = 2\n"
    out = normalize_chunk(src)
    assert out is not None
    assert "fmt:" not in out


def test_normalize_pylint_stripped() -> None:
    """`# pylint:` and `# mypy:` pragmas are stripped."""

    src = "x = 1  # pylint: disable=invalid-name\n"
    out = normalize_chunk(src)
    assert out is not None
    assert "pylint" not in out


def test_normalize_compile_fail_returns_none() -> None:
    """Invalid Python that won't parse returns None."""

    src = "def broken(x:\n    return x\n"
    out = normalize_chunk(src)
    assert out is None


def test_normalize_email_stripped() -> None:
    """Email addresses in comments are scrubbed."""

    src = "x = 1  # author: alice@example.com\n"
    out = normalize_chunk(src)
    assert out is not None
    assert "alice@example.com" not in out
    assert "example.com" not in out


def test_normalize_email_stripped_module_comment() -> None:
    """Email scrub also applies in module-level comments."""

    src = "# Contact: bob.smith+filter@sub.domain.co.uk for help\nx = 1\n"
    out = normalize_source(src)
    assert out is not None
    assert "@sub.domain" not in out
    assert "bob.smith" not in out


def test_normalize_trailing_whitespace_stripped() -> None:
    """Trailing whitespace is removed."""

    src = "x = 1   \ny = 2\t\n"
    out = normalize_chunk(src)
    assert out is not None
    for line in out.splitlines():
        assert line == line.rstrip()


def test_normalize_crlf_to_lf() -> None:
    """CRLF line endings are normalized to LF."""

    src = "x = 1\r\ny = 2\r\n"
    out = normalize_chunk(src)
    assert out is not None
    assert "\r" not in out


def test_normalize_idempotent() -> None:
    """normalize(normalize(x)) == normalize(x)."""

    src = '''
def foo(x ,y):
    """Docstring."""
    z = x+y  # type: ignore
    return z  # author: a@b.com
'''
    once = normalize_chunk(src)
    assert once is not None
    twice = normalize_chunk(once)
    assert twice == once


def test_normalize_isort_applied() -> None:
    """isort reorders imports inside the chunk."""

    src = "import os\nimport sys\nimport collections\n\nx = 1\n"
    out = normalize_source(src)
    assert out is not None
    # isort sorts stdlib alphabetically.
    lines = [line for line in out.splitlines() if line.startswith("import ")]
    assert lines == sorted(lines)


def test_normalize_empty_source() -> None:
    """Empty source returns empty (or near-empty) string, not None."""

    out = normalize_chunk("")
    assert out is not None
    assert out.strip() == ""


def test_normalize_returns_lf_terminated() -> None:
    """Output uses `\\n` line endings exclusively."""

    src = "x = 1\n"
    out = normalize_chunk(src)
    assert out is not None
    assert "\r" not in out


def test_normalize_source_same_as_chunk_for_top_level() -> None:
    """normalize_source and normalize_chunk produce the same result for clean code."""

    src = "x = 1\n"
    a = normalize_chunk(src)
    b = normalize_source(src)
    assert a == b
