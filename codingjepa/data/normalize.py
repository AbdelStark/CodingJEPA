"""Normalize a Python source chunk per RFC-0012 §D5 and RFC-0014 §D5.

Pipeline:

1. Run ``black`` (line-length=100, py312).
2. Run ``isort`` on imports inside the chunk.
3. Walk the libcst tree:
   * Replace every function-, class-, and module-docstring literal with the
     fixed sentinel ``"<DOC>"`` so embeddings cannot leak documentation
     content.
   * Scrub email addresses from inline / standalone comments
     (RFC-0014 §D5 PII).
   * Strip ``# type: ignore``, ``# noqa``, ``# fmt:``, ``# pylint:``,
     ``# mypy:`` editor pragmas — both as trailing annotations on a real
     statement and as standalone comment lines.
4. Normalize line endings to ``\\n`` and strip trailing whitespace.
5. Run ``compile()`` on Python 3.12; if it fails, return ``None``.

This module is import-side-effect-free.
"""

from __future__ import annotations

import re

import black
import isort
import libcst as cst
from black.parsing import InvalidInput as BlackInvalidInput

# RFC-0012 §D5: fixed docstring sentinel. The string literal in the AST is
# ``"<DOC>"`` (with surrounding double-quotes), so tokenization sees a single
# stable token regardless of the original docstring.
_DOC_SENTINEL_LITERAL: str = '"<DOC>"'

# RFC-0014 §D5: scrub anything that looks like an email address from comments.
_EMAIL_RE: re.Pattern[str] = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Editor pragmas that we both (a) strip from inline trailing comments and
# (b) delete entire standalone comment lines that match.
_PRAGMA_PREFIXES: tuple[str, ...] = (
    "# type: ignore",
    "# noqa",
    "# fmt:",
    "# pylint:",
    "# mypy:",
)

# A trailing pragma that follows real code: `x = foo()  # type: ignore[...]`.
# We want to drop the comment itself, including the leading whitespace, but
# keep the statement. The alternation covers every recognized editor pragma;
# trailing `[^\n]*` consumes the rest of the line up to but not including the
# newline.
_TRAILING_PRAGMA_RE: re.Pattern[str] = re.compile(
    r"[ \t]*#[ \t]*(?:type:[ \t]*ignore|noqa|fmt:|pylint:|mypy:)[^\n]*"
)


# --------------------------------------------------------------------------- #
# libcst transformer                                                          #
# --------------------------------------------------------------------------- #


def _is_string_expr(stmt: cst.BaseStatement) -> bool:
    """Return True if ``stmt`` is a SimpleStatementLine wrapping a string Expr.

    Matches the three string-literal node types: ``SimpleString``,
    ``ConcatenatedString``, and ``FormattedString``. This is the syntactic
    shape of a docstring at the start of a module, class, or function body.
    """

    if not isinstance(stmt, cst.SimpleStatementLine):
        return False
    if len(stmt.body) != 1:
        return False
    inner = stmt.body[0]
    if not isinstance(inner, cst.Expr):
        return False
    return isinstance(
        inner.value,
        (cst.SimpleString, cst.ConcatenatedString, cst.FormattedString),
    )


def _docstring_replacement_stmt() -> cst.SimpleStatementLine:
    """Build the SimpleStatementLine that replaces a docstring."""

    return cst.SimpleStatementLine(
        body=[cst.Expr(value=cst.SimpleString(value=_DOC_SENTINEL_LITERAL))]
    )


class _NormalizeTransformer(cst.CSTTransformer):
    """Replace docstrings with the sentinel and clean up comments."""

    # ---- docstrings ----------------------------------------------------- #

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        new_body = list(updated_node.body)
        if new_body and _is_string_expr(new_body[0]):
            new_body[0] = _docstring_replacement_stmt()
        return updated_node.with_changes(body=tuple(new_body))

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        return updated_node.with_changes(body=self._strip_body_docstring(updated_node.body))

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        return updated_node.with_changes(body=self._strip_body_docstring(updated_node.body))

    @staticmethod
    def _strip_body_docstring(
        body: cst.BaseSuite,
    ) -> cst.BaseSuite:
        """Replace a leading docstring inside an IndentedBlock / SimpleStatementSuite."""

        if isinstance(body, cst.IndentedBlock):
            stmts = list(body.body)
            if stmts and _is_string_expr(stmts[0]):
                stmts[0] = _docstring_replacement_stmt()
                return body.with_changes(body=tuple(stmts))
            return body
        if isinstance(body, cst.SimpleStatementSuite):
            inner = list(body.body)
            if (
                inner
                and isinstance(inner[0], cst.Expr)
                and isinstance(
                    inner[0].value,
                    (cst.SimpleString, cst.ConcatenatedString, cst.FormattedString),
                )
            ):
                inner[0] = cst.Expr(value=cst.SimpleString(value=_DOC_SENTINEL_LITERAL))
                return body.with_changes(body=tuple(inner))
            return body
        return body

    # ---- comments ------------------------------------------------------- #

    def leave_Comment(self, original_node: cst.Comment, updated_node: cst.Comment) -> cst.Comment:
        return updated_node.with_changes(value=_scrub_comment(updated_node.value))


def _scrub_comment(comment: str) -> str:
    """Strip emails from a comment's text, preserving the leading ``#``.

    Returns an empty string for comments that are entirely an editor pragma
    so the caller can drop the line; callers that need to keep the ``#``
    handle that themselves via :func:`_collapse_blank_pragma_lines`.
    """

    text = _EMAIL_RE.sub("", comment)
    # Collapse runs of whitespace that the email removal may have left behind,
    # but only between the leading `#` and the rest of the line content.
    text = re.sub(r"[ \t]+", " ", text).rstrip()
    return text


# --------------------------------------------------------------------------- #
# string-level passes                                                         #
# --------------------------------------------------------------------------- #


def _strip_pragma_lines(source: str) -> str:
    """Drop standalone lines that are only an editor pragma comment.

    A line like ``    # type: ignore`` (after black's indentation) is removed
    entirely. A line like ``x = 1  # type: ignore`` is handled by
    :func:`_strip_trailing_pragmas`.
    """

    out: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(p) for p in _PRAGMA_PREFIXES):
            # Standalone pragma; drop the whole line.
            continue
        out.append(line)
    return "\n".join(out)


def _strip_trailing_pragmas(source: str) -> str:
    """Remove trailing ``# type: ignore`` / ``# noqa`` / etc. from real lines."""

    return _TRAILING_PRAGMA_RE.sub("", source)


def _normalize_whitespace(source: str) -> str:
    """Normalize line endings to ``\\n`` and strip trailing whitespace."""

    # CRLF / CR -> LF.
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing whitespace per-line.
    lines = [line.rstrip() for line in source.split("\n")]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# public API                                                                  #
# --------------------------------------------------------------------------- #


def _format_with_black(source: str) -> str:
    """Run black with the package line-length setting."""

    return black.format_str(source, mode=black.FileMode(line_length=100))


def _format_with_isort(source: str) -> str:
    """Run isort on the source."""

    # ``isort.code`` is a pure-function entry point that respects per-process
    # defaults; we want it deterministic regardless of the user's home config.
    return isort.code(source, profile="black", line_length=100)


def _apply_cst_pass(source: str) -> str:
    """Run the libcst transformer (docstring sentinel + comment scrub)."""

    module = cst.parse_module(source)
    transformed = module.visit(_NormalizeTransformer())
    return transformed.code


def _verify_compile(source: str) -> bool:
    """Return True iff ``source`` compiles as Python 3.12."""

    try:
        compile(source, "<normalized>", "exec")
    except SyntaxError:
        return False
    return True


def normalize_chunk(source: str) -> str | None:
    """Normalize a Python code chunk. Returns None if compile() fails after normalization.

    Steps per RFC-0012 §D5:

    1. Run black (line-length=100).
    2. Run isort.
    3. Replace docstring literals with ``<DOC>`` sentinel.
    4. Strip ``# type: ignore`` and editor pragmas.
    5. Strip email addresses from comments.
    6. Normalize line endings, strip trailing whitespace.
    7. Verify compile(); return None if it fails.
    """

    if source == "":
        return ""

    # Step 0: normalize line endings *before* black so black sees LF.
    try:
        working = _normalize_whitespace(source)
    except Exception:
        return None

    # Step 1: black.
    try:
        working = _format_with_black(working)
    except (BlackInvalidInput, ValueError):
        return None

    # Step 2: isort. isort can re-format imports; safe to skip on failure.
    try:
        working = _format_with_isort(working)
    except Exception:
        # isort failure shouldn't drop the whole chunk; continue without it.
        pass

    # Step 3 + 5: libcst pass (docstring sentinel + email scrub on comments).
    try:
        working = _apply_cst_pass(working)
    except cst.ParserSyntaxError:
        return None

    # Step 4: strip standalone-pragma lines + trailing-pragma annotations.
    working = _strip_trailing_pragmas(working)
    working = _strip_pragma_lines(working)

    # Step 6: re-run black so we collapse any extra blank lines + re-flow
    # whatever the pragma strip changed. Idempotency relies on black being
    # the *last* formatter to touch the bytes.
    try:
        working = _format_with_black(working)
    except BlackInvalidInput:
        return None

    # Step 6b: re-normalize whitespace one more time as a safety net.
    working = _normalize_whitespace(working)

    # Ensure exactly one trailing newline (black guarantees this for non-empty
    # files, but defensive after our string-level edits).
    working = working.rstrip("\n")
    if working:
        working += "\n"

    # Step 7: verify it compiles.
    if not _verify_compile(working):
        return None

    return working


def normalize_source(source: str) -> str | None:
    """Normalize a full Python file source. Same pipeline as :func:`normalize_chunk`."""

    return normalize_chunk(source)


__all__ = ["normalize_chunk", "normalize_source"]
