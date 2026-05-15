"""CI gate: acceptance rules live in codingjepa.intents.acceptance only.

Spec/01 §Cross-cutting contracts: `codingjepa.intents.acceptance` is the
single source of truth for the per-intent acceptance rules. If any other
module imports the private rule helpers, fail.

The public surface goes through `from codingjepa.intents import
acceptance_check` (re-exported by `codingjepa.intents.__init__`).
"""

from __future__ import annotations

import ast
import pathlib

PKG_ROOT = pathlib.Path(__file__).resolve().parents[1] / "codingjepa"
ALLOWED_FILES = {
    PKG_ROOT / "intents" / "__init__.py",
    PKG_ROOT / "intents" / "acceptance.py",
}


def _imports_from(file_path: pathlib.Path) -> list[ast.ImportFrom]:
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    return [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]


def test_no_direct_acceptance_import_outside_intents_module() -> None:
    """`from codingjepa.intents.acceptance import …` is only allowed in the SSOT package itself."""

    offenders: list[tuple[str, int]] = []
    for path in PKG_ROOT.rglob("*.py"):
        if path in ALLOWED_FILES:
            continue
        for node in _imports_from(path):
            if node.module == "codingjepa.intents.acceptance":
                offenders.append((str(path.relative_to(PKG_ROOT.parent)), node.lineno))
    assert not offenders, f"direct codingjepa.intents.acceptance imports outside SSOT: {offenders}"


def test_acceptance_rules_are_module_private() -> None:
    """The 8 `_check_*` rule functions are private (leading underscore) so they
    cannot be imported as part of the public surface."""

    from codingjepa.intents import acceptance

    private_rules = [
        name
        for name in dir(acceptance)
        if name.startswith("_check_")
        and callable(getattr(acceptance, name))
        and name != "_check_type"  # belongs to a different module if present
    ]
    assert len(private_rules) == 8, f"expected 8 _check_<intent> rules, found {private_rules}"


def test_public_re_export_resolves() -> None:
    """The public surface `from codingjepa.intents import acceptance_check` is reachable."""

    from codingjepa.intents import acceptance_check

    assert callable(acceptance_check)
