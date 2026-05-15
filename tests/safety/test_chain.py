"""Property tests for the safety filter chain (RFC-0007 §D6, issue #98)."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from codingjepa.safety.filter import run

# ---------------------------------------------------------------------------
# Property: parse failures must not crash the filter
# ---------------------------------------------------------------------------


@given(st.text(min_size=1, max_size=200))
@settings(max_examples=50)
def test_parse_failure_is_safe(random_text: str) -> None:
    """Unparseable / arbitrary text must not raise — result.passed must be a bool."""
    result = run(random_text, random_text)
    assert isinstance(result.passed, bool)


# ---------------------------------------------------------------------------
# Property: known async/sync mutations always fire R006
# ---------------------------------------------------------------------------


@given(
    st.sampled_from(
        [
            ("def f():\n    pass\n", "async def f():\n    pass\n"),
            ("async def f():\n    pass\n", "def f():\n    pass\n"),
        ]
    )
)
@settings(max_examples=50)
def test_async_sync_mutation_fires(pair: tuple[str, str]) -> None:
    before, after = pair
    result = run(before, after)
    assert not result.passed
    assert result.refusal_code == "R006_SAFETY_CHECKER_REJECTED_ALL"


# ---------------------------------------------------------------------------
# Property: known side-effect introductions always fire
# ---------------------------------------------------------------------------


@given(
    st.sampled_from(
        [
            ("def f():\n    pass\n", "def f():\n    print('x')\n"),
            ("def f():\n    x = 1\n", "def f():\n    print(x)\n"),
        ]
    )
)
@settings(max_examples=50)
def test_side_effect_intro_fires(pair: tuple[str, str]) -> None:
    before, after = pair
    result = run(before, after)
    assert not result.passed


# ---------------------------------------------------------------------------
# Property: identical source always passes (idempotency)
# ---------------------------------------------------------------------------


@given(
    st.sampled_from(
        [
            "def f():\n    return 1\n",
            "def f(x: int) -> int:\n    return x\n",
            "async def g():\n    pass\n",
            "x = 1\n",
            "pass\n",
        ]
    )
)
@settings(max_examples=50)
def test_identical_source_passes(source: str) -> None:
    """Running the same source as both before and after must always pass."""
    result = run(source, source)
    assert result.passed


# ---------------------------------------------------------------------------
# Property: passing results have empty reason and refusal_code
# ---------------------------------------------------------------------------


@given(
    st.sampled_from(
        [
            ("def f():\n    return 1\n", "def f():\n    result = 1\n    return result\n"),
            ("x = 1\n", "x = 1\ny = x\n"),
            (
                "def f(a: int, b: int) -> int:\n    return a + b\n",
                "def f(a: int, b: int) -> int:\n    s = a + b\n    return s\n",
            ),
        ]
    )
)
@settings(max_examples=50)
def test_passing_result_fields_are_empty(pair: tuple[str, str]) -> None:
    """When the filter passes, reason and refusal_code must be empty strings."""
    before, after = pair
    result = run(before, after)
    assert result.passed
    assert result.reason == ""
    assert result.refusal_code == ""


# ---------------------------------------------------------------------------
# Property: failing results always have non-empty refusal_code
# ---------------------------------------------------------------------------


@given(
    st.sampled_from(
        [
            # side effect intro
            ("def f():\n    return 1\n", "def f():\n    print('x')\n    return 1\n"),
            # exception added
            ("def f():\n    return 1\n", "def f():\n    raise ValueError\n"),
            # async boundary
            ("def f():\n    return 1\n", "async def f():\n    return 1\n"),
            # public API changed
            ("def f(x: int) -> int:\n    return x\n", "def g(x: int) -> int:\n    return x\n"),
        ]
    )
)
@settings(max_examples=50)
def test_failing_result_has_refusal_code(pair: tuple[str, str]) -> None:
    """Every failing result must carry a non-empty refusal_code."""
    before, after = pair
    result = run(before, after)
    assert not result.passed
    assert result.refusal_code != ""
