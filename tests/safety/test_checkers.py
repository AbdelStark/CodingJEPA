"""Unit tests for individual safety checkers (RFC-0007 §D1)."""

from __future__ import annotations

import pytest

from codingjepa.safety.checkers.async_sync_boundary import check as check_async_sync
from codingjepa.safety.checkers.exception_contract_change import check as check_exception_contract
from codingjepa.safety.checkers.public_api_change import check as check_public_api
from codingjepa.safety.checkers.side_effect_elimination import check as check_side_effect_elim
from codingjepa.safety.checkers.side_effect_introduction import check as check_side_effect_intro
from codingjepa.safety.filter import SafetyResult, run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIMPLE_BEFORE = "def f(x: int) -> int:\n    return x + 1\n"
_SIMPLE_AFTER = "def f(x: int) -> int:\n    result = x + 1\n    return result\n"


# ---------------------------------------------------------------------------
# side_effect_introduction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "before, after",
    [
        # 1. Pure computation refactor — no new IO added
        (
            "def f(x: int) -> int:\n    return x + 1\n",
            "def f(x: int) -> int:\n    result = x + 1\n    return result\n",
        ),
        # 2. Both already have print — no *new* side effect
        (
            "def f():\n    print('a')\n",
            "def f():\n    print('b')\n",
        ),
        # 3. Both already use logging
        (
            "def f():\n    logging.info('start')\n",
            "def f():\n    logging.info('end')\n",
        ),
        # 4. Neither has side effects
        (
            "x = 1\ny = x + 2\n",
            "x = 1\ny = x + 2\nz = y\n",
        ),
        # 5. Empty sources — parse succeeds, no effects
        (
            "pass\n",
            "pass\n",
        ),
    ],
)
def test_side_effect_intro_passes(before: str, after: str) -> None:
    result = check_side_effect_intro(before, after)
    assert result.passed, f"Expected safe but got: {result.reason}"


@pytest.mark.parametrize(
    "before, after",
    [
        # 1. Adds print
        (
            "def f():\n    return 1\n",
            "def f():\n    print('debug')\n    return 1\n",
        ),
        # 2. Adds logging call
        (
            "def f():\n    return 1\n",
            "def f():\n    logger.info('x')\n    return 1\n",
        ),
        # 3. Adds file write via open
        (
            "def f():\n    return 1\n",
            "def f():\n    open('out.txt', 'w').write('x')\n    return 1\n",
        ),
        # 4. Adds os.system call
        (
            "def f():\n    return 1\n",
            "def f():\n    os.system('ls')\n    return 1\n",
        ),
        # 5. Adds requests.get
        (
            "def f(url: str) -> str:\n    return url\n",
            "def f(url: str) -> str:\n    requests.get(url)\n    return url\n",
        ),
    ],
)
def test_side_effect_intro_fails(before: str, after: str) -> None:
    result = check_side_effect_intro(before, after)
    assert not result.passed
    assert result.refusal_code == "R006_SAFETY_CHECKER_REJECTED_ALL"
    assert "new side effects" in result.reason


# ---------------------------------------------------------------------------
# side_effect_elimination
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "before, after",
    [
        # 1. Neither has side effects
        (
            "def f():\n    return 1\n",
            "def f():\n    return 1 + 0\n",
        ),
        # 2. print present in both
        (
            "def f():\n    print('a')\n    return 1\n",
            "def f():\n    print('b')\n    return 1\n",
        ),
        # 3. logging present in both
        (
            "def f():\n    logging.info('start')\n    return 1\n",
            "def f():\n    logging.warning('end')\n    return 1\n",
        ),
        # 4. os call present in both
        (
            "def f():\n    os.system('echo a')\n",
            "def f():\n    os.system('echo b')\n",
        ),
        # 5. requests in both
        (
            "def f():\n    requests.get('http://a')\n",
            "def f():\n    requests.get('http://b')\n",
        ),
    ],
)
def test_side_effect_elim_passes(before: str, after: str) -> None:
    result = check_side_effect_elim(before, after)
    assert result.passed, f"Expected safe but got: {result.reason}"


@pytest.mark.parametrize(
    "before, after",
    [
        # 1. Removes print
        (
            "def f():\n    print('debug')\n    return 1\n",
            "def f():\n    return 1\n",
        ),
        # 2. Removes logger call
        (
            "def f():\n    logger.info('x')\n    return 1\n",
            "def f():\n    return 1\n",
        ),
        # 3. Removes open/write
        (
            "def f():\n    open('out.txt', 'w').write('x')\n    return 1\n",
            "def f():\n    return 1\n",
        ),
        # 4. Removes os.system
        (
            "def f():\n    os.system('ls')\n    return 1\n",
            "def f():\n    return 1\n",
        ),
        # 5. Removes subprocess call
        (
            "def f():\n    subprocess.run(['ls'])\n    return 1\n",
            "def f():\n    return 1\n",
        ),
    ],
)
def test_side_effect_elim_fails(before: str, after: str) -> None:
    result = check_side_effect_elim(before, after)
    assert not result.passed
    assert result.refusal_code == "R006_SAFETY_CHECKER_REJECTED_ALL"
    assert "removes side effects" in result.reason


# ---------------------------------------------------------------------------
# exception_contract_change
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "before, after",
    [
        # 1. No raises in either
        (
            "def f():\n    return 1\n",
            "def f():\n    return 1\n",
        ),
        # 2. Same ValueError in both
        (
            "def f(x: int) -> int:\n    if x < 0:\n        raise ValueError('neg')\n    return x\n",
            (
                "def f(x: int) -> int:\n    if x <= 0:\n        raise ValueError('non-pos')"
                "\n    return x\n"
            ),
        ),
        # 3. Same TypeError in both
        (
            "def f():\n    raise TypeError('bad')\n",
            "def f():\n    raise TypeError('still bad')\n",
        ),
        # 4. Both raise custom.Exception (Attribute)
        (
            "def f():\n    raise custom.Error('a')\n",
            "def f():\n    raise custom.Error('b')\n",
        ),
        # 5. Multiple identical raises
        (
            "def f(x: int) -> int:\n    raise ValueError\n    raise TypeError\n",
            "def f(x: int) -> int:\n    raise TypeError\n    raise ValueError\n",
        ),
    ],
)
def test_exception_contract_passes(before: str, after: str) -> None:
    result = check_exception_contract(before, after)
    assert result.passed, f"Expected safe but got: {result.reason}"


@pytest.mark.parametrize(
    "before, after",
    [
        # 1. Adds ValueError
        (
            "def f():\n    return 1\n",
            "def f():\n    raise ValueError('oops')\n    return 1\n",
        ),
        # 2. Removes TypeError
        (
            "def f():\n    raise TypeError('bad')\n",
            "def f():\n    return 1\n",
        ),
        # 3. Changes exception name
        (
            "def f():\n    raise ValueError('x')\n",
            "def f():\n    raise RuntimeError('x')\n",
        ),
        # 4. Adds a new exception class on top of existing
        (
            "def f():\n    raise ValueError('x')\n",
            "def f():\n    raise ValueError('x')\n    raise KeyError('y')\n",
        ),
        # 5. Before has two, after only has one
        (
            "def f():\n    raise ValueError\n    raise KeyError\n",
            "def f():\n    raise ValueError\n",
        ),
    ],
)
def test_exception_contract_fails(before: str, after: str) -> None:
    result = check_exception_contract(before, after)
    assert not result.passed
    assert result.refusal_code == "R006_SAFETY_CHECKER_REJECTED_ALL"
    assert "exception contract changed" in result.reason


# ---------------------------------------------------------------------------
# public_api_change
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "before, after",
    [
        # 1. Same signature, pure body refactor
        (_SIMPLE_BEFORE, _SIMPLE_AFTER),
        # 2. No top-level functions
        (
            "x = 1\n",
            "x = 2\n",
        ),
        # 3. Same function name and params, no annotations
        (
            "def f(a, b):\n    return a + b\n",
            "def f(a, b):\n    c = a + b\n    return c\n",
        ),
        # 4. Multiple functions, same signatures
        (
            "def f(x: int) -> int:\n    return x\n\ndef g(y: str) -> str:\n    return y\n",
            "def f(x: int) -> int:\n    return x + 0\n\ndef g(y: str) -> str:\n    return y[:]\n",
        ),
        # 5. New function added in after (existing ones unchanged) — not a removal
        (
            "def f(x: int) -> int:\n    return x\n",
            "def f(x: int) -> int:\n    return x\n\ndef g(y: int) -> int:\n    return y\n",
        ),
    ],
)
def test_public_api_passes(before: str, after: str) -> None:
    result = check_public_api(before, after)
    assert result.passed, f"Expected safe but got: {result.reason}"


@pytest.mark.parametrize(
    "before, after",
    [
        # 1. Function renamed
        (
            "def f(x: int) -> int:\n    return x\n",
            "def g(x: int) -> int:\n    return x\n",
        ),
        # 2. Parameter added
        (
            "def f(x: int) -> int:\n    return x\n",
            "def f(x: int, y: int) -> int:\n    return x + y\n",
        ),
        # 3. Parameter removed
        (
            "def f(x: int, y: int) -> int:\n    return x + y\n",
            "def f(x: int) -> int:\n    return x\n",
        ),
        # 4. Return annotation changed
        (
            "def f(x: int) -> int:\n    return x\n",
            "def f(x: int) -> str:\n    return str(x)\n",
        ),
        # 5. Parameter renamed
        (
            "def f(x: int) -> int:\n    return x\n",
            "def f(z: int) -> int:\n    return z\n",
        ),
    ],
)
def test_public_api_fails(before: str, after: str) -> None:
    result = check_public_api(before, after)
    assert not result.passed
    assert result.refusal_code == "R006_SAFETY_CHECKER_REJECTED_ALL"


# ---------------------------------------------------------------------------
# async_sync_boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "before, after",
    [
        # 1. Both sync
        (
            "def f():\n    return 1\n",
            "def f():\n    return 1 + 0\n",
        ),
        # 2. Both async
        (
            "async def f():\n    return 1\n",
            "async def f():\n    await asyncio.sleep(0)\n    return 1\n",
        ),
        # 3. No functions
        (
            "x = 1\n",
            "x = 2\n",
        ),
        # 4. Multiple functions all unchanged
        (
            "def f():\n    pass\n\nasync def g():\n    pass\n",
            "def f():\n    pass\n\nasync def g():\n    pass\n",
        ),
        # 5. New function added (existing unchanged)
        (
            "def f():\n    pass\n",
            "def f():\n    pass\n\nasync def h():\n    pass\n",
        ),
    ],
)
def test_async_sync_passes(before: str, after: str) -> None:
    result = check_async_sync(before, after)
    assert result.passed, f"Expected safe but got: {result.reason}"


@pytest.mark.parametrize(
    "before, after",
    [
        # 1. sync → async
        (
            "def f():\n    pass\n",
            "async def f():\n    pass\n",
        ),
        # 2. async → sync
        (
            "async def f():\n    pass\n",
            "def f():\n    pass\n",
        ),
        # 3. One of two functions flips
        (
            "def f():\n    pass\n\ndef g():\n    pass\n",
            "def f():\n    pass\n\nasync def g():\n    pass\n",
        ),
        # 4. async → sync via different body
        (
            "async def fetch(url: str) -> str:\n    return url\n",
            "def fetch(url: str) -> str:\n    return url\n",
        ),
        # 5. sync → async with await in body
        (
            "def process(data: list[int]) -> list[int]:\n    return sorted(data)\n",
            "async def process(data: list[int]) -> list[int]:\n    return sorted(data)\n",
        ),
    ],
)
def test_async_sync_fails(before: str, after: str) -> None:
    result = check_async_sync(before, after)
    assert not result.passed
    assert result.refusal_code == "R006_SAFETY_CHECKER_REJECTED_ALL"
    assert "async/sync boundary changed" in result.reason


# ---------------------------------------------------------------------------
# filter chain (SafetyResult / run)
# ---------------------------------------------------------------------------


def test_filter_passing_candidate() -> None:
    """A clean refactor should pass the full filter chain."""
    result = run(_SIMPLE_BEFORE, _SIMPLE_AFTER)
    assert isinstance(result, SafetyResult)
    assert result.passed
    assert result.reason == ""
    assert result.refusal_code == ""


def test_filter_rejects_side_effect_intro() -> None:
    """Adding print should trigger R006 from the filter chain."""
    before = "def f():\n    return 1\n"
    after = "def f():\n    print('debug')\n    return 1\n"
    result = run(before, after)
    assert not result.passed
    assert result.refusal_code == "R006_SAFETY_CHECKER_REJECTED_ALL"
    assert "side effects" in result.reason


def test_filter_short_circuits() -> None:
    """When the first checker fails, reason should come from it (not later ones)."""
    # Introduce BOTH a side-effect change AND an async boundary change.
    # The side-effect checker runs first (index 0), so its reason must win.
    before = "def f():\n    return 1\n"
    after = "async def f():\n    print('x')\n    return 1\n"
    result = run(before, after)
    assert not result.passed
    # side_effect_intro fires before async_sync
    assert "side effects" in result.reason


def test_filter_result_has_correct_refusal_code() -> None:
    """Every failing result must carry R006."""
    cases = [
        # side effect intro
        ("def f():\n    return 1\n", "def f():\n    os.system('x')\n    return 1\n"),
        # exception change
        ("def f():\n    return 1\n", "def f():\n    raise ValueError\n"),
        # async boundary
        ("def f():\n    return 1\n", "async def f():\n    return 1\n"),
    ]
    for before, after in cases:
        result = run(before, after)
        assert not result.passed
        assert (
            result.refusal_code == "R006_SAFETY_CHECKER_REJECTED_ALL"
        ), f"Expected R006 for pair: {before!r} / {after!r}"
