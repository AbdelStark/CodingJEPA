"""Unit tests for the 8 intent labelers (issues #41–#48).

Per RFC-0002 §D6, labelers are *conservative*: they should prefer false
negatives over false positives. They are stricter than the permissive
acceptance heuristics in `codingjepa.intents.acceptance` (used as a gate).

For each labeler:
- ≥ 3 positive cases (matched=True)
- ≥ 3 negative cases (matched=False)
- edge cases (empty / trivial / parse-error inputs)

Also:
- `label_pair` returns "NONE" for trivial changes.
- `LABELERS` registry exposes all 8 intents.
"""

from __future__ import annotations

import textwrap

import pytest

from codingjepa.data.labelers import LABELERS, label_pair
from codingjepa.data.labelers.argument_defaulting import argument_defaulting_labeler
from codingjepa.data.labelers.comprehension_rewrite import comprehension_rewrite_labeler
from codingjepa.data.labelers.dataclass_migration import dataclass_migration_labeler
from codingjepa.data.labelers.exception_handling_cleanup import (
    exception_handling_cleanup_labeler,
)
from codingjepa.data.labelers.extract_helper import extract_helper_labeler
from codingjepa.data.labelers.inline_helper import inline_helper_labeler
from codingjepa.data.labelers.loop_to_vectorized import loop_to_vectorized_labeler
from codingjepa.data.labelers.none_typing_modernization import (
    none_typing_modernization_labeler,
)


def _src(code: str) -> str:
    return textwrap.dedent(code).strip() + "\n"


# ============================================================================
# extract-helper (#41)
# ============================================================================


EXTRACT_HELPER_POSITIVE = [
    # Strong: new top-level helper, body has ≥ 2 stmts, called at move site.
    (
        _src("""
            def f(x):
                a = x + 1
                b = a * 2
                return b
            """),
        _src("""
            def helper(x):
                a = x + 1
                b = a * 2
                return b

            def f(x):
                return helper(x)
            """),
    ),
    (
        _src("""
            def g(xs):
                total = 0
                for x in xs:
                    total += x * x
                return total
            """),
        _src("""
            def sum_squares(xs):
                total = 0
                for x in xs:
                    total += x * x
                return total

            def g(xs):
                return sum_squares(xs)
            """),
    ),
    (
        _src("""
            def h(s):
                x = s.upper()
                y = x.strip()
                return y
            """),
        _src("""
            def normalize(s):
                x = s.upper()
                y = x.strip()
                return y

            def h(s):
                return normalize(s)
            """),
    ),
]

EXTRACT_HELPER_NEGATIVE = [
    # No change at all.
    (_src("def f(x):\n    return x + 1\n"), _src("def f(x):\n    return x + 1\n")),
    # New helper added but never called.
    (
        _src("def f(x):\n    return x\n"),
        _src("def unused(x):\n    return x\n\ndef f(x):\n    return x\n"),
    ),
    # Single-line helper (body has < 2 statements) → conservatively reject.
    (
        _src("def f(x):\n    return x + 1\n"),
        _src("def trivial(x):\n    return x + 1\n\ndef f(x):\n    return trivial(x)\n"),
    ),
    # Class added, not a function.
    (_src("class A:\n    pass\n"), _src("class B:\n    pass\n")),
]


# ============================================================================
# inline-helper (#42)
# ============================================================================


INLINE_HELPER_POSITIVE = [
    (
        _src("""
            def helper(x):
                return x + 1

            def f(x):
                return helper(x)
            """),
        _src("""
            def f(x):
                return x + 1
            """),
    ),
    (
        _src("""
            def add(a, b):
                return a + b

            def use():
                return add(1, 2)
            """),
        _src("""
            def use():
                return 1 + 2
            """),
    ),
    (
        _src("""
            def to_int(x):
                return int(x)

            def main(s):
                return to_int(s)
            """),
        _src("""
            def main(s):
                return int(s)
            """),
    ),
]

INLINE_HELPER_NEGATIVE = [
    # No change.
    (_src("def f(x):\n    return x\n"), _src("def f(x):\n    return x\n")),
    # No removal happened.
    (
        _src("def f(x):\n    return x\n"),
        _src("def f(x):\n    return x + 1\n"),
    ),
    # Helper removed but still called somewhere (broken inline).
    (
        _src("def helper(x):\n    return x\n\ndef f(x):\n    return helper(x)\n"),
        _src("def f(x):\n    return helper(x)\n"),
    ),
    # Helper definition added (not removed).
    (
        _src("def f(x):\n    return x\n"),
        _src("def new_helper(y):\n    return y\n\ndef f(x):\n    return x\n"),
    ),
]


# ============================================================================
# comprehension-rewrite (#43)
# ============================================================================


COMPREHENSION_REWRITE_POSITIVE = [
    # list comp with append
    (
        _src("""
            def f(xs):
                r = []
                for x in xs:
                    r.append(x * 2)
                return r
            """),
        _src("""
            def f(xs):
                return [x * 2 for x in xs]
            """),
    ),
    # set comp with add
    (
        _src("""
            def f(xs):
                s = set()
                for x in xs:
                    s.add(x)
                return s
            """),
        _src("""
            def f(xs):
                return {x for x in xs}
            """),
    ),
    # dict comp
    (
        _src("""
            def f(items):
                d = {}
                for k, v in items:
                    d[k] = v
                return d
            """),
        _src("""
            def f(items):
                return {k: v for k, v in items}
            """),
    ),
]

COMPREHENSION_REWRITE_NEGATIVE = [
    # No change.
    (_src("def f(xs):\n    return xs\n"), _src("def f(xs):\n    return xs\n")),
    # Loop preserved in after.
    (
        _src("def f(xs):\n    r = []\n    for x in xs:\n        r.append(x)\n    return r\n"),
        _src("def f(xs):\n    r = []\n    for x in xs:\n        r.append(x * 2)\n    return r\n"),
    ),
    # Break in loop → unsafe; should reject.
    (
        _src(
            "def f(xs):\n"
            "    r = []\n"
            "    for x in xs:\n"
            "        if x < 0:\n"
            "            break\n"
            "        r.append(x)\n"
            "    return r\n"
        ),
        _src("def f(xs):\n    return [x for x in xs if x >= 0]\n"),
    ),
    # Comprehension was already present in before.
    (
        _src("def f():\n    return [x for x in range(3)]\n"),
        _src("def f():\n    return [x for x in range(3)]\n"),
    ),
]


# ============================================================================
# dataclass-migration (#44)
# ============================================================================


DATACLASS_MIGRATION_POSITIVE = [
    (
        _src("""
            class P:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
            """),
        _src("""
            @dataclass
            class P:
                x: int
                y: int
            """),
    ),
    (
        _src("""
            class Cfg:
                def __init__(self, host, port):
                    self.host = host
                    self.port = port
            """),
        _src("""
            @dataclass
            class Cfg:
                host: str
                port: int
            """),
    ),
    (
        _src("""
            class Person:
                def __init__(self, name, age):
                    self.name = name
                    self.age = age
            """),
        _src("""
            @dataclasses.dataclass
            class Person:
                name: str
                age: int
            """),
    ),
]

DATACLASS_MIGRATION_NEGATIVE = [
    # No change.
    (
        _src("class A:\n    def __init__(self, x):\n        self.x = x\n"),
        _src("class A:\n    def __init__(self, x):\n        self.x = x\n"),
    ),
    # Derived state in __init__ (self.x = x*2) → reject.
    (
        _src("class A:\n    def __init__(self, x):\n        self.x = x*2\n"),
        _src("@dataclass\nclass A:\n    x: int\n"),
    ),
    # Single field — labeler requires ≥ 2 fields per RFC-0002 §D6.
    (
        _src("""
            class Pt:
                def __init__(self, x):
                    self.x = x
            """),
        _src("""
            @dataclass
            class Pt:
                x: int
            """),
    ),
    # @dataclass on something with __init__ still — not a clean migration.
    (
        _src("class A:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n"),
        _src(
            "@dataclass\nclass A:\n"
            "    x: int\n"
            "    y: int\n"
            "    def __init__(self, x, y):\n"
            "        self.x = x\n"
            "        self.y = y\n"
        ),
    ),
]


# ============================================================================
# exception-handling-cleanup (#45)
# ============================================================================


EXCEPTION_HANDLING_POSITIVE = [
    # bare except → typed except + raise
    (
        _src("try:\n    f()\nexcept:\n    pass\n"),
        _src("try:\n    f()\nexcept Exception:\n    raise\n"),
    ),
    # bare except + log → typed except + log
    (
        _src("try:\n    f()\nexcept:\n    log()\n"),
        _src("try:\n    f()\nexcept Exception:\n    log()\n"),
    ),
    # try/except/pass → contextlib.suppress
    (
        _src("try:\n    f()\nexcept Exception:\n    pass\n"),
        _src("with contextlib.suppress(Exception):\n    f()\n"),
    ),
]

EXCEPTION_HANDLING_NEGATIVE = [
    # No change.
    (
        _src("try:\n    f()\nexcept Exception:\n    raise\n"),
        _src("try:\n    f()\nexcept Exception:\n    raise\n"),
    ),
    # No try/except in before.
    (_src("def f():\n    return 1\n"), _src("def f():\n    return 2\n")),
    # Regression — got worse (typed → bare).
    (
        _src("try:\n    f()\nexcept Exception:\n    raise\n"),
        _src("try:\n    f()\nexcept:\n    pass\n"),
    ),
]


# ============================================================================
# loop-to-vectorized (#46)
# ============================================================================


LOOP_TO_VECTORIZED_POSITIVE = [
    # for over df['c'] → .apply
    (
        _src("for x in df['c']:\n    print(x)\n"),
        _src("df['c'].apply(print)\n"),
    ),
    # for over series → .apply
    (
        _src("for x in s:\n    out.append(x*2)\n"),
        _src("out = s.apply(lambda x: x*2)\n"),
    ),
    # for over arr → .map
    (
        _src("for v in arr:\n    print(v)\n"),
        _src("arr.map(print)\n"),
    ),
]

LOOP_TO_VECTORIZED_NEGATIVE = [
    # No change.
    (_src("for x in xs:\n    pass\n"), _src("for x in xs:\n    pass\n")),
    # No loop in before.
    (_src("xs = []\n"), _src("xs.apply(f)\n")),
    # No vectorized API in after.
    (_src("for x in xs:\n    print(x)\n"), _src("xs.foo()\n")),
    # Iterating over a non-pandas/numpy-looking variable (conservative).
    (
        _src("for thing in collection:\n    print(thing)\n"),
        _src("collection.apply(print)\n"),
    ),
]


# ============================================================================
# argument-defaulting (#47)
# ============================================================================


ARGUMENT_DEFAULTING_POSITIVE = [
    (_src("def f(a, b):\n    return a + b\n"), _src("def f(a, b=0):\n    return a + b\n")),
    (_src("def f(x, y):\n    return x * y\n"), _src("def f(x, y=1):\n    return x * y\n")),
    (
        _src("def g(a, b, c):\n    return a + b + c\n"),
        _src("def g(a, b, c=0):\n    return a + b + c\n"),
    ),
]

ARGUMENT_DEFAULTING_NEGATIVE = [
    # No change.
    (_src("def f(a, b):\n    return a + b\n"), _src("def f(a, b):\n    return a + b\n")),
    # Default was already present.
    (
        _src("def f(a, b=0):\n    return a + b\n"),
        _src("def f(a, b=0):\n    return a + b\n"),
    ),
    # Mutable default → reject per RFC-0004.
    (
        _src("def f(a, b):\n    return a + b\n"),
        _src("def f(a, b=[]):\n    return a + b\n"),
    ),
    (
        _src("def f(a, b):\n    return a + b\n"),
        _src("def f(a, b={}):\n    return a + b\n"),
    ),
    # New parameter added (not a default).
    (
        _src("def f(a, b):\n    return a\n"),
        _src("def f(a, b, c):\n    return a\n"),
    ),
]


# ============================================================================
# none-typing-modernization (#48)
# ============================================================================


NONE_TYPING_POSITIVE = [
    # Optional[X] → X | None
    (
        _src("def f(x: Optional[int]) -> int:\n    return x or 0\n"),
        _src("def f(x: int | None) -> int:\n    return x or 0\n"),
    ),
    # Union[A, B] → A | B
    (_src("x: Union[int, str] = 1\n"), _src("x: int | str = 1\n")),
    # typing.List → list
    (
        _src("def k(xs: typing.List[int]) -> int:\n    return sum(xs)\n"),
        _src("def k(xs: list[int]) -> int:\n    return sum(xs)\n"),
    ),
]

NONE_TYPING_NEGATIVE = [
    # No change, no legacy.
    (
        _src("def f(x: int) -> int:\n    return x\n"),
        _src("def f(x: int) -> int:\n    return x\n"),
    ),
    # Unchanged legacy.
    (
        _src("def f(x: Optional[int]) -> int:\n    return x or 0\n"),
        _src("def f(x: Optional[int]) -> int:\n    return x or 0\n"),
    ),
    # Reverse direction (modern → legacy).
    (
        _src("def f(x: int | None) -> int:\n    return x or 0\n"),
        _src("def f(x: Optional[int]) -> int:\n    return x or 0\n"),
    ),
    # Legacy introduced (regression).
    (
        _src("def f(x: int) -> int:\n    return x\n"),
        _src("def f(x: Optional[int]) -> int:\n    return x or 0\n"),
    ),
]


# ============================================================================
# Per-labeler parametrized tests
# ============================================================================


@pytest.mark.parametrize("before, after", EXTRACT_HELPER_POSITIVE)
def test_extract_helper_positive(before: str, after: str) -> None:
    matched, confidence = extract_helper_labeler(before, after)
    assert matched is True
    assert 0.5 <= confidence <= 1.0


@pytest.mark.parametrize("before, after", EXTRACT_HELPER_NEGATIVE)
def test_extract_helper_negative(before: str, after: str) -> None:
    matched, confidence = extract_helper_labeler(before, after)
    assert matched is False
    assert confidence == 0.0


@pytest.mark.parametrize("before, after", INLINE_HELPER_POSITIVE)
def test_inline_helper_positive(before: str, after: str) -> None:
    matched, confidence = inline_helper_labeler(before, after)
    assert matched is True
    assert 0.5 <= confidence <= 1.0


@pytest.mark.parametrize("before, after", INLINE_HELPER_NEGATIVE)
def test_inline_helper_negative(before: str, after: str) -> None:
    matched, confidence = inline_helper_labeler(before, after)
    assert matched is False
    assert confidence == 0.0


@pytest.mark.parametrize("before, after", COMPREHENSION_REWRITE_POSITIVE)
def test_comprehension_rewrite_positive(before: str, after: str) -> None:
    matched, confidence = comprehension_rewrite_labeler(before, after)
    assert matched is True
    assert 0.5 <= confidence <= 1.0


@pytest.mark.parametrize("before, after", COMPREHENSION_REWRITE_NEGATIVE)
def test_comprehension_rewrite_negative(before: str, after: str) -> None:
    matched, confidence = comprehension_rewrite_labeler(before, after)
    assert matched is False
    assert confidence == 0.0


@pytest.mark.parametrize("before, after", DATACLASS_MIGRATION_POSITIVE)
def test_dataclass_migration_positive(before: str, after: str) -> None:
    matched, confidence = dataclass_migration_labeler(before, after)
    assert matched is True
    assert 0.5 <= confidence <= 1.0


@pytest.mark.parametrize("before, after", DATACLASS_MIGRATION_NEGATIVE)
def test_dataclass_migration_negative(before: str, after: str) -> None:
    matched, confidence = dataclass_migration_labeler(before, after)
    assert matched is False
    assert confidence == 0.0


@pytest.mark.parametrize("before, after", EXCEPTION_HANDLING_POSITIVE)
def test_exception_handling_cleanup_positive(before: str, after: str) -> None:
    matched, confidence = exception_handling_cleanup_labeler(before, after)
    assert matched is True
    assert 0.5 <= confidence <= 1.0


@pytest.mark.parametrize("before, after", EXCEPTION_HANDLING_NEGATIVE)
def test_exception_handling_cleanup_negative(before: str, after: str) -> None:
    matched, confidence = exception_handling_cleanup_labeler(before, after)
    assert matched is False
    assert confidence == 0.0


@pytest.mark.parametrize("before, after", LOOP_TO_VECTORIZED_POSITIVE)
def test_loop_to_vectorized_positive(before: str, after: str) -> None:
    matched, confidence = loop_to_vectorized_labeler(before, after)
    assert matched is True
    assert 0.5 <= confidence <= 1.0


@pytest.mark.parametrize("before, after", LOOP_TO_VECTORIZED_NEGATIVE)
def test_loop_to_vectorized_negative(before: str, after: str) -> None:
    matched, confidence = loop_to_vectorized_labeler(before, after)
    assert matched is False
    assert confidence == 0.0


@pytest.mark.parametrize("before, after", ARGUMENT_DEFAULTING_POSITIVE)
def test_argument_defaulting_positive(before: str, after: str) -> None:
    matched, confidence = argument_defaulting_labeler(before, after)
    assert matched is True
    assert 0.5 <= confidence <= 1.0


@pytest.mark.parametrize("before, after", ARGUMENT_DEFAULTING_NEGATIVE)
def test_argument_defaulting_negative(before: str, after: str) -> None:
    matched, confidence = argument_defaulting_labeler(before, after)
    assert matched is False
    assert confidence == 0.0


@pytest.mark.parametrize("before, after", NONE_TYPING_POSITIVE)
def test_none_typing_modernization_positive(before: str, after: str) -> None:
    matched, confidence = none_typing_modernization_labeler(before, after)
    assert matched is True
    assert 0.5 <= confidence <= 1.0


@pytest.mark.parametrize("before, after", NONE_TYPING_NEGATIVE)
def test_none_typing_modernization_negative(before: str, after: str) -> None:
    matched, confidence = none_typing_modernization_labeler(before, after)
    assert matched is False
    assert confidence == 0.0


# ============================================================================
# Parse-error / edge-case handling
# ============================================================================


@pytest.mark.parametrize(
    "labeler",
    [
        extract_helper_labeler,
        inline_helper_labeler,
        comprehension_rewrite_labeler,
        dataclass_migration_labeler,
        exception_handling_cleanup_labeler,
        loop_to_vectorized_labeler,
        argument_defaulting_labeler,
        none_typing_modernization_labeler,
    ],
)
def test_labeler_handles_parse_errors(labeler) -> None:  # type: ignore[no-untyped-def]
    """libcst parse errors must be swallowed — return (False, 0.0)."""

    bad = "def f(:\n    return\n"
    matched, confidence = labeler(bad, "def f(): return 1\n")
    assert matched is False
    assert confidence == 0.0

    matched, confidence = labeler("def f(): return 1\n", bad)
    assert matched is False
    assert confidence == 0.0


@pytest.mark.parametrize(
    "labeler",
    [
        extract_helper_labeler,
        inline_helper_labeler,
        comprehension_rewrite_labeler,
        dataclass_migration_labeler,
        exception_handling_cleanup_labeler,
        loop_to_vectorized_labeler,
        argument_defaulting_labeler,
        none_typing_modernization_labeler,
    ],
)
def test_labeler_handles_empty_input(labeler) -> None:  # type: ignore[no-untyped-def]
    matched, confidence = labeler("", "")
    assert matched is False
    assert confidence == 0.0


@pytest.mark.parametrize(
    "labeler",
    [
        extract_helper_labeler,
        inline_helper_labeler,
        comprehension_rewrite_labeler,
        dataclass_migration_labeler,
        exception_handling_cleanup_labeler,
        loop_to_vectorized_labeler,
        argument_defaulting_labeler,
        none_typing_modernization_labeler,
    ],
)
def test_labeler_confidence_in_range(labeler) -> None:  # type: ignore[no-untyped-def]
    """Confidence is always in [0,1] regardless of inputs."""

    matched, confidence = labeler("x = 1\n", "y = 2\n")
    assert 0.0 <= confidence <= 1.0
    assert isinstance(matched, bool)


# ============================================================================
# Registry & label_pair
# ============================================================================


def test_labelers_registry_has_8_entries() -> None:
    assert len(LABELERS) == 8
    expected = {
        "extract-helper",
        "inline-helper",
        "comprehension-rewrite",
        "dataclass-migration",
        "exception-handling-cleanup",
        "loop-to-vectorized",
        "argument-defaulting",
        "none-typing-modernization",
    }
    assert set(LABELERS.keys()) == expected


def test_labelers_registry_all_callable() -> None:
    for name, fn in LABELERS.items():
        assert callable(fn), f"{name} is not callable"


def test_label_pair_returns_none_for_trivial_change() -> None:
    intent, confidence = label_pair("x = 1\n", "x = 1\n")
    assert intent == "NONE"
    assert confidence == 0.0


def test_label_pair_returns_none_for_unrelated_change() -> None:
    intent, confidence = label_pair("x = 1\n", "y = 2\n")
    assert intent == "NONE"
    assert confidence == 0.0


def test_label_pair_detects_comprehension_rewrite() -> None:
    before = _src("""
        def f(xs):
            r = []
            for x in xs:
                r.append(x * 2)
            return r
        """)
    after = _src("""
        def f(xs):
            return [x * 2 for x in xs]
        """)
    intent, confidence = label_pair(before, after)
    assert intent == "comprehension-rewrite"
    assert 0.5 <= confidence <= 1.0


def test_label_pair_detects_dataclass_migration() -> None:
    before = _src("""
        class P:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        """)
    after = _src("""
        @dataclass
        class P:
            x: int
            y: int
        """)
    intent, confidence = label_pair(before, after)
    assert intent == "dataclass-migration"
    assert 0.5 <= confidence <= 1.0


def test_label_pair_detects_none_typing_modernization() -> None:
    before = _src("def f(x: Optional[int]) -> int:\n    return x or 0\n")
    after = _src("def f(x: int | None) -> int:\n    return x or 0\n")
    intent, confidence = label_pair(before, after)
    assert intent == "none-typing-modernization"
    assert 0.5 <= confidence <= 1.0


def test_label_pair_returns_highest_confidence_when_ambiguous() -> None:
    """If multiple labelers fire, the highest-confidence wins (stable break tie)."""

    # An extract-helper case (does not also fire comprehension-rewrite).
    before = _src("""
        def f(x):
            a = x + 1
            b = a * 2
            return b
        """)
    after = _src("""
        def helper(x):
            a = x + 1
            b = a * 2
            return b

        def f(x):
            return helper(x)
        """)
    intent, confidence = label_pair(before, after)
    assert intent == "extract-helper"
    assert confidence > 0.0


def test_label_pair_with_parse_error() -> None:
    """A parse error → NONE / 0.0 (not a crash)."""

    intent, confidence = label_pair("def f(:\n", "def g():\n    return 1\n")
    assert intent == "NONE"
    assert confidence == 0.0
