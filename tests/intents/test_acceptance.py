"""10 positive + 10 negative cases per acceptance rule (RFC-0004 §D2).

Each case is a (before_source, after_source, expected_bool) triple. The
heuristics are intentionally permissive per RFC-0004 §D2; labelers are
expected to be more conservative.
"""

from __future__ import annotations

import textwrap

import pytest

from codingjepa.intents import acceptance_check
from codingjepa.intents.acceptance import parse


def _src(code: str) -> str:
    return textwrap.dedent(code).strip() + "\n"


# ============================ extract-helper ================================

EXTRACT_HELPER = [
    # ---- Positive: a new top-level function appears and is called ----------
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
        True,
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
        True,
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
        True,
    ),
    (
        _src("""
        def p(a, b):
            c = a + b
            d = c * 2
            return d
    """),
        _src("""
        def combine(a, b):
            c = a + b
            d = c * 2
            return d

        def p(a, b):
            return combine(a, b)
    """),
        True,
    ),
    (
        _src("""
        def m(xs):
            xs.sort()
            return xs[-1]
    """),
        _src("""
        def sort_and_max(xs):
            xs.sort()
            return xs[-1]

        def m(xs):
            return sort_and_max(xs)
    """),
        True,
    ),
    (
        _src("""
        def write_thing(x, fh):
            fh.write(str(x))
            fh.flush()
    """),
        _src("""
        def write_line(fh, s):
            fh.write(s)
            fh.flush()

        def write_thing(x, fh):
            write_line(fh, str(x))
    """),
        True,
    ),
    (
        _src("""
        def k(seq):
            r = []
            for v in seq:
                r.append(v + 1)
            return r
    """),
        _src("""
        def increment_all(seq):
            r = []
            for v in seq:
                r.append(v + 1)
            return r

        def k(seq):
            return increment_all(seq)
    """),
        True,
    ),
    (
        _src("""
        def filt(seq):
            r = []
            for v in seq:
                if v > 0:
                    r.append(v)
            return r
    """),
        _src("""
        def positives(seq):
            r = []
            for v in seq:
                if v > 0:
                    r.append(v)
            return r

        def filt(seq):
            return positives(seq)
    """),
        True,
    ),
    (
        _src("""
        def parse_line(s):
            parts = s.split(',')
            return [p.strip() for p in parts]
    """),
        _src("""
        def split_csv(s):
            parts = s.split(',')
            return [p.strip() for p in parts]

        def parse_line(s):
            return split_csv(s)
    """),
        True,
    ),
    (
        _src("""
        def caller():
            r = compute_x() + compute_y()
            return r
    """),
        _src("""
        def both():
            return compute_x() + compute_y()

        def caller():
            return both()
    """),
        True,
    ),
    # ---- Negative: no new helper, or new helper never called ----------------
    (_src("def f(x): return x + 1\n"), _src("def f(x): return x + 1\n"), False),
    (_src("def f(x):\n    return x\n"), _src("def f(x):\n    return x + 1\n"), False),
    (
        _src("def f(x):\n    return x\n"),
        _src("def unused(x): return x\ndef f(x): return x\n"),
        False,
    ),
    (
        _src("def f(x):\n    return x\n"),
        _src("def f(x):\n    return x\ndef g(x):\n    return x*2\n"),
        False,
    ),
    (_src("def f(x):\n    return x*2\n"), _src("def f(x):\n    return x*2 + 1\n"), False),
    (_src("def f():\n    pass\n"), _src("def f():\n    pass\n"), False),
    (_src("def f(x):\n    return x\n"), _src("# only a comment\ndef f(x):\n    return x\n"), False),
    (_src("import os\n"), _src("import os\nimport sys\n"), False),
    (_src("class A:\n    pass\n"), _src("class B:\n    pass\n"), False),
    (_src("def a():\n    return 1\n"), _src("def a():\n    return 2\n"), False),
]

# ============================ inline-helper =================================

INLINE_HELPER = [
    # ---- Positive: a helper present in before is gone & uncalled in after ---
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
        True,
    ),
    (
        _src("""
        def small(x): return x*2

        def main():
            return small(3)
    """),
        _src("""
        def main():
            return 3*2
    """),
        True,
    ),
    (
        _src("""
        def add(a, b): return a + b

        def use():
            return add(1, 2)
    """),
        _src("""
        def use():
            return 1 + 2
    """),
        True,
    ),
    (
        _src("""
        def parse(s): return s.split(',')

        def go(s):
            return parse(s)
    """),
        _src("""
        def go(s):
            return s.split(',')
    """),
        True,
    ),
    (
        _src("""
        def to_int(x): return int(x)

        def main(s):
            return to_int(s)
    """),
        _src("""
        def main(s):
            return int(s)
    """),
        True,
    ),
    (
        _src("""
        def square(x): return x*x

        def sum_sq(xs):
            return sum(square(x) for x in xs)
    """),
        _src("""
        def sum_sq(xs):
            return sum(x*x for x in xs)
    """),
        True,
    ),
    (
        _src("""
        def negate(x): return -x

        def main(x):
            return negate(x)
    """),
        _src("""
        def main(x):
            return -x
    """),
        True,
    ),
    (
        _src("""
        def reverse(s): return s[::-1]

        def main(s):
            return reverse(s)
    """),
        _src("""
        def main(s):
            return s[::-1]
    """),
        True,
    ),
    (
        _src("""
        def upper(s): return s.upper()

        def main(s):
            return upper(s)
    """),
        _src("""
        def main(s):
            return s.upper()
    """),
        True,
    ),
    (
        _src("""
        def is_pos(x): return x > 0

        def main(x):
            return is_pos(x)
    """),
        _src("""
        def main(x):
            return x > 0
    """),
        True,
    ),
    # ---- Negative ---------------------------------------------------------
    (_src("def f(x):\n    return x\n"), _src("def f(x):\n    return x\n"), False),
    (_src("def f(x):\n    return x\n"), _src("def f(x):\n    return x + 1\n"), False),
    (
        _src("def helper(x):\n    return x\n\ndef f(x):\n    return helper(x)\n"),
        _src("def helper(x):\n    return x\n\ndef f(x):\n    return helper(x)\n"),
        False,
    ),
    (_src("def helper(x):\n    return x\n"), _src("def helper(x):\n    return x\n"), False),
    (
        _src("def f(x):\n    return x\n"),
        _src("def f(x):\n    return x\n\ndef new_helper(y):\n    return y\n"),
        False,
    ),
    (_src("class A:\n    pass\n"), _src("class A:\n    pass\n"), False),
    (
        _src("def f(x):\n    return helper(x)\n"),
        _src("def f(x):\n    return helper(x)\n"),
        False,
    ),
    # Removed helper but still called somewhere — fails.
    (
        _src("def helper(x):\n    return x\n\ndef f(x):\n    return helper(x)\n"),
        _src("def f(x):\n    return helper(x)\n"),
        False,
    ),
    (_src("import os\n"), _src("import sys\n"), False),
    (
        _src("def main():\n    return 1\n"),
        _src("def main():\n    return 2\n"),
        False,
    ),
]

# ============================ comprehension-rewrite =========================

COMPREHENSION_REWRITE = [
    # ---- Positive: for+append → list comprehension --------------------------
    (
        _src("""
            def f(xs):
                r = []
                for x in xs:
                    r.append(x*2)
                return r
        """),
        _src("""
            def f(xs):
                return [x*2 for x in xs]
        """),
        True,
    ),
    (
        _src("""
            def f(xs):
                r = []
                for x in xs:
                    r.append(x + 1)
                return r
        """),
        _src("""
            def f(xs):
                return [x + 1 for x in xs]
        """),
        True,
    ),
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
        True,
    ),
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
        True,
    ),
    (
        _src("""
            def f(xs):
                r = []
                for x in xs:
                    r.append(x*x)
                return r
        """),
        _src("""
            def f(xs):
                return [x*x for x in xs]
        """),
        True,
    ),
    (
        _src("""
            def f(xs):
                r = []
                for x in xs:
                    r.append(str(x))
                return r
        """),
        _src("""
            def f(xs):
                return [str(x) for x in xs]
        """),
        True,
    ),
    (
        _src("""
            def f(xs):
                r = []
                for x in xs:
                    r.append(x.strip())
                return r
        """),
        _src("""
            def f(xs):
                return [x.strip() for x in xs]
        """),
        True,
    ),
    (
        _src("""
            def f(words):
                r = []
                for w in words:
                    r.append(w.upper())
                return r
        """),
        _src("""
            def f(words):
                return [w.upper() for w in words]
        """),
        True,
    ),
    (
        _src("""
            def f(seq):
                out = []
                for v in seq:
                    out.append(v + 10)
                return out
        """),
        _src("""
            def f(seq):
                return [v + 10 for v in seq]
        """),
        True,
    ),
    (
        _src("""
            def f(seq):
                bag = set()
                for v in seq:
                    bag.add(v.lower())
                return bag
        """),
        _src("""
            def f(seq):
                return {v.lower() for v in seq}
        """),
        True,
    ),
    # ---- Negative ---------------------------------------------------------
    (_src("def f(xs):\n    return xs\n"), _src("def f(xs):\n    return xs\n"), False),
    (
        _src("def f(xs):\n    r = []\n    for x in xs:\n        r.append(x)\n    return r\n"),
        _src("def f(xs):\n    r = []\n    for x in xs:\n        r.append(x)\n    return r\n"),
        False,
    ),
    # break in the loop → unsafe to comprehend.
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
        False,
    ),
    (
        _src(
            "def f(xs):\n"
            "    r = []\n"
            "    for x in xs:\n"
            "        if x < 0:\n"
            "            continue\n"
            "        r.append(x)\n"
            "    return r\n"
        ),
        _src("def f(xs):\n    return [x for x in xs if x >= 0]\n"),
        False,
    ),
    (
        _src("def f():\n    return [x for x in range(3)]\n"),
        _src("def f():\n    return [x for x in range(3)]\n"),
        False,
    ),
    (
        _src("def f(xs):\n    return list(xs)\n"),
        _src("def f(xs):\n    return [x for x in xs]\n"),
        False,
    ),
    # Loop preserved in after → no rewrite.
    (
        _src("def f(xs):\n    r = []\n    for x in xs:\n        r.append(x)\n    return r\n"),
        _src("def f(xs):\n    r = []\n    for x in xs:\n        r.append(x*2)\n    return r\n"),
        False,
    ),
    (_src("import os\n"), _src("import os\n"), False),
    (_src("def f():\n    return 1\n"), _src("def f():\n    return [1]\n"), False),
    (_src("class A:\n    pass\n"), _src("class A:\n    pass\n"), False),
]

# ============================ dataclass-migration ===========================

DATACLASS_MIGRATION = [
    # ---- Positive --------------------------------------------------------
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
        True,
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
        True,
    ),
    (
        _src("""
            class A:
                def __init__(self, a):
                    self.a = a
        """),
        _src("""
            @dataclass
            class A:
                a: int
        """),
        True,
    ),
    (
        _src("""
            class B:
                def __init__(self, b, c):
                    self.b = b
                    self.c = c
        """),
        _src("""
            @dataclass(frozen=True)
            class B:
                b: int
                c: int
        """),
        True,
    ),
    (
        _src("""
            class Box:
                def __init__(self, w, h, d):
                    self.w = w
                    self.h = h
                    self.d = d
        """),
        _src("""
            @dataclass
            class Box:
                w: int
                h: int
                d: int
        """),
        True,
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
        True,
    ),
    (
        _src("""
            class Pixel:
                def __init__(self, r, g, b):
                    self.r = r
                    self.g = g
                    self.b = b
        """),
        _src("""
            @dataclass
            class Pixel:
                r: int
                g: int
                b: int
        """),
        True,
    ),
    (
        _src("""
            class Pair:
                def __init__(self, a, b):
                    self.a = a
                    self.b = b
        """),
        _src("""
            @dataclass
            class Pair:
                a: int
                b: int
        """),
        True,
    ),
    (
        _src("""
            class Item:
                def __init__(self, sku, qty):
                    self.sku = sku
                    self.qty = qty
        """),
        _src("""
            @dataclass
            class Item:
                sku: str
                qty: int
        """),
        True,
    ),
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
        True,
    ),
    # ---- Negative --------------------------------------------------------
    (
        _src("class A:\n    def __init__(self, x):\n        self.x = x\n"),
        _src("class A:\n    def __init__(self, x):\n        self.x = x\n"),
        False,
    ),
    (
        _src("class A:\n    pass\n"),
        _src("@dataclass\nclass A:\n    pass\n"),
        False,
    ),
    (
        _src("class A:\n    def __init__(self, x):\n        self.x = x*2\n"),
        _src("@dataclass\nclass A:\n    x: int\n"),
        # Derived state → labeler would reject, acceptance also rejects (no plain self.x=x).
        False,
    ),
    (_src("def f():\n    return 1\n"), _src("def f():\n    return 1\n"), False),
    (
        _src("class A:\n    def __init__(self, x):\n        self.x = x\n"),
        _src(
            "class A:\n"
            "    def __init__(self, x):\n"
            "        self.x = x\n"
            "    def m(self):\n"
            "        return self.x\n"
        ),
        False,
    ),
    (
        _src("class A:\n    def __init__(self, x):\n        self.x = x\n"),
        _src("@dataclass\nclass A:\n    x: int\n    def __init__(self, x):\n        self.x = x\n"),
        False,
    ),
    (_src("class A:\n    pass\n"), _src("class A:\n    pass\n"), False),
    (
        _src("class A:\n    pass\nclass B:\n    pass\n"),
        _src("class A:\n    pass\nclass B:\n    pass\n"),
        False,
    ),
    (
        _src("def f(x):\n    return x\n"),
        _src("@decorator\ndef f(x):\n    return x\n"),
        False,
    ),
    (
        _src("class A:\n    def __init__(self):\n        pass\n"),
        _src("@dataclass\nclass A:\n    pass\n"),
        False,
    ),
]

# ============================ exception-handling-cleanup ====================

EXCEPTION_HANDLING_CLEANUP = [
    # ---- Positive --------------------------------------------------------
    (
        _src("try:\n    f()\nexcept:\n    pass\n"),
        _src("try:\n    f()\nexcept Exception:\n    raise\n"),
        True,
    ),
    (
        _src("try:\n    f()\nexcept:\n    log()\n"),
        _src("try:\n    f()\nexcept Exception:\n    log()\n"),
        True,
    ),
    (
        _src("try:\n    f()\nexcept ValueError:\n    pass\n"),
        _src("try:\n    f()\nexcept ValueError:\n    raise\n"),
        True,
    ),
    (
        _src("try:\n    f()\nexcept Exception:\n    pass\n"),
        _src("with contextlib.suppress(Exception):\n    f()\n"),
        True,
    ),
    (
        _src("try:\n    g()\nexcept KeyError:\n    pass\n"),
        _src("with contextlib.suppress(KeyError):\n    g()\n"),
        True,
    ),
    (
        _src("def w():\n    try:\n        f()\n    except:\n        pass\n"),
        _src(
            "def w():\n"
            "    try:\n"
            "        f()\n"
            "    except Exception:\n"
            "        log.exception('boom')\n"
        ),
        True,
    ),
    (
        _src("def w():\n    try:\n        f()\n    except:\n        pass\n"),
        _src("def w():\n    try:\n        f()\n    except KeyError:\n        raise\n"),
        True,
    ),
    (
        _src("try:\n    a()\nexcept:\n    pass\n"),
        _src("try:\n    a()\nexcept Exception as e:\n    raise e\n"),
        True,
    ),
    (
        _src("try:\n    b()\nexcept BaseException:\n    pass\n"),
        _src("try:\n    b()\nexcept BaseException:\n    raise\n"),
        True,
    ),
    (
        _src("try:\n    h()\nexcept:\n    pass\n"),
        _src("with contextlib.suppress(Exception):\n    h()\n"),
        True,
    ),
    # ---- Negative --------------------------------------------------------
    (_src("def f():\n    return 1\n"), _src("def f():\n    return 1\n"), False),
    (
        _src("try:\n    f()\nexcept Exception:\n    raise\n"),
        _src("try:\n    f()\nexcept Exception:\n    raise\n"),
        False,
    ),
    (
        _src("try:\n    f()\nexcept Exception:\n    log()\n"),
        _src("try:\n    f()\nexcept Exception:\n    log()\n"),
        False,
    ),
    (
        _src("try:\n    f()\nexcept Exception:\n    raise\n"),
        _src("try:\n    f()\nexcept:\n    pass\n"),
        False,
    ),  # Regression — got worse.
    (_src("def w():\n    return 1\n"), _src("def w():\n    return 2\n"), False),
    (_src("class A:\n    pass\n"), _src("class A:\n    pass\n"), False),
    (_src("import os\n"), _src("import os\n"), False),
    (
        _src("try:\n    f()\nexcept Exception:\n    log()\n"),
        _src("try:\n    g()\nexcept Exception:\n    log()\n"),
        False,
    ),
    (_src("def x():\n    pass\n"), _src("def x():\n    pass\n"), False),
    (_src("# nothing\n"), _src("# nothing\n"), False),
]

# ============================ loop-to-vectorized ============================

LOOP_TO_VECTORIZED = [
    # ---- Positive: a loop in before is gone, vectorized API in after -------
    (_src("for x in df['c']:\n    print(x)\n"), _src("df['c'].apply(print)\n"), True),
    (_src("for x in s:\n    out.append(x*2)\n"), _src("out = s.apply(lambda x: x*2)\n"), True),
    (_src("for v in arr:\n    print(v)\n"), _src("arr.map(print)\n"), True),
    (
        _src("for x in df['a']:\n    r.append(x+1)\n"),
        _src("r = df['a'].apply(lambda x: x+1)\n"),
        True,
    ),
    (
        _src("for x in xs:\n    ys.append(np.sqrt(x))\n"),
        _src("ys = np.where(xs > 0, np.sqrt(xs), 0)\n"),
        True,
    ),
    (_src("for v in s:\n    r.append(v.strip())\n"), _src("r = s.map(str.strip)\n"), True),
    (
        _src("for x in series:\n    out.append(x.lower())\n"),
        _src("out = series.apply(str.lower)\n"),
        True,
    ),
    (
        _src("for x in df['c']:\n    if x > 0:\n        out.append(x)\n"),
        _src("out = df['c'].where(df['c'] > 0)\n"),
        True,
    ),
    (_src("for v in a:\n    b.append(v + 1)\n"), _src("b = a.apply(lambda v: v + 1)\n"), True),
    (
        _src("for x in col:\n    r.append(x ** 2)\n"),
        _src("r = col.apply(lambda x: x ** 2)\n"),
        True,
    ),
    # ---- Negative -------------------------------------------------------
    (_src("for x in xs:\n    pass\n"), _src("for x in xs:\n    pass\n"), False),
    (_src("for x in xs:\n    r.append(x)\n"), _src("for x in xs:\n    r.append(x)\n"), False),
    (_src("def f():\n    return 1\n"), _src("def f():\n    return 1\n"), False),
    (_src("for x in xs:\n    print(x)\n"), _src("xs.foo()\n"), False),  # no vectorized API tokens
    (_src("xs = []\n"), _src("xs.apply(f)\n"), False),  # no loop disappeared
    (
        _src("for a in b:\n    pass\n"),
        _src("for a in b:\n    pass\nxs.apply(f)\n"),
        False,
    ),  # loop preserved
    (_src("import os\n"), _src("import os\n"), False),
    (
        _src("for x in xs:\n    pass\nfor y in ys:\n    pass\n"),
        _src("for y in ys:\n    pass\n"),
        False,
    ),  # gone-loop without vectorized
    (_src("# nothing\n"), _src("# nothing\n"), False),
    (_src("class A:\n    pass\n"), _src("class A:\n    pass\n"), False),
]

# ============================ argument-defaulting ===========================

ARGUMENT_DEFAULTING = [
    # ---- Positive --------------------------------------------------------
    (_src("def f(a, b):\n    return a + b\n"), _src("def f(a, b=0):\n    return a + b\n"), True),
    (_src("def f(x, y):\n    return x*y\n"), _src("def f(x, y=1):\n    return x*y\n"), True),
    (
        _src("def f(s):\n    return s.lower()\n"),
        _src("def f(s, lowercase=True):\n    return s.lower() if lowercase else s\n"),
        True,
    ),
    (
        _src("def g(a, b, c):\n    return a + b + c\n"),
        _src("def g(a, b, c=0):\n    return a + b + c\n"),
        True,
    ),
    (
        _src("def h(*args, key):\n    return key\n"),
        _src("def h(*args, key=None):\n    return key\n"),
        True,
    ),
    (_src("def k(x, **kw):\n    return x\n"), _src("def k(x, y=42, **kw):\n    return x\n"), True),
    (
        _src("def p(a, /, b):\n    return a + b\n"),
        _src("def p(a, /, b=0):\n    return a + b\n"),
        True,
    ),
    (
        _src("def q(a, b, c, d):\n    return a\n"),
        _src("def q(a, b, c, d=5):\n    return a\n"),
        True,
    ),
    (_src("def t(s, n):\n    return s * n\n"), _src("def t(s, n=1):\n    return s * n\n"), True),
    (
        _src("def u(items, sep):\n    return sep.join(items)\n"),
        _src("def u(items, sep=','):\n    return sep.join(items)\n"),
        True,
    ),
    # ---- Negative --------------------------------------------------------
    (_src("def f(a, b):\n    return a + b\n"), _src("def f(a, b):\n    return a + b\n"), False),
    (_src("def f(a, b=0):\n    return a + b\n"), _src("def f(a, b=0):\n    return a + b\n"), False),
    (
        _src("def f(a, b):\n    return a + b\n"),
        _src("def f(a, b=[]):\n    return a + b\n"),
        False,
    ),  # mutable default → reject.
    (_src("def f(a, b):\n    return a + b\n"), _src("def f(a, b={}):\n    return a + b\n"), False),
    (
        _src("def f(a, b):\n    return a + b\n"),
        _src("def f(a, b):\n    return b + a\n"),
        False,
    ),  # body changed but no defaults gained.
    (_src("def f(a):\n    return a\n"), _src("def g(b):\n    return b\n"), False),
    (_src("class A:\n    pass\n"), _src("class A:\n    pass\n"), False),
    (_src("import os\n"), _src("import os\n"), False),
    (
        _src("def f(a, b):\n    return a\n"),
        _src("def f(a, b, c):\n    return a\n"),
        False,
    ),  # new param, not a default.
    (_src("def f(a):\n    return a + 1\n"), _src("def f(a):\n    return a + 2\n"), False),
]

# ============================ none-typing-modernization =====================

NONE_TYPING = [
    # ---- Positive --------------------------------------------------------
    (
        _src("def f(x: Optional[int]) -> int:\n    return x or 0\n"),
        _src("def f(x: int | None) -> int:\n    return x or 0\n"),
        True,
    ),
    (
        _src("def g() -> Optional[str]:\n    return None\n"),
        _src("def g() -> str | None:\n    return None\n"),
        True,
    ),
    (_src("x: Union[int, str] = 1\n"), _src("x: int | str = 1\n"), True),
    (
        _src("def k(xs: typing.List[int]) -> int:\n    return sum(xs)\n"),
        _src("def k(xs: list[int]) -> int:\n    return sum(xs)\n"),
        True,
    ),
    (
        _src("def m(d: typing.Dict[str, int]) -> int:\n    return len(d)\n"),
        _src("def m(d: dict[str, int]) -> int:\n    return len(d)\n"),
        True,
    ),
    (
        _src("def n(s: typing.Set[int]) -> int:\n    return len(s)\n"),
        _src("def n(s: set[int]) -> int:\n    return len(s)\n"),
        True,
    ),
    (
        _src("def p(t: typing.Tuple[int, int]) -> int:\n    return t[0]\n"),
        _src("def p(t: tuple[int, int]) -> int:\n    return t[0]\n"),
        True,
    ),
    (_src("v: Optional[Optional[int]] = None\n"), _src("v: int | None | None = None\n"), True),
    (
        _src("def q(x: Union[int, None]) -> int:\n    return x or 0\n"),
        _src("def q(x: int | None) -> int:\n    return x or 0\n"),
        True,
    ),
    (
        _src("def r(x: Optional[List[int]]) -> int:\n    return sum(x or [])\n"),
        _src("def r(x: list[int] | None) -> int:\n    return sum(x or [])\n"),
        True,
    ),
    # ---- Negative --------------------------------------------------------
    (
        _src("def f(x: int) -> int:\n    return x\n"),
        _src("def f(x: int) -> int:\n    return x\n"),
        False,
    ),
    (
        _src("def f(x: Optional[int]) -> int:\n    return x or 0\n"),
        _src("def f(x: Optional[int]) -> int:\n    return x or 0\n"),
        False,
    ),  # unchanged
    (
        _src("def f(x: int) -> int:\n    return x\n"),
        _src("def f(x: Optional[int]) -> int:\n    return x or 0\n"),
        False,
    ),  # introduced legacy
    (
        _src("def f(x: int | None) -> int:\n    return x or 0\n"),
        _src("def f(x: Optional[int]) -> int:\n    return x or 0\n"),
        False,
    ),  # reverse
    (
        _src("def f(x: list[int]) -> int:\n    return len(x)\n"),
        _src("def f(x: list[int]) -> int:\n    return len(x)\n"),
        False,
    ),
    (
        _src("def f() -> Optional[int]:\n    return None\n"),
        _src("def f() -> Optional[int]:\n    return None\n"),
        False,
    ),  # no rewrite
    (_src("import os\n"), _src("import os\n"), False),
    (_src("class A:\n    pass\n"), _src("class A:\n    pass\n"), False),
    (
        _src("def f(x: int) -> int:\n    return x*2\n"),
        _src("def f(x: int) -> int:\n    return x*3\n"),
        False,
    ),
    (
        _src("def f(x: Optional[int]) -> int:\n    return x or 0\n"),
        _src("def f(x: Optional[int]) -> int:\n    return x or 1\n"),
        False,
    ),
]


_PARAMS = {
    "extract-helper": EXTRACT_HELPER,
    "inline-helper": INLINE_HELPER,
    "comprehension-rewrite": COMPREHENSION_REWRITE,
    "dataclass-migration": DATACLASS_MIGRATION,
    "exception-handling-cleanup": EXCEPTION_HANDLING_CLEANUP,
    "loop-to-vectorized": LOOP_TO_VECTORIZED,
    "argument-defaulting": ARGUMENT_DEFAULTING,
    "none-typing-modernization": NONE_TYPING,
}


@pytest.mark.parametrize(
    "intent, before, after, expected",
    [(intent, b, a, e) for intent, cases in _PARAMS.items() for b, a, e in cases],
)
def test_acceptance_rule(intent: str, before: str, after: str, expected: bool) -> None:
    assert acceptance_check(intent, parse(before), parse(after)) is expected


def test_intent_none_always_accepts() -> None:
    assert acceptance_check("NONE", parse("x = 1\n"), parse("y = 2\n")) is True


def test_unknown_intent_raises() -> None:
    from codingjepa.errors import UsageError

    with pytest.raises(UsageError):
        acceptance_check("not-an-intent", parse(""), parse(""))


def test_each_intent_has_10_positive_and_10_negative() -> None:
    for intent, cases in _PARAMS.items():
        pos = sum(1 for _, _, e in cases if e is True)
        neg = sum(1 for _, _, e in cases if e is False)
        assert pos >= 10, f"{intent}: only {pos} positive cases"
        assert neg >= 10, f"{intent}: only {neg} negative cases"
