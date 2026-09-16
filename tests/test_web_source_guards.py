"""Source guards for two front-end failure modes that have no unit test.

There is no JS test runner in this repo, and adding one to catch two specific
mistakes would cost more than it saves. These grep for the mistakes instead.

Both are real: `util.POS_ORDER` is an object of RANKS, and calling `.map` on
it threw inside a render, which unmounts the whole React tree — "show XI" on
the chip advisor produced a **blank white screen** with the squad and the
unsaved plan gone from view.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "web", "src")


def jsx_files():
    for base, _dirs, names in os.walk(SRC):
        for n in names:
            if n.endswith((".jsx", ".js")):
                yield os.path.join(base, n)


def read(p):
    with open(p, encoding="utf-8") as fh:
        return fh.read()


def code(p):
    """The file with comments stripped. A guard that trips on the comment
    explaining the bug it guards against is worse than no guard at all."""
    t = re.sub(r"/\*.*?\*/", "", read(p), flags=re.S)
    return re.sub(r"(?m)//.*$", "", t)


ARRAY_USE = re.compile(r"POS_ORDER\s*\.\s*(map|filter|forEach|slice|some|every)\b")
IMPORTS_IT = re.compile(r"import\s*\{[^}]*\bPOS_ORDER\b[^}]*\}\s*from", re.S)
DEFINES_OWN = re.compile(r"const\s+POS_ORDER\s*=\s*\[")


def test_util_pos_order_is_never_used_as_an_array():
    """`util.POS_ORDER` maps position -> RANK, for sorting; `POSITIONS` is the
    array to map over. A module that defines its OWN local POS_ORDER array is
    fine — only the imported one is the trap."""
    bad = []
    for p in jsx_files():
        src = code(p)
        if IMPORTS_IT.search(src) and not DEFINES_OWN.search(src) and ARRAY_USE.search(src):
            bad.append(os.path.relpath(p, ROOT))
    assert not bad, f"util's POS_ORDER used as an array in: {bad}"


def test_the_guard_can_actually_see_the_bug():
    """A grep test that cannot fail is decoration. This is the shape the chip
    advisor had, and the guard must reject it."""
    broken = ("import { POS_ORDER } from '../util'\n"
              "const X = () => POS_ORDER.map((p) => p)\n")
    assert IMPORTS_IT.search(broken) and ARRAY_USE.search(broken)
    assert not DEFINES_OWN.search(broken)


def test_positions_and_pos_order_are_both_exported():
    util = read(os.path.join(SRC, "util.js"))
    assert "export const POSITIONS = ['GK', 'DEF', 'MID', 'FWD']" in util
    assert "export const POS_ORDER = {" in util


def test_every_tab_goes_through_the_error_boundary():
    """A render error in one tab must not blank the app. `Pane` owns that, so
    every tab has to be mounted inside one."""
    app = read(os.path.join(SRC, "App.jsx"))
    assert "<ErrorBoundary where={name}>" in app
    for name in ("Planner", "Projections", "Fixtures", "Prices", "Solver",
                 "MiniLeague", "Live", "Deadline", "Model"):
        assert re.search(rf"<Pane[^>]*>\s*<{name}\b", app, re.S), name


def test_the_admin_model_tab_is_gated_on_both_sides():
    app = read(os.path.join(SRC, "App.jsx"))
    assert "visited.Model && isAdmin" in app
    main = read(os.path.join(ROOT, "app", "main.py"))
    assert "def admin_model(" in main and "require_admin" in main
