"""The planner's free-transfer ledger (web/src/util.js: applyFtLedger).

A draft built from the squad used to stamp the same FT count on every
gameweek and a hard ``hits: 0`` — so rolling never accrued and a hand-made
plan that ran out of free transfers never took its -4, which ``gwEV``
subtracts, so those plans showed inflated totals.

The ledger is JavaScript and the repo has no JS test runner, so this drives
it through ``node`` when node is available and skips otherwise. The expected
numbers are the MILP's rules (optimise/chips.py) and FPL's.
"""
import json
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
UTIL = (ROOT / "web" / "src" / "util.js").as_uri()
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not on PATH")


def ledger(ft0, weeks):
    """weeks: [(n_transfers_in, chip_or_None)] -> per-gw ledger dicts."""
    script = f"""
import {{ applyFtLedger }} from {json.dumps(UTIL)}
const weeks = {json.dumps(weeks)}
const d = applyFtLedger({{ gws: weeks.map(([n, chip], i) => ({{
  gw: i + 1, chip, transfers_in: Array(n).fill(1) }})) }}, {json.dumps(ft0)})
console.log(JSON.stringify(d.gws.map((g) => [g.ft_available, g.free_used, g.hits, g.free_after])))
"""
    out = subprocess.run([NODE, "--input-type=module", "-e", script],
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_rolling_accrues_one_a_week():
    """The visible bug: every gameweek read "2 FT" however long you rolled."""
    assert [g[0] for g in ledger(2, [(0, None)] * 3)] == [2, 3, 4]


def test_moves_beyond_the_stock_are_hits():
    # 3 moves on 2 FT: 2 free, 1 hit, nothing left, then 1 FT next week
    assert ledger(2, [(3, None), (0, None)]) == [[2, 2, 1, 0], [1, 0, 0, 1]]


def test_a_wildcard_preserves_the_stock_without_the_plus_one():
    assert ledger(2, [(9, "wildcard"), (0, None)]) == [[2, 0, 0, 2], [2, 0, 0, 2]]


def test_a_free_hit_preserves_it_too():
    assert ledger(3, [(11, "freehit"), (1, None)]) == [[3, 0, 0, 3], [3, 1, 0, 2]]


def test_bench_boost_and_triple_captain_count_transfers_normally():
    assert ledger(1, [(2, "bench_boost")]) == [[1, 1, 1, 0]]
    assert ledger(1, [(2, "triple_captain")]) == [[1, 1, 1, 0]]


def test_the_stock_caps_at_five():
    assert [g[0] for g in ledger(4, [(0, None)] * 4)] == [4, 5, 5, 5]


def test_a_squad_build_is_free_and_banks_one():
    assert ledger(0, [(15, None), (0, None)]) == [[0, 0, 0, 0], [1, 0, 0, 1]]
