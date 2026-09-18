"""A gameweek's transfer record accumulates; it is not the last action's diff.

"Best single transfer from here" wrote its own solve's moves over the week's
record while leaving the squad it had already changed. Clicking it twice, the
second solve — seeded from the squad the first one bought — often found no move
worth making, and writing that empty result back erased the first transfer: two
new players in the squad, no transfers on the record, no free transfer spent
and no hit charged, so the week read as a roll.
"""
import json
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
UTIL = (ROOT / "web" / "src" / "util.js").as_uri()
node = shutil.which("node")
pytestmark = pytest.mark.skipif(node is None, reason="node is not installed")

HARNESS = """
import {{ applyFtLedger, movesBetween, pairMoves, recordMoves }} from {util}

const POS = {{ 1: 'DEF', 2: 'DEF', 3: 'DEF', 4: 'MID', 5: 'MID', 6: 'MID', 7: 'FWD', 8: 'FWD' }}
const posOf = (id) => POS[id]
const week = () => ({{ gw: 7, chip: null, transfers_in: [], transfers_out: [], sold: {{}} }})
const sellOf = (id) => id / 10
const out = {{}}

// one solve: A(1) out, B(2) in
const g = week()
recordMoves(g, pairMoves([1], [2], posOf), sellOf)
out.first = {{ ins: g.transfers_in, outs: g.transfers_out, sold: g.sold }}

// a second solve that finds nothing must not erase it
recordMoves(g, pairMoves([], [], posOf), sellOf)
out.after_empty = {{ ins: g.transfers_in, outs: g.transfers_out }}

// a second, different move is appended
recordMoves(g, pairMoves([4], [5], posOf), sellOf)
out.after_second = {{ ins: g.transfers_in, outs: g.transfers_out }}

// selling someone bought THIS week rewrites that pair: 1->2 then 2->3 is 1->3
const h = week()
recordMoves(h, pairMoves([1], [2], posOf), sellOf)
recordMoves(h, pairMoves([2], [3], posOf), sellOf)
out.rewritten = {{ ins: h.transfers_in, outs: h.transfers_out, n: h.transfers_in.length }}

// like-for-like pairing, whatever order the solver lists them in
out.paired = pairMoves([1, 4], [5, 2], posOf)

// the ledger charges what the record says: one free, one at -4
const draft = {{ gws: [
  {{ gw: 7, transfers_in: [...g.transfers_in], chip: null }},
  {{ gw: 8, transfers_in: [], chip: null }},
] }}
applyFtLedger(draft, 1)
out.ledger = draft.gws.map((x) => ({{ used: x.free_used, hits: x.hits, after: x.free_after }}))

// a chip week's record is the whole fifteen against what it carried in
const before = [1, 2, 3, 4].map((id) => ({{ id, sell: id / 10 }}))
const after = [1, 5, 6, 7].map((id) => ({{ id }}))
const c = week()
recordMoves(c, movesBetween(before, after, posOf), (id) => before.find((s) => s.id === id)?.sell)
out.chip = {{ ins: c.transfers_in.slice().sort(), outs: c.transfers_out.slice().sort(), sold: c.sold }}

console.log(JSON.stringify(out))
"""


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    f = tmp_path_factory.mktemp("moves") / "harness.mjs"
    f.write_text(HARNESS.format(util=json.dumps(UTIL)), encoding="utf-8")
    r = subprocess.run([node, str(f)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_the_first_move_is_recorded_with_its_sell_price(out):
    assert out["first"] == {"ins": [2], "outs": [1], "sold": {"1": 0.1}}


def test_a_solve_that_moves_nobody_leaves_the_record_alone(out):
    assert out["after_empty"] == {"ins": [2], "outs": [1]}


def test_a_second_move_is_added_not_substituted(out):
    assert out["after_second"] == {"ins": [2, 5], "outs": [1, 4]}


def test_selling_someone_bought_this_week_rewrites_that_pair(out):
    assert out["rewritten"] == {"ins": [3], "outs": [1], "n": 1}


def test_moves_pair_like_for_like(out):
    assert out["paired"] == [[1, 2], [4, 5]]


def test_the_ledger_then_charges_two_transfers_on_one_free(out):
    assert out["ledger"][0] == {"used": 1, "hits": 1, "after": 0}
    assert out["ledger"][1]["after"] == 1


def test_a_chip_week_records_the_whole_swap_against_what_it_carried_in(out):
    assert out["chip"]["ins"] == [5, 6, 7] and out["chip"]["outs"] == [2, 3, 4]
    assert out["chip"]["sold"] == {"2": 0.2, "3": 0.3, "4": 0.4}
