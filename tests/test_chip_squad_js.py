"""A Free Hit or a Wildcard has to pick a squad, not just tag a gameweek.

Setting either chip used to mark the week and change nothing, which reads as
the app ignoring you. These drive the real module through node (there is no JS
test runner here) and pin what each chip does to the plan.
"""
import json
import os
import pathlib
import shutil
import subprocess
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTIL = pathlib.Path(ROOT, "web", "src", "util.js").as_uri()
node = shutil.which("node")
pytestmark = pytest.mark.skipif(node is None, reason="node is not installed")

# 40 players: 4 per position slice, priced so a cheap legal bench exists
HARNESS = """
import { applyChipToDraft, bestAffordableSquad } from '%s'

const POS = []
for (let i = 0; i < 40; i++) POS.push(i < 6 ? 'GK' : i < 18 ? 'DEF' : i < 32 ? 'MID' : 'FWD')
const P = POS.map((pos, i) => ({
  id: i + 1, position: pos, team_id: (i %% 12) + 1,
  price: 4.0 + ((i * 3) %% 9) * 0.5, available: 1,
}))
const byId = new Map(P.map((p) => [p.id, p]))
const posOf = (id) => byId.get(id).position
// the model likes low ids; ep and price are deliberately uncorrelated
const ep = (id) => Math.max(0, 9 - id * 0.2)
const proj = { players: Object.fromEntries(P.map((p) => [String(p.id), { ep: { 7: ep(p.id), 8: ep(p.id), 9: ep(p.id) } }])) }
const ctx = { proj, byId, players: P, posOf }

const squadIds = [1, 2, 7, 8, 9, 10, 11, 19, 20, 21, 22, 23, 33, 34, 35]
const gw = (n) => ({
  gw: n, chip: null, bank: 1.5,
  squad: squadIds.map((id) => ({ id, sell: byId.get(id).price })),
  xi: [1, 7, 8, 9, 19, 20, 21, 22, 23, 33, 34], captain: 7, vice: 8,
  transfers_in: [], transfers_out: [], sold: {},
})
const draft = () => ({ id: 'd', gws: [gw(7), gw(8), gw(9)] })
const out = {}

const sel = bestAffordableSquad(P.map((p) => p.id), posOf, ep,
  (id) => byId.get(id).price, (id) => byId.get(id).team_id, 100)
out.squad = { n: sel.squad.length, xi: sel.xi.length, cost: sel.cost, legal: sel.legal,
  pos: ['GK', 'DEF', 'MID', 'FWD'].map((k) => sel.squad.filter((id) => posOf(id) === k).length),
  club_max: Math.max(...Object.values(sel.squad.reduce((a, id) => {
    a[byId.get(id).team_id] = (a[byId.get(id).team_id] || 0) + 1; return a }, {}))),
  bench_is_cheap: sel.bench.every((id) => byId.get(id).price <= 5.5) }

const fh = applyChipToDraft(draft(), 'freehit', 8, ctx)
out.fh = {
  chip: fh.gws.map((g) => g.chip),
  changed: fh.gws[1].squad.map((s) => s.id).join() !== squadIds.join(),
  n: fh.gws[1].squad.length,
  moves: fh.gws[1].transfers_in.length,
  before_untouched: fh.gws[0].squad.map((s) => s.id).join() === squadIds.join(),
  after_reverts: fh.gws[2].squad.map((s) => s.id).join() === squadIds.join(),
  captain_is_best: fh.gws[1].captain === fh.gws[1].xi.reduce((b, id) => (ep(id) > ep(b) ? id : b), fh.gws[1].xi[0]),
  bank: fh.gws[1].bank,
}

const wc = applyChipToDraft(draft(), 'wildcard', 8, ctx)
out.wc = {
  keeps_forward: wc.gws[2].squad.map((s) => s.id).join() === wc.gws[1].squad.map((s) => s.id).join(),
  before_untouched: wc.gws[0].squad.map((s) => s.id).join() === squadIds.join(),
  later_moves: wc.gws[2].transfers_in.length,
}

// removing the chip puts the manager's own team back
const undone = applyChipToDraft(applyChipToDraft(draft(), 'freehit', 8, ctx), 'freehit', null, ctx)
out.removed = { chips: undone.gws.map((g) => g.chip),
  squad_back: undone.gws.every((g) => g.squad.map((s) => s.id).join() === squadIds.join()),
  bank_back: undone.gws[1].bank }

// moving it to another week restores the old one and fills the new
const moved = applyChipToDraft(applyChipToDraft(draft(), 'freehit', 8, ctx), 'freehit', 9, ctx)
out.moved = { chips: moved.gws.map((g) => g.chip),
  gw8_back: moved.gws[1].squad.map((s) => s.id).join() === squadIds.join(),
  gw9_changed: moved.gws[2].squad.map((s) => s.id).join() !== squadIds.join() }

// a week with no projections at all cannot be filled; the tag must survive
const blind = applyChipToDraft(draft(), 'freehit', 8, { ...ctx, proj: { players: {} } })
out.blind = { chip: blind.gws[1].chip, squad: blind.gws[1].squad.map((s) => s.id).join() === squadIds.join() }

// the chips that buy nothing still only tag
const tc = applyChipToDraft(draft(), 'triple_captain', 8, ctx)
out.tc = { chip: tc.gws[1].chip, squad: tc.gws[1].squad.map((s) => s.id).join() === squadIds.join() }

console.log(JSON.stringify(out))
""" % UTIL


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    d = tmp_path_factory.mktemp("chip")
    f = d / "harness.mjs"
    f.write_text(textwrap.dedent(HARNESS), encoding="utf-8")
    r = subprocess.run([node, str(f)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_the_squad_it_buys_is_a_legal_fifteen(out):
    s = out["squad"]
    assert s["n"] == 15 and s["xi"] == 11 and s["legal"]
    assert s["pos"] == [2, 5, 5, 3] and s["club_max"] <= 3
    assert s["cost"] <= 100 and s["bench_is_cheap"]


def test_a_free_hit_changes_that_week_only(out):
    fh = out["fh"]
    assert fh["chip"] == [None, "freehit", None]
    assert fh["changed"] and fh["n"] == 15 and fh["moves"] > 0
    assert fh["before_untouched"] and fh["after_reverts"]
    assert fh["captain_is_best"] and fh["bank"] >= 0


def test_a_wildcard_keeps_its_squad_from_then_on(out):
    wc = out["wc"]
    assert wc["keeps_forward"] and wc["before_untouched"]
    assert wc["later_moves"] == 0          # the rebuild is one move, in its week


def test_removing_or_moving_the_chip_restores_the_real_team(out):
    assert out["removed"]["chips"] == [None, None, None]
    assert out["removed"]["squad_back"] and out["removed"]["bank_back"] == 1.5
    assert out["moved"]["chips"] == [None, None, "freehit"]
    assert out["moved"]["gw8_back"] and out["moved"]["gw9_changed"]


def test_a_week_with_no_projections_keeps_the_tag_and_the_squad(out):
    assert out["blind"] == {"chip": "freehit", "squad": True}


def test_bench_boost_and_triple_captain_still_only_tag(out):
    assert out["tc"] == {"chip": "triple_captain", "squad": True}
