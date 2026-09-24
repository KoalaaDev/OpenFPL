// Shared helpers: images, EV maths, squad legality, CSV.

// POSITIONS is the ORDER (an array, to map over); POS_ORDER is the RANK of
// each (an object, to sort by). Conflating them cost a blank screen: the chip
// advisor called POS_ORDER.map, which threw and took the whole app down.
export const POSITIONS = ['GK', 'DEF', 'MID', 'FWD']
export const POS_ORDER = { GK: 0, DEF: 1, MID: 2, FWD: 3 }
export const POS_LABEL = { GK: 'GKP', DEF: 'DEF', MID: 'MID', FWD: 'FWD' }
export const CHIP_SHORT = {
  wildcard: 'WC', freehit: 'FH', bench_boost: 'BB', triple_captain: 'TC',
}
export const CHIP_LONG = {
  wildcard: 'Wildcard Played', freehit: 'Free Hit Played',
  bench_boost: 'Bench Boost Played', triple_captain: 'Triple Captain Played',
}

export const CHIP_NAME = {
  wildcard: 'Wildcard', freehit: 'Free Hit',
  bench_boost: 'Bench Boost', triple_captain: 'Triple Captain',
}
export const ALL_CHIPS = ['wildcard', 'freehit', 'bench_boost', 'triple_captain']

// What the entry can still do with each chip, over a given run of gameweeks.
// `chipState` is /api/entry's `chips` block: the season's chip windows (two of
// each since 2024-25, one per half) with the gameweek each was spent in, plus
// the one activated for the coming deadline — which only an authenticated
// my-team import can see, because the public API freezes at the last deadline.
//
// With no chip state at all (offline, or FPL changed the payload) everything
// reads as usable: the planner falling back to "you tell me" is the old
// behaviour, and is much better than silently refusing a chip you do hold.
export function chipAvailability(chipState, gws = []) {
  const windows = chipState?.windows || []
  const known = windows.length > 0
  const active = chipState?.active || null
  const out = {}
  for (const c of ALL_CHIPS) {
    const mine = windows.filter((w) => w.chip === c)
    const open = mine.filter((w) => w.used_gw == null)
    const inRange = gws.length
      ? open.filter((w) => gws.some((g) => g >= w.start && g <= w.stop))
      : open
    out[c] = {
      known,
      active: active === c,
      held: !known || open.length > 0,
      usable: !known || active === c || inRange.length > 0,
      played: mine.filter((w) => w.used_gw != null).map((w) => w.used_gw),
      windows: open.map((w) => [w.start, w.stop]),
    }
  }
  return out
}

// One line of English for a chip's status — the tooltip that stops anyone
// wondering why a button is greyed out.
export function chipNote(a, name) {
  if (!a) return name
  if (a.active) return `${name} is ACTIVE for the coming gameweek`
  if (a.played.length && !a.held) return `${name} already played (GW${a.played.join(', GW')})`
  if (!a.usable) {
    const w = a.windows[0]
    return w ? `${name} is next available in GW${w[0]}–${w[1]}`
      : `${name} already played`
  }
  if (a.played.length) return `${name} held (the GW${a.played.join(', GW')} one is spent)`
  return `${name} available`
}

// Served from the local disk cache (app/images.py), not the Premier League
// CDN. A pitch is 15 shirts and the projections table is a badge per row; as
// cross-origin requests on a cold cache that was seconds of blank boxes.
// 1x1 transparent gif: a player whose club could not be resolved should show
// nothing, not fire a request for `.../shirt/undefined`
const BLANK = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'
const ok = (c) => c != null && c !== '' && Number.isFinite(Number(c))

export const shirtUrl = (teamCode, isGk = false) =>
  (ok(teamCode) ? `/api/img/${isGk ? 'shirt_gk' : 'shirt'}/${Number(teamCode)}` : BLANK)

export const badgeUrl = (teamCode) =>
  (ok(teamCode) ? `/api/img/badge/${Number(teamCode)}` : BLANK)

// player cut-out, keyed by player.code (NOT player_id)
export const photoUrl = (playerCode) =>
  (ok(playerCode) ? `/api/img/photo/${Number(playerCode)}` : null)

export const fmt1 = (x) => (x == null || Number.isNaN(x) ? '–' : Number(x).toFixed(1))
export const money = (x) => (x == null ? '–' : `£${Number(x).toFixed(1)}m`)

// Continuous fixture-difficulty colour: 1 (easy, green) -> 3 (neutral grey)
// -> 5 (hard, deep red). Returns {bg, fg} for a cell.
export function fdrColor(v) {
  if (v == null || Number.isNaN(v)) return { bg: 'var(--panel-2)', fg: 'var(--muted-2)' }
  const t = Math.max(1, Math.min(5, Number(v)))
  const lerp = (a, b, u) => a.map((x, i) => Math.round(x + (b[i] - x) * u))
  const green = [39, 160, 90], grey = [75, 75, 104], red = [139, 23, 50]
  const c = t <= 3 ? lerp(green, grey, (t - 1) / 2) : lerp(grey, red, (t - 3) / 2)
  return { bg: `rgb(${c.join(',')})`, fg: t <= 1.9 ? '#06301a' : '#f2f2fc' }
}

// Availability % from FPL status + chance_of_playing flags.
export function availPct(p) {
  if (!p) return 100
  if (p.status == null || p.status === 'a') return p.chance ?? 100
  return p.chance ?? 0
}

// Interpolated blue for projection cells: low -> deep panel blue, high -> bright.
export function epColor(v, max = 8) {
  const t = Math.max(0, Math.min(1, (v ?? 0) / max))
  const from = [45, 55, 100]
  const to = [90, 130, 235]
  const c = from.map((f, i) => Math.round(f + (to[i] - f) * t))
  return `rgb(${c[0]},${c[1]},${c[2]})`
}

// ---------------- draft model ----------------
// draft = { id, label, note, source, gws: [gwPlan...] }
// gwPlan = { gw, chip, squad: [{id, sell}], xi: [ids], captain, vice,
//            transfers_in: [ids], transfers_out: [ids], bank, free_after, hits }

// A gameweek only has projections while it is still upcoming: once it kicks
// off the pipeline drops it, so a saved draft that still lists it would show
// 0.0 for every player. Callers use this to say "already played" instead of
// silently rendering zeros.
export function gwHasProj(proj, gw) {
  return !!(proj?.gws && String(gw) in proj.gws)
}

export function epOf(proj, pid, gw) {
  const rec = proj?.players?.[String(pid)]
  return rec ? rec.ep?.[String(gw)] ?? 0 : 0
}

export function gwEV(plan, proj) {
  if (!plan) return 0
  let ev = 0
  for (const pid of plan.xi) ev += epOf(proj, pid, plan.gw)
  const capMult = plan.chip === 'triple_captain' ? 2 : 1
  if (plan.captain) ev += (1 + (capMult - 1)) * epOf(proj, plan.captain, plan.gw)
  if (plan.chip === 'bench_boost') {
    for (const s of plan.squad) {
      if (!plan.xi.includes(s.id)) ev += epOf(proj, s.id, plan.gw)
    }
  }
  ev -= 4 * (plan.hits || 0)
  return ev
}

export function draftTotalEV(draft, proj) {
  return (draft?.gws || []).reduce((a, p) => a + gwEV(p, proj), 0)
}

// A legal XI: 1 GK, >=3 DEF, >=2 MID, >=1 FWD, 11 total.
export function xiLegal(xiIds, posOf) {
  if (xiIds.length !== 11) return false
  const c = { GK: 0, DEF: 0, MID: 0, FWD: 0 }
  for (const id of xiIds) c[posOf(id)]++
  return c.GK === 1 && c.DEF >= 3 && c.DEF <= 5 && c.MID >= 2 && c.MID <= 5 &&
    c.FWD >= 1 && c.FWD <= 3
}

// Best legal XI from a 15-man squad by projected points for a gw:
// 1 GK, then minimum quotas (3 DEF / 2 MID / 1 FWD), then best of the rest
// within maxima (5 DEF / 5 MID / 3 FWD).
export function bestXI(squadIds, posOf, epFor) {
  const byPos = { GK: [], DEF: [], MID: [], FWD: [] }
  for (const id of squadIds) byPos[posOf(id)]?.push(id)
  for (const pos of Object.keys(byPos)) byPos[pos].sort((a, b) => epFor(b) - epFor(a))
  const xi = []
  const take = (pos, n) => { xi.push(...byPos[pos].splice(0, n)) }
  take('GK', 1); take('DEF', 3); take('MID', 2); take('FWD', 1)
  const max = { DEF: 2, MID: 3, FWD: 2 }   // remaining headroom vs maxima
  const rest = [...byPos.DEF.map((id) => ['DEF', id]), ...byPos.MID.map((id) => ['MID', id]),
                ...byPos.FWD.map((id) => ['FWD', id])].sort((a, b) => epFor(b[1]) - epFor(a[1]))
  for (const [pos, id] of rest) {
    if (xi.length >= 11) break
    if (max[pos] > 0) { xi.push(id); max[pos]-- }
  }
  return xi
}

// Best legal XI you could actually field on a one-week chip, within a spend
// cap. The old Free Hit hint compared your XI against the highest-EP eleven in
// the game with no budget and no club limit, so it advertised a gain nobody
// could buy. This respects the formation (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD),
// max 3 per club, and reserves money for the four bench players a legal 15
// still needs. Greedy by points, then repaired by swapping out whichever pick
// loses the fewest points per pound freed.
//
// It is an estimate, not an optimum — the Solver remains the authority.
export function bestAffordableXI(pool, posOf, epFor, priceOf, clubOf, budget) {
  const MIN = { GK: 1, DEF: 3, MID: 2, FWD: 1 }
  const MAX = { GK: 1, DEF: 5, MID: 5, FWD: 3 }
  const byPos = { GK: [], DEF: [], MID: [], FWD: [] }
  for (const id of pool) if (byPos[posOf(id)]) byPos[posOf(id)].push(id)
  for (const k of Object.keys(byPos)) {
    byPos[k] = byPos[k].filter((id) => epFor(id) > 0 || priceOf(id) > 0)
    byPos[k].sort((a, b) => epFor(b) - epFor(a))
  }
  // reserve the cheapest bench that completes a legal 15 (1 GK + 3 outfield)
  const cheapest = (pos, n) => [...byPos[pos]].sort((a, b) => priceOf(a) - priceOf(b)).slice(0, n)
  const bench = [...cheapest('GK', 1), ...cheapest('DEF', 1), ...cheapest('MID', 1), ...cheapest('FWD', 1)]
  const benchCost = bench.reduce((a, id) => a + priceOf(id), 0)
  const cap = Math.max(0, budget - benchCost)

  const picked = []
  const count = { GK: 0, DEF: 0, MID: 0, FWD: 0 }
  const perClub = {}
  const canAdd = (id) => {
    const pos = posOf(id)
    if (count[pos] >= MAX[pos]) return false
    if ((perClub[clubOf(id)] || 0) >= 3) return false
    return true
  }
  const add = (id) => {
    picked.push(id); count[posOf(id)]++
    perClub[clubOf(id)] = (perClub[clubOf(id)] || 0) + 1
  }
  const drop = (id) => {
    picked.splice(picked.indexOf(id), 1); count[posOf(id)]--
    perClub[clubOf(id)]--
  }
  // minimum quotas first, then the best of the rest up to 11
  for (const pos of ['GK', 'DEF', 'MID', 'FWD']) {
    for (const id of byPos[pos]) {
      if (count[pos] >= MIN[pos]) break
      if (canAdd(id)) add(id)
    }
  }
  const rest = [...byPos.DEF, ...byPos.MID, ...byPos.FWD]
    .filter((id) => !picked.includes(id))
    .sort((a, b) => epFor(b) - epFor(a))
  for (const id of rest) {
    if (picked.length >= 11) break
    if (canAdd(id)) add(id)
  }

  const cost = () => picked.reduce((a, id) => a + priceOf(id), 0)
  // repair: swap the pick with the worst points-lost-per-pound-freed
  for (let guard = 0; guard < 60 && cost() > cap; guard++) {
    let best = null
    for (const id of picked) {
      const pos = posOf(id)
      for (const alt of byPos[pos]) {
        if (picked.includes(alt) || priceOf(alt) >= priceOf(id)) continue
        if (alt !== id && (perClub[clubOf(alt)] || 0) >= 3 && clubOf(alt) !== clubOf(id)) continue
        const saved = priceOf(id) - priceOf(alt)
        if (saved <= 0) continue
        const lost = epFor(id) - epFor(alt)
        const ratio = lost / saved
        if (!best || ratio < best.ratio) best = { out: id, in: alt, ratio }
        break     // alternatives are EP-sorted; the first cheaper one is best
      }
    }
    if (!best) break
    drop(best.out); add(best.in)
  }
  return { xi: picked, cost: cost(), affordable: cost() <= cap, benchCost }
}

export function formationRows(xiIds, posOf) {
  const rows = { GK: [], DEF: [], MID: [], FWD: [] }
  for (const id of xiIds) rows[posOf(id)]?.push(id)
  return rows
}

// Convert a solver plan (backend per_gw) into a client draft.
export function planToDraft(plan, meta, label, note) {
  return {
    id: `d${Date.now()}${Math.floor(Math.random() * 1e4)}`,
    label,
    note: note || '',
    source: 'solver',
    entry: meta?.entry_id || null,
    objective: plan.objective,
    // the stock the solve started from, so a hand edit re-runs the same ledger
    ft0: meta?.state?.free_transfers ?? null,
    baseline: null,   // set below: the plan as delivered, for change highlighting
    gws: plan.per_gw.map((g) => ({
      gw: g.gw,
      chip: g.chip,
      squad: g.squad.map((s) => ({ id: s.player_id, sell: s.sell })),
      xi: g.squad.filter((s) => s.in_xi).map((s) => s.player_id),
      captain: g.captain_id,
      vice: g.vice_id,
      transfers_in: g.transfers_in.map((t) => t.player_id),
      transfers_out: g.transfers_out.map((t) => t.player_id),
      bank: g.bank,
      free_after: g.free_after,
      free_used: g.free_used,
      hits: g.hits,
    })),
  }
}

/* The free-transfer ledger for a draft, recomputed from its transfers.

   A draft built from the squad used to stamp the SAME free-transfer count on
   every gameweek and a hard `hits: 0`. So rolling a transfer never accrued,
   and a hand-made plan that ran out of free transfers never took its -4 —
   gwEV subtracts `4 * hits`, so those plans showed inflated totals. Only
   Solver drafts carried real numbers, and even those went stale the moment
   a transfer was edited by hand.

   The rules are the MILP's (optimise/chips.py), so a Solver draft and a hand
   edit agree:
     * each gameweek's moves use free transfers first, the rest are -4 hits
     * whatever is left rolls, +1, capped at 5
     * a Wildcard or Free Hit week spends nothing, costs nothing, and PRESERVES
       the stock — no +1 either ("if you had 2 saved free transfers before
       playing your Wildcard, you will still have 2 the following Gameweek")
     * a 15-player build (pre-season) is free and banks nothing */
export const MAX_FT = 5
export const HIT_COST = 4

/* The whole fifteen, not just the eleven: a Free Hit or a Wildcard buys a
   squad, so the bench has to be bought out of the same money. The XI is the
   affordable-XI solve; the bench is then the cheapest legal completion of
   2/5/5/3 that the leftover budget and the 3-per-club cap allow. */
export function bestAffordableSquad(pool, posOf, epFor, priceOf, clubOf, budget) {
  const NEED = { GK: 2, DEF: 5, MID: 5, FWD: 3 }
  const best = bestAffordableXI(pool, posOf, epFor, priceOf, clubOf, budget)
  const picked = [...best.xi]
  const count = { GK: 0, DEF: 0, MID: 0, FWD: 0 }
  const perClub = {}
  for (const id of picked) {
    count[posOf(id)] = (count[posOf(id)] || 0) + 1
    perClub[clubOf(id)] = (perClub[clubOf(id)] || 0) + 1
  }
  let left = budget - best.cost
  const rest = pool.filter((id) => NEED[posOf(id)] && !picked.includes(id))
    .sort((a, b) => priceOf(a) - priceOf(b) || epFor(b) - epFor(a))
  const bench = []
  for (const pos of ['GK', 'DEF', 'MID', 'FWD']) {
    while (count[pos] < NEED[pos]) {
      const ok = (id) => posOf(id) === pos && !picked.includes(id)
        && (perClub[clubOf(id)] || 0) < 3
      // a player nobody projects is a wasted bench slot, so prefer the
      // cheapest one the model still expects to play
      const pick = rest.find((id) => ok(id) && epFor(id) > 0.5 && priceOf(id) <= left + 1e-9)
        || rest.find((id) => ok(id) && priceOf(id) <= left + 1e-9)
        || rest.find(ok)
      if (!pick) break
      picked.push(pick); bench.push(pick); count[pos]++
      perClub[clubOf(pick)] = (perClub[clubOf(pick)] || 0) + 1
      left = Math.round((left - priceOf(pick)) * 10) / 10
    }
  }
  bench.sort((a, b) => (posOf(a) === 'GK' ? -1 : posOf(b) === 'GK' ? 1 : epFor(b) - epFor(a)))
  const cost = Math.round(picked.reduce((a, id) => a + priceOf(id), 0) * 10) / 10
  return {
    xi: best.xi, bench, squad: picked, cost,
    bank: Math.round((budget - cost) * 10) / 10,
    legal: picked.length === 15 && cost <= budget + 1e-9,
  }
}

/* Setting a chip used to tag the gameweek and change nothing, which for the
   two chips that BUY something reads as the app ignoring you: a Free Hit is a
   different fifteen for one week, a Wildcard a different fifteen from then on.
   Both are now filled in from the same money FPL would give you — the squad's
   selling value plus the bank — and the week each one overwrote is kept, so
   removing or moving the chip puts your own team back.

   Bench Boost and Triple Captain change no players, so they still only tag. */
export const chipSnapshot = (g, by) => ({
  by,
  squad: structuredClone(g.squad), xi: [...g.xi], captain: g.captain, vice: g.vice,
  bank: g.bank, transfers_in: [...(g.transfers_in || [])],
  transfers_out: [...(g.transfers_out || [])], sold: { ...(g.sold || {}) },
})

export function restoreChip(d, chip) {
  for (const g of d.gws) {
    if (g.chip_before?.by !== chip) continue
    const { by, ...was } = g.chip_before          // eslint-disable-line no-unused-vars
    Object.assign(g, structuredClone(was))
    delete g.chip_before
  }
}

export function applyChipToDraft(d, chip, gw, ctx) {
  restoreChip(d, chip)
  for (const p of d.gws) if (p.chip === chip) p.chip = null
  if (gw == null) return d
  const t = d.gws.findIndex((p) => p.gw === gw)
  if (t < 0) return d
  d.gws[t].chip = chip
  if (chip !== 'freehit' && chip !== 'wildcard') return d

  const g0 = d.gws[t]
  const priceOf = (id) => ctx.byId.get(id)?.price ?? 0
  const clubOf = (id) => ctx.byId.get(id)?.team_id ?? 0
  const epAt = (id, w) => epOf(ctx.proj, id, w)
  const budget = g0.squad.reduce((a, s) => a + (s.sell ?? priceOf(s.id)), 0) + (g0.bank || 0)
  const pool = ctx.players.filter((p) => (p.available ?? 1) > 0).map((p) => p.id)
  // a gameweek the model has not projected has no team to pick — filling it
  // would swap a real squad for whoever happens to be cheapest
  if (!pool.some((id) => epAt(id, gw) > 0)) return d
  const sel = bestAffordableSquad(pool, ctx.posOf, (id) => epAt(id, gw),
                                  priceOf, clubOf, budget)
  if (!sel.legal) return d

  const was = chipSnapshot(g0, chip)
  const end = chip === 'freehit' ? t + 1 : d.gws.length
  for (let i = t; i < end; i++) {
    const g = d.gws[i]
    g.chip_before = i === t ? was : chipSnapshot(g, chip)
    g.squad = sel.squad.map((id) => ({ id, sell: priceOf(id) }))
    g.bank = sel.bank
    g.xi = i === t ? [...sel.xi]
      : bestXI(sel.squad, ctx.posOf, (id) => epAt(id, g.gw))
    const sorted = [...g.xi].sort((a, b) => epAt(b, g.gw) - epAt(a, g.gw))
    g.captain = sorted[0] ?? null
    g.vice = sorted.find((id) => id !== g.captain) ?? null
    // the swap is one move of fifteen in the chip week; later Wildcard weeks
    // simply own the new squad, and anything planned off the old one is gone
    g.transfers_in = i === t ? sel.squad.filter((id) => !was.squad.some((x) => x.id === id)) : []
    g.transfers_out = i === t ? was.squad.filter((x) => !sel.squad.includes(x.id)).map((x) => x.id) : []
    g.sold = i === t
      ? Object.fromEntries(g.transfers_out.map((id) => [id, was.squad.find((x) => x.id === id)?.sell ?? priceOf(id)]))
      : {}
  }
  return d
}

/* A gameweek's transfer record is the difference between the squad it starts
   with and the squad it plays — not the moves of whatever action ran last.

   "Best single transfer from here" used to REPLACE the record with its own
   solve's moves while leaving the squad it had already changed. Run it twice
   and the second solve, seeded from the squad the first one bought, would
   often decide the best move is no move — and writing that empty result back
   erased the first transfer. The squad had two new players, the record said
   none, so the ledger charged nothing and the week read as a roll.

   These accumulate the way manual transfers always have: a move whose OUT was
   bought earlier in the same week rewrites that pair (A->B then B->C is A->C),
   anything else is appended, and an action that moves nobody changes nothing. */
export function pairMoves(outs, ins, posOf) {
  const rest = [...ins]
  const pairs = []
  for (const out of outs) {
    // FPL transfers are like-for-like, so pair on position where we can
    let k = posOf ? rest.findIndex((id) => posOf(id) === posOf(out)) : 0
    if (k < 0) k = 0
    const inId = rest.splice(k, 1)[0]
    if (inId != null) pairs.push([out, inId])
  }
  return pairs
}

export function recordMoves(g, pairs, sellOf) {
  g.transfers_in = [...(g.transfers_in || [])]
  g.transfers_out = [...(g.transfers_out || [])]
  g.sold = { ...(g.sold || {}) }
  for (const [out, inId] of pairs) {
    const j = g.transfers_in.indexOf(out)
    if (j >= 0) {
      g.transfers_in[j] = inId              // he was bought this week: rewrite
    } else {
      g.transfers_out.push(out)
      g.transfers_in.push(inId)
      if (sellOf) g.sold[out] = sellOf(out)
    }
  }
  return g
}

/* The same question answered from squads rather than moves, for a week whose
   whole fifteen was replaced (a chip): what changed against what it carried
   in. Pairing is by position so the list reads as transfers. */
export function movesBetween(before, after, posOf) {
  const had = before.map((s) => (typeof s === 'object' ? s.id : s))
  const now = after.map((s) => (typeof s === 'object' ? s.id : s))
  return pairMoves(had.filter((id) => !now.includes(id)),
                   now.filter((id) => !had.includes(id)), posOf)
}

/* Undo history that survives a reload.

   It began as a ref inside the Planner (gone on a tab switch), then a module
   map (gone on refresh). A plan is a document people come back to, so the
   history belongs with it: localStorage, per draft, capped — big enough to
   walk back a Wildcard, small enough that a few drafts cannot fill the quota.
   Every access is wrapped: private windows and blocked site data throw, and a
   convenience must never take the tab down with it. */
export const HISTORY_KEEP = 12          // entries kept per draft
export const HISTORY_BYTES = 400000     // ~0.4 MB per draft, of a ~5 MB quota
const HKEY = (id) => `fplabs.hist.${id}`

export function trimHistory(h, { keep = HISTORY_KEEP, bytes = HISTORY_BYTES } = {}) {
  let out = { undo: (h.undo || []).slice(-keep), redo: (h.redo || []).slice(-keep) }
  // a long plan is a big snapshot; drop the oldest until it fits rather than
  // storing nothing at all
  while (JSON.stringify(out).length > bytes && (out.undo.length || out.redo.length)) {
    if (out.redo.length) out.redo = out.redo.slice(1)
    else out.undo = out.undo.slice(1)
  }
  return out
}

export function loadHistory(id, store) {
  const ls = store || (typeof localStorage === 'undefined' ? null : localStorage)
  try {
    const raw = ls?.getItem(HKEY(id))
    if (raw) {
      const h = JSON.parse(raw)
      if (Array.isArray(h?.undo) && Array.isArray(h?.redo)) return h
    }
  } catch { /* private window, blocked storage: start empty */ }
  return { undo: [], redo: [] }
}

export function saveHistory(id, h, store) {
  const ls = store || (typeof localStorage === 'undefined' ? null : localStorage)
  try {
    ls?.setItem(HKEY(id), JSON.stringify(trimHistory(h)))
  } catch { /* quota or blocked: the plan itself is saved, this is extra */ }
}

/* Drafts are deleted; their history must not sit in storage for ever. */
export function pruneHistory(keepIds, store) {
  const ls = store || (typeof localStorage === 'undefined' ? null : localStorage)
  try {
    const live = new Set((keepIds || []).map((id) => HKEY(id)))
    const dead = []
    for (let i = 0; i < ls.length; i++) {
      const k = ls.key(i)
      if (k && k.startsWith('fplabs.hist.') && !live.has(k)) dead.push(k)
    }
    dead.forEach((k) => ls.removeItem(k))
    return dead.length
  } catch { return 0 }
}

export function applyFtLedger(draft, ft0) {
  if (!draft?.gws?.length) return draft
  const start = Number.isFinite(ft0) ? ft0 : null
  if (start == null) return draft            // nothing to anchor the stock to
  let ft = Math.max(0, Math.min(MAX_FT, start))
  draft.ft0 = start
  for (const g of draft.gws) {
    const n = (g.transfers_in || []).length
    const frozen = g.chip === 'wildcard' || g.chip === 'freehit'
    const build = n === 15
    g.ft_available = ft
    if (frozen || build) {
      g.free_used = 0
      g.hits = 0
      g.free_after = ft
      // a chip week preserves the stock; a build banks nothing past the one
      // free transfer the next gameweek grants
      ft = build ? 1 : ft
      continue
    }
    const used = Math.min(n, ft)
    g.free_used = used
    g.hits = Math.max(0, n - ft)
    g.free_after = ft - used
    ft = Math.min(MAX_FT, g.free_after + 1)
  }
  return draft
}

/* Where a draft's stock starts. A stored `ft0` wins; a draft saved before the
   ledger existed recovers it from its first gameweek; a draft built from the
   squad for the CURRENT gameweek always trusts the live entry, because the
   number it was built with may itself have been wrong (the estimator used to
   grant an extra free transfer after a Wildcard). */
export function draftFt0(draft, entry, editableGw) {
  const g0 = draft?.gws?.[0]
  if (draft?.source === 'entry' && entry?.free_transfers != null
      && g0 && editableGw != null && g0.gw === editableGw) {
    return entry.free_transfers
  }
  if (Number.isFinite(draft?.ft0)) return draft.ft0
  if (g0 && Number.isFinite(g0.free_after)) return g0.free_after + (g0.free_used || 0)
  return entry?.free_transfers ?? null
}

// Snapshot a draft's gws as its baseline (what the deltas are measured against).
export function withBaseline(draft) {
  return { ...draft, baseline: structuredClone(draft.gws) }
}

// Per-gw EV delta of a draft vs its baseline (null when no baseline).
export function baselineDeltas(draft, proj) {
  if (!draft?.baseline) return null
  return draft.gws.map((p, i) => {
    const b = draft.baseline[i]
    return b ? gwEV(p, proj) - gwEV(b, proj) : 0
  })
}

// Moves vs baseline for one gw: [{out, in}] (ids), ignoring a full build.
export function movesVsBaseline(plan, basePlan) {
  if (!plan || !basePlan) return []
  const before = new Set(basePlan.squad.map((s) => s.id))
  const after = new Set(plan.squad.map((s) => s.id))
  const outs = [...before].filter((id) => !after.has(id))
  const ins = [...after].filter((id) => !before.has(id))
  return { outs, ins }
}

export function downloadCSV(filename, header, rows) {
  const esc = (v) => {
    const s = String(v ?? '')
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
  }
  const text = [header, ...rows].map((r) => r.map(esc).join(',')).join('\n')
  const blob = new Blob([text], { type: 'text/csv;charset=utf-8' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = filename
  a.click()
  URL.revokeObjectURL(a.href)
}
