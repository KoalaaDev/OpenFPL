import React, { useMemo, useState } from 'react'
import { api, pollJob } from '../api'
import { useStore } from '../store'
import {
  CHIP_NAME, bestXI, chipAvailability, chipNote, epOf, fmt1, movesBetween,
  pairMoves, recordMoves,
} from '../util'

/* The Planner used to be a drawing tool: it let you move players around and
   told you the total afterwards. Everything the model knew lived in the Solver
   tab, which answers one big question ("plan the next five gameweeks") and
   cannot answer the small ones you actually hit while planning — what is the
   best XI for THIS gameweek, what is the best Free Hit squad if I chip here,
   what single transfer is worth most from this exact position.

   This bar puts the model at each of those decision points. Every action is
   seeded from the squad the DRAFT reaches at this gameweek, not from the
   fifteen currently owned, so it still works at GW9 of a plan.

   The XI number is always on screen, computed locally, because the most common
   mistake is not a bad transfer - it is leaving points on the bench. */
export default function ModelAssist({
  draft, gwIdx, plan, posOf, updateDraft, setToast, embedded = false,
}) {
  const { byId, proj, status, entry, entryId } = useStore()
  const [busy, setBusy] = useState(null)
  const [note, setNote] = useState(null)

  const epFor = (id) => epOf(proj, id, plan?.gw)
  // Asking for the best Free Hit you cannot play is a plan you cannot execute:
  // the backend refuses the chip anyway, so say so here rather than hand back
  // a "wildcard" plan that is really just three transfers and four hits.
  const avail = chipAvailability(entry?.chips, plan ? [plan.gw] : [])

  /* --- the always-visible gap: your XI against the best legal one --------- */
  const xiGap = useMemo(() => {
    if (!plan) return null
    const ids = plan.squad.map((s) => s.id)
    const best = bestXI(ids, posOf, epFor)
    const capOf = (xi) => xi.slice().sort((a, b) => epFor(b) - epFor(a))[0]
    const score = (xi, cap) => xi.reduce((a, id) => a + epFor(id), 0) + epFor(cap)
    const mine = score(plan.xi, plan.captain)
    const top = score(best, capOf(best))
    return { mine, top, gain: top - mine, best, cap: capOf(best) }
  }, [plan, proj, byId])   // eslint-disable-line react-hooks/exhaustive-deps

  if (!plan) return null

  const optimiseXI = () => {
    if (!xiGap || xiGap.gain <= 0.001) {
      setToast({ kind: 'ok', msg: 'That is already the best legal XI for this gameweek.' })
      return
    }
    const { best, cap } = xiGap
    updateDraft((d) => {
      const g = d.gws[gwIdx]
      g.xi = best
      g.captain = cap
      const rest = best.filter((id) => id !== cap).sort((a, b) => epFor(b) - epFor(a))
      g.vice = rest[0] ?? null
      return d
    })
    setToast({ kind: 'ok', msg: `XI optimised — +${fmt1(xiGap.gain)} projected points.` })
  }

  /* What this gameweek has left. Asking for "one free transfer" however many
     moves the week already carries makes the solver price a move at zero that
     the ledger will charge -4 for. */
  const freeLeft = Math.max(0, (plan?.ft_available ?? 1) - (plan?.transfers_in?.length ?? 0))

  /* --- server-side solves, seeded from this gameweek's drafted squad ------ */
  const runSolve = async (kind) => {
    setBusy(kind); setNote(null)
    const seed = {}
    for (const s of plan.squad) {
      seed[s.id] = s.sell || byId.get(s.id)?.price || 0
    }
    if (Object.values(seed).some((v) => !v)) {
      setBusy(null)
      setToast({ kind: 'err', msg: 'This draft has a £0.0 selling price — recreate it from your current squad.' })
      return
    }
    // A Free Hit is a one-week squad, so it is solved over this gameweek
    // alone. A Wildcard is permanent, so it is solved over the run that is
    // left in the draft and then carried forward.
    const rest = draft.gws.slice(gwIdx).map((g) => g.gw)
    const horizon = kind === 'freehit' ? 1 : Math.min(rest.length, 5)
    const gws = rest.slice(0, horizon)
    const params = {
      // the entry rides along so the solver can refuse a chip you have
      // already spent; the seeded squad below still overrides its fifteen
      entry: entryId,
      solve_from: plan.gw,
      horizon,
      initial_squad: seed,
      bank: plan.bank || 0,
      free_transfers: kind === 'transfer' ? freeLeft : 5,
      n_plans: 1,
      time_limit: 45,
      // as many moves as the week has banked, not one: with 2 or 3 free
      // transfers the best PAIR is often not the best single move plus the
      // next one, and clicking twice only ever finds the greedy version
      max_transfers: kind === 'transfer' ? Math.min(5, Math.max(1, freeLeft)) : 3,
    }
    if (kind === 'freehit' || kind === 'wildcard') {
      params.chips = { [kind]: { enabled: true, gws: [plan.gw], force: plan.gw } }
      params.chip_reserve = { [kind]: 0 }   // we are asking "play it HERE"
    }
    try {
      const { job_id } = await api.solve(params)
      const res = await pollJob(job_id, (j) => {
        const last = j.progress[j.progress.length - 1]
        if (last) setNote(last.msg)
      })
      const per = res?.plans?.[0]?.per_gw?.[0]
      if (!per) throw new Error('the solver returned no plan')
      applySolved(per, kind)
    } catch (e) {
      setToast({ kind: 'err', msg: `Could not solve: ${e.message}` })
    } finally {
      setBusy(null); setNote(null)
    }
  }

  const applySolved = (per, kind) => {
    const squad = per.squad.map((r) => ({ id: r.player_id, sell: r.price }))
    const xi = per.squad.filter((r) => r.in_xi).map((r) => r.player_id)
    const captain = per.squad.find((r) => r.is_captain)?.player_id ?? null
    const vice = per.squad.find((r) => r.is_vice)?.player_id ?? null
    const sellOf = (id) => plan.squad.find((s) => s.id === id)?.sell ?? byId.get(id)?.price ?? 0
    // A Free Hit reverts, so it touches exactly one gameweek. Anything else
    // changes the squad you carry, so it propagates until the next Free Hit.
    const single = kind === 'freehit'
    updateDraft((d) => {
      const end = single ? gwIdx + 1 : d.gws.length
      for (let i = gwIdx; i < end; i++) {
        const g = d.gws[i]
        if (i > gwIdx && g.chip === 'freehit') break
        g.squad = structuredClone(squad)
        g.bank = per.bank ?? g.bank
        if (i === gwIdx) {
          g.xi = xi; g.captain = captain; g.vice = vice
          if (kind === 'freehit' || kind === 'wildcard') {
            g.chip = kind
            // the chip replaced the fifteen, so the week's moves are what it
            // changed against the squad carried IN — what the chip put aside,
            // or failing that the week before
            const before = g.chip_before?.squad || d.gws[i - 1]?.squad || plan.squad
            g.transfers_in = []; g.transfers_out = []; g.sold = {}
            recordMoves(g, movesBetween(before, squad, posOf),
                        (id) => before.find((x) => x.id === id)?.sell ?? sellOf(id))
          }
          if (kind === 'transfer') {
            // accumulate: a second solve must not erase the first one's move
            recordMoves(g, pairMoves(per.transfers_out.map((r) => r.player_id),
                                     per.transfers_in.map((r) => r.player_id), posOf),
                        sellOf)
          }
        } else {
          // later gameweeks keep their own best XI for their own fixtures
          const ids = g.squad.map((s) => s.id)
          const ef = (id) => epOf(proj, id, g.gw)
          g.xi = bestXI(ids, posOf, ef)
          const sorted = g.xi.slice().sort((a, b) => ef(b) - ef(a))
          g.captain = sorted[0] ?? null
          g.vice = sorted[1] ?? null
        }
      }
      return d
    })
    const label = { freehit: 'Free Hit squad', wildcard: 'Wildcard squad',
                    transfer: 'transfer' }[kind]
    const n = per.transfers_in.length
    setToast({ kind: 'ok', msg: kind !== 'transfer'
      ? `Applied the model's ${label} for GW${plan.gw}.`
      : n === 0
        ? `Nothing is worth transferring from here for GW${plan.gw}`
          + `${freeLeft ? ` — the model would roll ${freeLeft === 1 ? 'the' : 'its'} free `
            + `transfer${freeLeft === 1 ? '' : 's'}.` : ' at -4 — the model would keep this squad.'}`
        : `Applied ${n} transfer${n === 1 ? '' : 's'} for GW${plan.gw}`
          + `${n > freeLeft ? ` — ${n - freeLeft} at -4.` : '.'}` })
  }

  const chip = plan.chip
  const nextGw = status?.editable_gw ?? status?.next_gw
  const isPast = nextGw != null && plan.gw < nextGw

  /* Four identical buttons in a row is a menu, not advice: it asks the reader
     to know which of them applies to their situation. These are the same four
     actions, each stated as what it would do and what it is worth, ordered so
     the one with something to say is at the top. */
  const gain = xiGap && xiGap.gain > 0.05 ? xiGap.gain : 0
  const actions = [
    {
      key: 'xi',
      icon: '⚡',
      title: gain ? `Your XI leaves ${fmt1(gain)} pts on the bench` : 'Your XI is already the best one',
      body: gain
        ? `Best legal XI scores ${fmt1(xiGap.top)} against your ${fmt1(xiGap.mine)}, captain included.`
        : `Nothing in this fifteen beats the eleven you have picked for GW${plan.gw}.`,
      cta: 'Fix my XI',
      value: gain,
      run: optimiseXI,
      disabled: isPast || !gain,
    },
    {
      key: 'transfer',
      icon: '↔',
      title: freeLeft > 1 ? `Best use of your ${freeLeft} free transfers`
        : 'Best single transfer from here',
      body: `Solves the squad this draft reaches at this gameweek — `
        + (freeLeft > 1
          ? `up to ${freeLeft} moves, no hit, and it takes fewer if fewer are worth it.`
          : freeLeft
            ? 'one move, free transfer, no hit.'
            : 'no free transfer left, so it only moves if the gain beats -4.'),
      cta: 'Find it',
      run: () => runSolve('transfer'),
      disabled: !!busy || isPast,
    },
    {
      key: 'freehit',
      icon: '🃏',
      title: chip === 'freehit' ? 'Free Hit is set for this gameweek' : 'Best Free Hit for this gameweek',
      body: avail.freehit?.usable
        ? 'A one-week squad, solved for GW' + plan.gw + ' alone. It does not carry forward.'
        : chipNote(avail.freehit, CHIP_NAME.freehit),
      cta: 'Build it',
      run: () => runSolve('freehit'),
      disabled: !!busy || isPast || !avail.freehit?.usable,
      on: chip === 'freehit',
    },
    {
      key: 'wildcard',
      icon: '♻',
      title: chip === 'wildcard' ? 'Wildcard is set for this gameweek' : 'Best Wildcard from this gameweek',
      body: avail.wildcard?.usable
        ? 'Rebuilds the fifteen from GW' + plan.gw + ' and carries that squad forward.'
        : chipNote(avail.wildcard, CHIP_NAME.wildcard),
      cta: 'Rebuild',
      run: () => runSolve('wildcard'),
      disabled: !!busy || isPast || !avail.wildcard?.usable,
      on: chip === 'wildcard',
    },
  ]
  // the one with points on the table first; the rest keep their order
  actions.sort((a, b) => (b.value || 0) - (a.value || 0))

  const body = (
    <>
      {isPast ? (
        <div className="fold-note">GW{plan.gw} has been played — there is nothing
          left to optimise. Move the plan forward to GW{nextGw}.</div>
      ) : (
        <div className="assist-list">
          {actions.map((a) => (
            <div key={a.key}
              className={`assist-card ${a.value ? 'hot' : ''} ${a.on ? 'on' : ''} ${a.disabled ? 'off' : ''}`}>
              <span className="ac-icon" aria-hidden="true">
                {busy === a.key ? <span className="spinner" /> : a.icon}
              </span>
              <div className="ac-text">
                <div className="ac-title">{a.title}
                  {a.value ? <span className="ac-val">+{fmt1(a.value)}</span> : null}
                </div>
                <div className="ac-body">{a.body}</div>
              </div>
              <button className={`pill-btn ${a.value ? 'accent' : ''}`}
                disabled={a.disabled} onClick={a.run}>{a.cta}</button>
            </div>
          ))}
        </div>
      )}

      {note && <div className="assist-note"><span className="spinner" /> {note}</div>}
    </>
  )

  // inside the Advice panel it supplies only its body; the panel owns the
  // header and the tab strip
  if (embedded) return body
  return (
    <div className="panel assist">
      <div className="panel-head">
        Model assist
        <span className="assist-gw">GW{plan.gw}</span>
      </div>
      {body}
    </div>
  )
}
