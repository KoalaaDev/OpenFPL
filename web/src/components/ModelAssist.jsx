import React, { useMemo, useState } from 'react'
import { api, pollJob } from '../api'
import { useStore } from '../store'
import { bestXI, epOf, fmt1 } from '../util'

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
  draft, gwIdx, plan, posOf, updateDraft, setToast,
}) {
  const { byId, proj, status } = useStore()
  const [busy, setBusy] = useState(null)
  const [note, setNote] = useState(null)

  const epFor = (id) => epOf(proj, id, plan?.gw)

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
      solve_from: plan.gw,
      horizon,
      initial_squad: seed,
      bank: plan.bank || 0,
      free_transfers: kind === 'transfer' ? 1 : 5,
      n_plans: 1,
      time_limit: 45,
      max_transfers: kind === 'transfer' ? 1 : 3,
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
          if (kind === 'freehit' || kind === 'wildcard') g.chip = kind
          if (kind === 'transfer') {
            g.transfers_in = per.transfers_in.map((r) => r.player_id)
            g.transfers_out = per.transfers_out.map((r) => r.player_id)
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
    setToast({ kind: 'ok', msg: `Applied the model's ${label} for GW${plan.gw}.` })
  }

  const chip = plan.chip
  const nextGw = status?.next_gw
  const isPast = nextGw != null && plan.gw < nextGw

  return (
    <div className="assist">
      <div className="assist-head">
        <span className="section-label">Model assist — GW{plan.gw}</span>
        {xiGap && (
          <span className="assist-gap">
            your XI <b>{fmt1(xiGap.mine)}</b>
            {xiGap.gain > 0.05 ? (
              <>
                {' · '}best legal XI <b>{fmt1(xiGap.top)}</b>
                <span className="up"> +{fmt1(xiGap.gain)}</span>
              </>
            ) : <span className="ok"> · already optimal</span>}
          </span>
        )}
      </div>

      <div className="assist-row">
        <button className={`pill-btn ${xiGap?.gain > 0.05 ? 'accent' : ''}`}
          onClick={optimiseXI} disabled={isPast}
          title="Pick the highest-projected legal XI and captain from this squad">
          ⚡ Optimise XI
        </button>

        <button className="pill-btn" disabled={!!busy || isPast}
          onClick={() => runSolve('transfer')}
          title="The single best transfer from this exact position">
          {busy === 'transfer' ? <span className="spinner" /> : '↔'} Best transfer
        </button>

        <button className={`pill-btn ${chip === 'freehit' ? 'accent' : ''}`}
          disabled={!!busy || isPast}
          onClick={() => runSolve('freehit')}
          title="Build the best possible one-week squad for this gameweek">
          {busy === 'freehit' ? <span className="spinner" /> : '🃏'} Best Free Hit
        </button>

        <button className={`pill-btn ${chip === 'wildcard' ? 'accent' : ''}`}
          disabled={!!busy || isPast}
          onClick={() => runSolve('wildcard')}
          title="Rebuild the squad from this gameweek onward">
          {busy === 'wildcard' ? <span className="spinner" /> : '♻'} Best Wildcard
        </button>
      </div>

      {note && <div className="assist-note"><span className="spinner" /> {note}</div>}
      {isPast && (
        <div className="assist-note">
          GW{plan.gw} has been played — there is nothing left to optimise.
        </div>
      )}
      {!isPast && !note && (
        <p className="assist-hint">
          Free Hit solves this gameweek alone and does not carry forward; Wildcard
          rebuilds and does. Both start from the squad this draft reaches here, so
          they work mid-plan, not just in GW{nextGw ?? '?'}.
        </p>
      )}
    </div>
  )
}
