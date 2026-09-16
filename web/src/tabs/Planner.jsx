import React, { useEffect, useMemo, useRef, useState } from 'react'
import Flag from '../components/Flag'
import PlayerModal from '../components/PlayerModal'
import ModelAssist from '../components/ModelAssist'
import PitchLines from '../components/Pitch'
import Section from '../components/Section'
import { api, pollJob } from '../api'
import { useFixtureLookup, usePersisted, useStore } from '../store'
import { Radar, VIZ, VIZ_NEUTRAL as VIZ_MUTED } from '../charts'
import { DNA_AXES, dnaOf, dnaRaw, dnaScaled } from '../dna'
import {
  CHIP_LONG, CHIP_NAME, CHIP_SHORT, POSITIONS, applyFtLedger, baselineDeltas, draftFt0,
  bestAffordableXI, bestXI, chipAvailability, chipNote, epOf, fdrColor,
  formationRows, fmt1, gwEV, gwHasProj, money, shirtUrl, withBaseline, xiLegal,
} from '../util'

const DEFAULT_HORIZON = 8    // gameweeks in a new draft

export default function Planner() {
  const { drafts, setDrafts, activeDraftId, setActiveDraftId, proj, byId, players,
          entry, status, setToast, refreshProjections, isAdmin, editableGw } = useStore()
  const draft = drafts.find((d) => d.id === activeDraftId) || drafts[0] || null
  const [gwIdx, setGwIdx] = useState(0)
  const [sel, setSel] = useState(null)          // {pid, mode: 'swap'}
  const [statPid, setStatPid] = useState(null)  // player stats modal
  const [xfer, setXfer] = useState(null)        // pid being transferred out (modal)
  const [armed, setArmed] = useState(null)      // player picked from search, to bring in
  const undoRef = useRef([])                    // previous draft states (this session)
  const [undoN, setUndoN] = useState(0)

  const plan = draft?.gws?.[Math.min(gwIdx, (draft?.gws?.length || 1) - 1)] || null
  // A draft saved before the season still lists played gameweeks; those have no
  // projections (every player would read 0.0), so open the first live one.
  // Open on the first gameweek still ahead of us. This used to key off
  // whether PROJECTIONS existed, so an empty cache (a data pull clears it)
  // dropped you on GW1 - a gameweek already played, where every control is
  // correctly disabled and the whole tab looks broken.
  const firstLiveGw = useMemo(() => {
    const gws = draft?.gws || []
    const next = editableGw
    let i = next != null ? gws.findIndex((g) => g.gw >= next) : -1
    if (i < 0) i = gws.findIndex((g) => gwHasProj(proj, g.gw))
    return i < 0 ? 0 : i
  }, [draft?.id, draft?.gws, proj, editableGw])
  useEffect(() => { setGwIdx(firstLiveGw) }, [draft?.id, firstLiveGw])
  // These are NOT the same thing and conflating them told the user that GW2-5
  // had "already been played" when the projection cache was simply empty
  // (a data pull clears it). Played is decided by the calendar; missing
  // projections are a job you can run.
  // locked once its deadline has passed — a gameweek in progress cannot be
  // changed, whatever the model still projects for it
  const planIsPast = plan && editableGw != null
    ? plan.gw < editableGw : false
  const planUnprojected = plan ? !gwHasProj(proj, plan.gw) : false
  const posOf = (pid) => byId.get(pid)?.position || 'MID'

  // Build exactly the gameweeks this draft spans, so the button next to the
  // warning fixes the thing the warning is about.
  const [building, setBuilding] = useState(false)
  const buildHorizon = async () => {
    if (building || !draft?.gws?.length) return
    setBuilding(true)
    setToast({ kind: 'info', msg: 'Building projections…' })
    try {
      const { job_id } = await api.buildProjections(draft.gws.map((g) => g.gw))
      await pollJob(job_id, (j) => {
        const last = j.progress[j.progress.length - 1]
        if (last) setToast({ kind: 'info', msg: last.msg })
      })
      refreshProjections()
      setToast({ kind: 'ok', msg: 'Projections built.' })
    } catch (e) {
      setToast({ kind: 'err', msg: `Build failed: ${e.message}` })
    } finally {
      setBuilding(false)
    }
  }

  // every edit goes through here: snapshot for undo, then mutate a clone
  // Every edit re-runs the free-transfer ledger, so a transfer, a chip or a
  // rolled week immediately shows the right FT count and any -4 it costs.
  const updateDraft = (fn, { record = true } = {}) => {
    setDrafts((ds) => ds.map((d) => {
      if (d.id !== draft.id) return d
      if (record) {
        undoRef.current.push(structuredClone(d))
        if (undoRef.current.length > 60) undoRef.current.shift()
      }
      const next = fn(structuredClone(d))
      return applyFtLedger(next, draftFt0(next, entry, editableGw))
    }))
    if (record) setUndoN((n) => n + 1)
  }

  // A draft saved before the ledger existed (or built while the estimator was
  // still over-counting after a Wildcard) shows stale numbers until it is
  // edited. Correct it on sight, without an undo step — it is not a change
  // the user made.
  useEffect(() => {
    if (!draft?.gws?.length) return
    const ft0 = draftFt0(draft, entry, editableGw)
    if (ft0 == null) return
    const fixed = applyFtLedger(structuredClone(draft), ft0)
    const stale = fixed.ft0 !== draft.ft0 || fixed.gws.some((g, i) => {
      const o = draft.gws[i]
      return g.free_after !== o.free_after || g.hits !== o.hits || g.free_used !== o.free_used
    })
    if (stale) setDrafts((ds) => ds.map((d) => (d.id === draft.id ? fixed : d)))
  }, [draft?.id, entry?.free_transfers, editableGw, draft?.ft0])   // eslint-disable-line react-hooks/exhaustive-deps
  const undo = () => {
    const prev = undoRef.current.pop()
    if (!prev) return
    setDrafts((ds) => ds.map((d) => (d.id === prev.id ? prev : d)))
    setUndoN((n) => Math.max(0, n - 1))
    setToast({ kind: 'ok', msg: 'Undone.' })
  }
  useEffect(() => {
    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && !e.target.closest('input,textarea')) {
        e.preventDefault(); undo()
      }
      if (e.key === 'Escape') { setArmed(null); setSel(null) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  // A transfer out->in from the current gw forward (Free Hit gws don't
  // propagate). Records the out player's sell price so it can be undone.
  const applyTransfer = (outId, inP) => {
    const cur = draft.gws[gwIdx]
    const out = cur.squad.find((s) => s.id === outId)
    // Drafts created before selling prices were wired through store sell: 0;
    // treat that as unknown and fall back to the current price, never £0.0.
    const outSell = out?.sell || byId.get(outId)?.price || 0
    const budget = (cur.bank || 0) + outSell
    if (posOf(outId) !== inP.position) {
      setToast({ kind: 'err', msg: `${inP.web_name} is a ${inP.position} — replace a ${inP.position}.` })
      return false
    }
    if (cur.squad.some((s) => s.id === inP.id)) {
      setToast({ kind: 'err', msg: `${inP.web_name} is already in the squad.` }); return false
    }
    if (inP.price > budget + 1e-9) {
      setToast({ kind: 'err', msg: `Can't afford ${inP.web_name} (${money(inP.price)} > ${money(budget)} available).` })
      return false
    }
    const club = cur.squad.filter((s) => s.id !== outId && byId.get(s.id)?.team_id === inP.team_id).length
    if (club >= 3) {
      setToast({ kind: 'err', msg: `Already 3 players from ${inP.team_id ? 'that club' : 'the club'}.` }); return false
    }
    updateDraft((d) => {
      const delta = outSell - inP.price
      const end = d.gws[gwIdx].chip === 'freehit' ? gwIdx + 1 : d.gws.length
      for (let t = gwIdx; t < end; t++) {
        const g = d.gws[t]
        if (!g.squad.some((s) => s.id === outId)) break
        g.squad = g.squad.filter((s) => s.id !== outId)
        g.squad.push({ id: inP.id, sell: inP.price })
        if (g.xi.includes(outId)) g.xi = g.xi.map((id) => (id === outId ? inP.id : id))
        if (g.captain === outId) g.captain = inP.id
        if (g.vice === outId) g.vice = inP.id
        g.bank = Math.round(((g.bank || 0) + delta) * 10) / 10
        if (t === gwIdx) {
          g.transfers_out = [...g.transfers_out, outId]
          g.transfers_in = [...g.transfers_in, inP.id]
          g.sold = { ...(g.sold || {}), [outId]: outSell }
        }
      }
      return d
    })
    return true
  }

  // Reverse one manual transfer (identified by the incoming player).
  const undoTransfer = (inId) => {
    const t0 = draft.gws.findIndex((g) => g.transfers_in.includes(inId))
    if (t0 < 0) return
    const g0 = draft.gws[t0]
    const k = g0.transfers_in.indexOf(inId)
    const outId = g0.transfers_out[k]
    const outSell = g0.sold?.[outId] ?? byId.get(outId)?.price ?? 0
    const inPrice = g0.squad.find((s) => s.id === inId)?.sell ?? byId.get(inId)?.price ?? 0
    updateDraft((d) => {
      for (let t = t0; t < d.gws.length; t++) {
        const g = d.gws[t]
        if (!g.squad.some((s) => s.id === inId)) break
        g.squad = g.squad.filter((s) => s.id !== inId)
        g.squad.push({ id: outId, sell: outSell })
        if (g.xi.includes(inId)) g.xi = g.xi.map((id) => (id === inId ? outId : id))
        if (g.captain === inId) g.captain = outId
        if (g.vice === inId) g.vice = outId
        g.bank = Math.round(((g.bank || 0) + inPrice - outSell) * 10) / 10
      }
      const g = d.gws[t0]
      g.transfers_in = g.transfers_in.filter((_, i) => i !== k)
      g.transfers_out = g.transfers_out.filter((_, i) => i !== k)
      return d
    })
  }

  const createFromEntry = () => {
    if (!entry?.squad) return
    // Four gameweeks is not a plan - chips, fixture swings and price moves all
    // play out over longer than that, and the draft length is what caps how far
    // ahead the Planner can look at all.
    const horizon = (status?.scheduled_gws || [])
      .filter((g) => g >= (editableGw ?? status.next_gw)).slice(0, DEFAULT_HORIZON)
    if (!horizon.length) {
      setToast({ kind: 'err', msg: 'No upcoming gameweeks known yet — run a data pull (⟳ Data, top right) first.' })
      return
    }
    const squad = entry.squad.map((p) => ({
      id: p.element,
      // never store a £0.0 sell — it makes every later transfer unaffordable
      sell: p.selling_price || byId.get(p.element)?.price || 0,
    }))
    const ids = squad.map((s) => s.id)
    const picksXi = entry.squad.filter((p) => p.multiplier > 0).map((p) => p.element)
    const gws = horizon.map((gw) => {
      const epFor = (id) => epOf(proj, id, gw)
      const xi = xiLegal(picksXi, posOf) ? [...picksXi] : bestXI(ids, posOf, epFor)
      const cap = entry.squad.find((p) => p.is_captain)?.element
      const vice = entry.squad.find((p) => p.is_vice)?.element
      const sorted = [...xi].sort((a, b) => epFor(b) - epFor(a))
      const captain = cap && xi.includes(cap) ? cap : sorted[0]
      return {
        // a chip already activated on the FPL site is a fact about this
        // gameweek, not a plan — carry it in rather than making it be re-set
        gw, chip: gw === horizon[0] ? entry?.chips?.active || null : null,
        squad: structuredClone(squad), xi,
        captain,
        vice: vice && xi.includes(vice) && vice !== captain
          ? vice : sorted.find((id) => id !== captain) || null,
        transfers_in: [], transfers_out: [],
        bank: entry.bank,
      }
    })
    const label = String.fromCharCode(65 + drafts.length)
    const d = withBaseline(applyFtLedger(
      { id: `d${Date.now()}`, label: `Draft ${label}`, source: 'entry', gws },
      entry.free_transfers))
    setDrafts((ds) => [...ds, d])
    setActiveDraftId(d.id)
  }

  if (!draft) {
    return (
      <div className="panel center-note">
        <h3>No drafts yet</h3>
        <p style={{ maxWidth: 560, margin: '0 auto 18px' }}>
          Run the <b>Solver</b> to generate optimised plans (they land here as
          drafts){entry?.squad ? ', or start from your current squad:' : '.'}
          {!entry?.squad && (
            <> The public FPL API can't see your picks before the deadline —
            use <b>⚠ set my team</b> (top right) to import your squad with your
            FPL login, or enter it manually. Otherwise the solver builds a
            fresh 15 from £100m.</>
          )}
        </p>
        {entry?.squad && (
          <button className="pill-btn accent" onClick={createFromEntry}>
            + Draft from current squad
          </button>
        )}
      </div>
    )
  }

  if (!draft.gws?.length) {
    return (
      <div className="panel center-note">
        <h3>{draft.label} has no gameweeks</h3>
        <p style={{ maxWidth: 520, margin: '0 auto 18px' }}>
          It was created before any fixture data was loaded, so there is
          nothing to plan. Delete it and re-create it from your squad.
        </p>
        <button className="pill-btn accent" onClick={() => {
          setDrafts((ds) => ds.filter((x) => x.id !== draft.id))
          setActiveDraftId(drafts.find((x) => x.id !== draft.id)?.id || null)
        }}>
          ✕ Delete this draft
        </button>
      </div>
    )
  }

  const evs = draft.gws.map((p) => gwEV(p, proj))
  const deltas = baselineDeltas(draft, proj)
  const isBuild = plan?.transfers_in.length === 15   // pre-season squad build
  const nMoves = plan && !isBuild ? plan.transfers_in.length : 0

  // manual moves vs baseline for the current gw (solver moves are in the baseline)
  const baseSquad = new Set((draft.baseline?.[gwIdx]?.squad || plan.squad).map((s) => s.id))
  const manualMoves = plan ? plan.transfers_in
    .map((inId, k) => ({ inId, outId: plan.transfers_out[k] }))
    .filter((m) => !draft.baseline || !baseSquad.has(m.inId)) : []

  return (
    <div>
      <GwBar draft={draft} gwIdx={gwIdx} setGwIdx={setGwIdx} evs={evs} deltas={deltas}
        plan={plan} nMoves={nMoves} updateDraft={updateDraft} undo={undo} canUndo={undoN > 0} />
      <div className="planner-grid">
        <div className="planner-main">
          {planIsPast && (
            <div className="past-gw-note">
              {status?.gw_in_progress && plan.gw === status?.next_gw
                ? <>GW{plan.gw} is in progress — its deadline has passed, so this gameweek is locked. Changes apply from <b>GW{editableGw}</b>.</>
                : <>GW{plan.gw} has already been played and is locked. Plan from GW{editableGw}.</>}
            </div>
          )}
          {!planIsPast && planUnprojected && (
            <div className="past-gw-note build">
              <span>
                No projections for GW{plan.gw} yet, so every point reads <b>0.0</b>.
                {isAdmin ? ' A data pull clears the cache.' : ' The lab projects the next six gameweeks on every automatic refresh — a longer draft fills in as the season moves.'}
              </span>
              {isAdmin && (
                <button className="pill-btn accent" disabled={building}
                  onClick={buildHorizon}>
                  {building ? <span className="spinner" /> : '⚙'} Build projections
                </button>
              )}
            </div>
          )}
          {plan && (
            <PitchView plan={plan} draft={draft} sel={sel} setSel={setSel}
              setStatPid={setStatPid} posOf={posOf} armed={armed} setArmed={setArmed}
              applyTransfer={applyTransfer}
              gwIdx={gwIdx} setToast={setToast} updateDraft={updateDraft} />
          )}
          {plan && (
            <ChangesStrip moves={manualMoves} plan={plan} draft={draft} gwIdx={gwIdx}
              byId={byId} proj={proj} undoTransfer={undoTransfer} />
          )}
          {/* the plan itself, under the pitch rather than folded away in the
              rail: it is the answer the whole tab exists to produce */}
          <PathsPanel draft={draft} byId={byId} gwIdx={gwIdx} setGwIdx={setGwIdx}
            proj={proj} />
          {/* The search belongs beside the pitch it acts on, not across the
              page from it: you pick a player here and then click the man he
              replaces, and having those two things in different columns made
              a two-step action feel like two unrelated ones. */}
          <Section id="add" title="Add a player"
            hint="search, then click who he replaces on the pitch"
            badge={armed ? armed.web_name : null} defaultOpen={!!armed}>
            <AddPlayerPanel plan={plan} players={players} byId={byId} proj={proj}
              posOf={posOf} armed={armed} setArmed={setArmed} />
          </Section>
        </div>

        {/* One column, read top to bottom: which route am I on, and what does
            the model say to do about it. Everything that is a TOOL rather than
            an answer lives on the left under the pitch, where there is room
            for it and where the thing it acts on is already on screen. */}
        <aside className="planner-rail">
          <DraftsPanel drafts={drafts} setDrafts={setDrafts} proj={proj}
            activeDraftId={draft.id} setActiveDraftId={setActiveDraftId}
            gwIdx={gwIdx} setGwIdx={setGwIdx} createFromEntry={createFromEntry}
            entry={entry} />
          {plan && (
            <Advice draft={draft} gwIdx={gwIdx} plan={plan} posOf={posOf}
              updateDraft={updateDraft} setToast={setToast} proj={proj}
              byId={byId} players={players} entryChips={entry?.chips} />
          )}
          {plan && (
            <Section id="dna" title="Team DNA" hint="the shape of the squad, not its total">
              <TeamDna plan={plan} draft={draft} gwIdx={gwIdx} byId={byId} proj={proj} />
            </Section>
          )}
        </aside>
      </div>
      {statPid && plan && (
        <PlayerModal pid={statPid} draft={draft} plan={plan}
          close={() => setStatPid(null)}
          actions={{
            captain: () => {
              if (!plan.xi.includes(statPid)) return
              updateDraft((d) => {
                const g = d.gws[gwIdx]
                if (g.vice === statPid) g.vice = g.captain
                g.captain = statPid
                return d
              })
              setStatPid(null)
            },
            vice: () => {
              if (!plan.xi.includes(statPid) || plan.captain === statPid) return
              updateDraft((d) => { d.gws[gwIdx].vice = statPid; return d })
              setStatPid(null)
            },
            swap: () => { setSel({ pid: statPid, mode: 'swap' }); setStatPid(null) },
            transfer: () => { setXfer(statPid); setStatPid(null) },
            undo: manualMoves.some((m) => m.inId === statPid)
              ? () => { undoTransfer(statPid); setStatPid(null) } : null,
          }} />
      )}
      {xfer && plan && (
        <TransferModal draft={draft} gwIdx={gwIdx} outId={xfer} byId={byId}
          proj={proj} posOf={posOf} close={() => setXfer(null)}
          applyTransfer={applyTransfer} />
      )}
    </div>
  )
}

/* Model assist and the chip advisor answered the same question — "what should
   I do about this gameweek?" — from two panels a screen apart, one of which
   said "Best Free Hit for this gameweek" while the other said "play a Free
   Hit in GW10". One panel, two tabs: what to do NOW, and where a chip is
   worth playing across the draft. */
function Advice({ draft, gwIdx, plan, posOf, updateDraft, setToast, proj,
                  byId, players, entryChips }) {
  const [view, setView] = usePersisted('planner.advice', 'now')
  return (
    <div className="panel assist advice-panel">
      <div className="panel-head">
        Model assist
        <span className="seg" role="tablist" aria-label="advice view">
          <button role="tab" aria-selected={view === 'now'}
            className={view === 'now' ? 'on' : ''}
            onClick={() => setView('now')}>GW{plan.gw}</button>
          <button role="tab" aria-selected={view === 'chips'}
            className={view === 'chips' ? 'on' : ''}
            onClick={() => setView('chips')}>Chips</button>
        </span>
      </div>
      {view === 'now' ? (
        <ModelAssist draft={draft} gwIdx={gwIdx} plan={plan} posOf={posOf}
          updateDraft={updateDraft} setToast={setToast} embedded />
      ) : (
        <div className="advice-chips">
          <ChipAdvisor draft={draft} proj={proj} byId={byId} players={players}
            posOf={posOf} updateDraft={updateDraft} entryChips={entryChips} />
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ */

const ALL_CHIPS = ['bench_boost', 'triple_captain', 'wildcard', 'freehit']

function GwBar({ draft, gwIdx, setGwIdx, evs, deltas, plan, nMoves, updateDraft, undo, canUndo }) {
  const { proj, status, entry, editableGw } = useStore()
  const avail = chipAvailability(entry?.chips, draft.gws.map((p) => p.gw))
  const total = evs.reduce((a, b) => a + b, 0)
  const dTotal = deltas ? deltas.reduce((a, b) => a + b, 0) : null
  const [openChip, setOpenChip] = useState(null)

  const setChip = (c, gw) => {
    updateDraft((d) => {
      for (const p of d.gws) if (p.chip === c) p.chip = null
      if (gw != null) {
        const t = d.gws.find((p) => p.gw === gw)
        if (t) t.chip = c
      }
      return d
    })
    setOpenChip(null)
  }

  const planned = new Set(draft.gws.map((p) => p.gw))
  const nextUnplanned = (status?.scheduled_gws || [])
    .filter((g) => g >= (editableGw ?? 1) && !planned.has(g))[0] ?? null

  const Delta = ({ v }) => (v == null || Math.abs(v) < 0.05 ? null : (
    <span className={`dv ${v > 0 ? 'up' : 'down'}`}>{v > 0 ? '+' : ''}{fmt1(v)}</span>
  ))

  return (
    <div className="gwbar">
      <div className="pager">
        {draft.gws.map((p, i) => {
          // played is the calendar; unprojected is a job you can run
          const past = editableGw != null && p.gw < editableGw
          const noProj = !past && !gwHasProj(proj, p.gw)
          return (
            <button key={p.gw}
              className={`gw-dot ${i === gwIdx ? 'active' : ''} ${past ? 'past' : ''} ${noProj ? 'noproj' : ''}`}
              title={past ? `GW${p.gw} has already been played`
                : noProj ? `GW${p.gw} has no projections yet — build them from the Projections tab`
                : undefined}
              onClick={() => setGwIdx(i)}>
              {p.gw}
              {p.chip && <span className="chipmark">{CHIP_SHORT[p.chip]}</span>}
            </button>
          )
        })}
        {/* a draft can only look as far ahead as it is long */}
        <button className="gw-dot add" title="add the next gameweek to this plan"
          disabled={!nextUnplanned}
          onClick={() => updateDraft((d) => {
            const last = d.gws[d.gws.length - 1]
            d.gws.push({
              ...structuredClone(last), gw: nextUnplanned, chip: null,
              transfers_in: [], transfers_out: [], sold: {},
            })
            return d
          })}>+</button>
      </div>
      <div className="chipbar">
        {openChip && (
          <div style={{ position: 'fixed', inset: 0, zIndex: 39 }}
            onClick={() => setOpenChip(null)} />
        )}
        {ALL_CHIPS.map((c) => {
          const at = draft.gws.find((p) => p.chip === c)
          const a = avail[c]
          return (
            <div className="dd" key={c}>
              <button className={`chip-btn ${at ? 'on' : ''} ${!a?.usable ? 'spent' : ''}`}
                title={chipNote(a, CHIP_NAME[c])}
                disabled={!a?.usable}
                onClick={() => setOpenChip(openChip === c ? null : c)}>
                ⚡ {CHIP_SHORT[c]}{at ? ` GW${at.gw}` : ''}
                {a?.active && <em> live</em>}
                {!a?.usable && a?.played?.length ? <em> GW{a.played[a.played.length - 1]}</em> : null} ▾
              </button>
              {openChip === c && (
                <div className="dd-menu" style={{ minWidth: 150 }}>
                  <div className="ttl">{CHIP_LONG[c].replace(' Played', '')}</div>
                  <button className="dd-item" onClick={() => setChip(c, null)}>
                    Don't use
                  </button>
                  {draft.gws.filter((p) => !a?.known
                    || a.windows.some(([x, y]) => p.gw >= x && p.gw <= y))
                    .map((p) => (
                    <button key={p.gw}
                      className={`dd-item ${p.chip === c ? 'on' : ''}`}
                      onClick={() => setChip(c, p.gw)}>
                      GW{p.gw}
                      {p.chip && p.chip !== c && (
                        <span style={{ color: 'var(--muted-2)' }}> · has {CHIP_SHORT[p.chip]}</span>
                      )}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )
        })}
      </div>
      <button className="pill-btn" onClick={undo} disabled={!canUndo} title="Undo last change (Ctrl+Z)">
        ↶ Undo
      </button>
      <div className="stats">
        <div className="stat"><span className="k">GW pts</span>
          <span className="v">{fmt1(evs[gwIdx])}</span>
          {deltas && <Delta v={deltas[gwIdx]} />}</div>
        <div className="stat"><span className="k">Total</span>
          <span className="v">{fmt1(total)}</span>
          <Delta v={dTotal} /></div>
        <div className="stat"><span className="k">ITB</span>
          <span className="v">{money(plan?.bank)}</span></div>
        <div className="stat"><span className="k">Moves</span>
          <span className="v" style={{ color: plan?.hits ? 'var(--red)' : undefined }}>
            {nMoves}{plan?.hits ? ` (-${plan.hits * 4})` : ''}
          </span></div>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */

function PitchView({ plan, sel, setSel, setStatPid, posOf, updateDraft, gwIdx, setToast,
                     armed, setArmed, applyTransfer }) {
  const { proj } = useStore()
  const rows = formationRows(plan.xi, posOf)
  const bench = plan.squad
    .filter((s) => !plan.xi.includes(s.id))
    .sort((a, b) => (posOf(a.id) === 'GK' ? -1 : posOf(b.id) === 'GK' ? 1 : 0))

  const trySwap = (a, b) => {
    const inXiA = plan.xi.includes(a)
    const inXiB = plan.xi.includes(b)
    if (inXiA === inXiB) return false
    const [xiP, benchP] = inXiA ? [a, b] : [b, a]
    const nextXi = plan.xi.map((id) => (id === xiP ? benchP : id))
    if (!xiLegal(nextXi, posOf)) {
      setToast({ kind: 'err', msg: 'Illegal formation — pick a different swap.' })
      return false
    }
    updateDraft((d) => {
      const p = d.gws[gwIdx]
      p.xi = nextXi
      if (p.captain === xiP) p.captain = benchP
      if (p.vice === xiP) p.vice = benchP
      return d
    })
    return true
  }

  const onCardClick = (pid, e) => {
    e.stopPropagation()
    if (armed) {
      if (applyTransfer(pid, armed)) setArmed(null)
      return
    }
    if (sel?.mode === 'swap') {
      if (sel.pid !== pid && trySwap(sel.pid, pid)) setSel(null)
      return
    }
    setStatPid(pid)
  }

  // when a player is armed, show what swapping him in would do this gw
  const armedDelta = (pid) => (armed && posOf(pid) === armed.position
    ? epOf(proj, armed.id, plan.gw) - epOf(proj, pid, plan.gw) : null)

  return (
    <div className="panel pitch-wrap" onClick={() => { setSel(null) }}>
      {armed && (
        <div className="armed-banner">
          Bringing in <b>{armed.web_name}</b> ({armed.position}, {money(armed.price)}) — click the
          {' '}{armed.position} to replace. Green/red = projected change this GW.
          <button className="pill-btn" style={{ marginLeft: 10 }} onClick={() => setArmed(null)}>cancel</button>
        </div>
      )}
      <div className="pitch">
        <PitchLines />
        <div className="pitch-rows">
          {POSITIONS.map((pp) => (
            <div className="pitch-row" key={pp}>
              {rows[pp].map((pid) => (
                <Card key={pid} pid={pid} plan={plan} sel={sel}
                  onClick={onCardClick} posOf={posOf}
                  dim={armed && posOf(pid) !== armed.position} delta={armedDelta(pid)} />
              ))}
            </div>
          ))}
        </div>
      </div>
      <div className="bench">
        {bench.map((s) => (
          <Card key={s.id} pid={s.id} plan={plan} sel={sel}
            onClick={onCardClick} posOf={posOf} bench
            dim={armed && posOf(s.id) !== armed.position} delta={armedDelta(s.id)} />
        ))}
      </div>
      {sel?.mode === 'swap' && (
        <p style={{ textAlign: 'center', color: 'var(--gold)', marginTop: 10, fontSize: 12.5 }}>
          Swap mode — click the player to exchange with, or click the pitch to cancel.
        </p>
      )}
    </div>
  )
}

function Card({ pid, plan, sel, onClick, posOf, dim, delta }) {
  const { byId, teams, proj } = useStore()
  const fixOf = useFixtureLookup()
  const p = byId.get(pid)
  const team = teams[String(p?.team_id)]
  const ep = epOf(proj, pid, plan.gw)
  const fixes = fixOf(p?.team_id, plan.gw)
  const attacking = !['GK', 'DEF'].includes(posOf(pid))
  const isCap = plan.captain === pid
  const isVice = plan.vice === pid
  const isNew = plan.transfers_in.length < 15 && plan.transfers_in.includes(pid)
  const selected = sel?.pid === pid

  /* A clickable <div> is invisible to a keyboard and to a screen reader, and
     the pitch is the primary control on the tab — so every card is a real
     control: focusable, operable with Enter/Space, and named. */
  return (
    <div className={`pcard ${selected ? 'selected' : ''} ${dim ? 'dim' : ''}`}
      role="button" tabIndex={0}
      aria-label={`${p?.web_name || pid}, ${posOf(pid)}, ${fmt1(ep)} projected points`}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick(pid, e) }
      }}
      onClick={(e) => onClick(pid, e)}>
      {isCap && <span className="armband">{plan.chip === 'triple_captain' ? 'T' : 'C'}</span>}
      {isVice && <span className="armband vice">V</span>}
      <Flag p={p} />
      {delta != null && (
        <span className={`swapdelta ${delta >= 0 ? 'up' : 'down'}`}>
          {delta >= 0 ? '+' : ''}{fmt1(delta)}
        </span>
      )}
      <img className="shirt" alt=""
        src={shirtUrl(team?.code, posOf(pid) === 'GK')}
        onError={(e) => { e.currentTarget.style.visibility = 'hidden' }} />
      <div className="pname" style={isNew ? { color: 'var(--green)' } : undefined}>
        {p?.web_name || pid}
      </div>
      <div className="pmeta">
        <span className="ep">{fmt1(ep)}</span>
        <span className="fix">
          {fixes.length ? fixes.map((f, i) => {
            // attackers care how hard the opponent is to score against,
            // defenders how hard it is to keep a clean sheet against them
            const d = f[attacking ? 'diff_att' : 'diff_def'] ?? f.fdr ?? 3
            const c = fdrColor(d)
            return (
              <span key={i} className="fixchip"
                style={{ background: c.bg, color: c.fg }}
                title={`${f.oppShort} ${f.home ? '(H)' : '(A)'} — ${attacking ? 'attacking' : 'defensive'} difficulty ${fmt1(d)}/5`}>
                {f.oppShort.toLowerCase()}{f.home ? '' : '↓'}
              </span>
            )
          }) : '–'}
        </span>
        <span className="pr">{p ? p.price.toFixed(1) : ''}</span>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */

// Manual moves vs the draft's baseline, with the model's verdict per move.
function ChangesStrip({ moves, plan, draft, gwIdx, byId, proj, undoTransfer }) {
  if (!moves.length) return null
  const later = draft.gws.slice(gwIdx).map((g) => g.gw)
  return (
    <div className="panel changes">
      <div className="panel-head">Your changes this GW — model verdict</div>
      {moves.map((m) => {
        const dNow = epOf(proj, m.inId, plan.gw) - epOf(proj, m.outId, plan.gw)
        const dHor = later.reduce((a, g) => a + epOf(proj, m.inId, g) - epOf(proj, m.outId, g), 0)
        return (
          <div key={`${m.outId}-${m.inId}`} className="change-row">
            <span className="out">{byId.get(m.outId)?.web_name || m.outId}</span>
            <span className="arrow">→</span>
            <span className="in">{byId.get(m.inId)?.web_name || m.inId}</span>
            <span className={`dv ${dNow >= 0 ? 'up' : 'down'}`} title="this gameweek">
              {dNow >= 0 ? '+' : ''}{fmt1(dNow)} GW
            </span>
            <span className={`dv ${dHor >= 0 ? 'up' : 'down'}`} title={`GW${later[0]}–${later[later.length - 1]}`}>
              {dHor >= 0 ? '+' : ''}{fmt1(dHor)} horizon
            </span>
            <button className="pill-btn" onClick={() => undoTransfer(m.inId)}>↶ undo</button>
          </div>
        )
      })}
    </div>
  )
}

/* ------------------------------------------------------------------ */

// Search any player and arm him; then click the squad player to replace.
/* What the plan does to the SHAPE of the squad, not just its total.

   Two squads worth the same expected points can be completely different
   animals - one template and balanced, one loaded with premiums and carrying a
   dead bench. The axes are the same six the Mini League uses (they share
   `dna.js`), so "my Attack" means the same thing in both places.

   Scaled against fixed absolute domains rather than against the two squads
   being compared: min-maxing two series would pin every axis at one end or the
   other and make a single transfer look like a personality transplant. */
function TeamDna({ plan, draft, gwIdx, byId, proj }) {
  const base = draft.baseline?.[gwIdx]?.squad
  const series = useMemo(() => {
    const asRows = (squad, xi) => (squad || []).map((s) => ({
      id: s.id, benched: !(xi || []).includes(s.id),
    }))
    const now = base
      ? dnaOf(asRows(base, draft.baseline?.[gwIdx]?.xi), { byId, proj, gw: plan.gw })
      : null
    const after = dnaOf(asRows(plan.squad, plan.xi), { byId, proj, gw: plan.gw })
    if (!after) return null
    const out = []
    if (now) {
      out.push({ name: 'Before', color: VIZ_MUTED, values: dnaScaled(now), raw: dnaRaw(now) })
    }
    out.push({ name: now ? 'After plan' : 'This squad', color: VIZ[0],
               values: dnaScaled(after), raw: dnaRaw(after) })
    return out
  }, [plan, base, byId, proj, gwIdx])   // eslint-disable-line react-hooks/exhaustive-deps

  if (!series) return null
  return (
    <div className="dna-panel">
      <Radar axes={DNA_AXES} series={series} size={290} />
      <p className="viz-note">
        {DNA_AXES.map((a) => a.label + ' = ' + a.hint).join(' · ')}
      </p>
    </div>
  )
}


function AddPlayerPanel({ plan, players, byId, proj, posOf, armed, setArmed }) {
  const [q, setQ] = useState('')
  const [pos, setPos] = useState('ALL')
  const owned = new Set((plan?.squad || []).map((s) => s.id))
  const gw = plan?.gw
  const opts = useMemo(() => {
    if (!gw) return []
    const lq = q.toLowerCase()
    return players
      .filter((p) => !owned.has(p.id) && (pos === 'ALL' || p.position === pos))
      .filter((p) => !lq || p.web_name.toLowerCase().includes(lq) || p.name.toLowerCase().includes(lq))
      .map((p) => ({ ...p, ep: epOf(proj, p.id, gw) }))
      .sort((a, b) => b.ep - a.ep)
      .slice(0, 30)
  }, [players, q, pos, gw, proj, plan])
  if (!plan) return null
  return (
    <>
      <div className="add-controls">
        <div className="search" style={{ minWidth: 0, flex: 1 }}>
          🔍<input placeholder="Search players…" value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        <span style={{ display: 'flex', gap: 4 }}>
          {['ALL', 'GK', 'DEF', 'MID', 'FWD'].map((v) => (
            <button key={v} className={`mode-pill ${pos === v ? 'on' : ''}`} onClick={() => setPos(v)}>{v}</button>
          ))}
        </span>
      </div>
      <div className="addlist">
        {opts.map((p) => (
          <PlayerRow key={p.id} p={p} armed={armed?.id === p.id}
            onClick={() => setArmed(armed?.id === p.id ? null : p)} />
        ))}
        {!opts.length && <div className="ml-note">No players match.</div>}
      </div>
      <div className="fold-note">Click a player to pick him up, then click the
        man he replaces on the pitch.</div>
    </>
  )
}

/* A player in a list: his club's shirt, then name, then club · role ·
   ownership, then the two numbers. The shirt does the identifying work a
   three-letter club code cannot — you recognise a kit before you read a
   name — and it gives the row a fixed left edge so a column of them scans
   as a column instead of as ragged text. */
function PlayerRow({ p, armed, onClick, right }) {
  const { teams } = useStore()
  const team = teams[String(p.team_id)]
  return (
    <div className={`ta-item prow ${armed ? 'armed' : ''}`}
      role={onClick ? 'button' : undefined} tabIndex={onClick ? 0 : undefined}
      aria-pressed={onClick ? !!armed : undefined}
      onKeyDown={onClick ? (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick(e) }
      } : undefined}
      onClick={onClick}>
      <img className="prow-shirt" alt="" loading="lazy"
        src={shirtUrl(team?.code, p.position === 'GK')}
        onError={(e) => { e.currentTarget.style.visibility = 'hidden' }} />
      <div className="prow-id">
        <div className="prow-name">{p.web_name}<Flag p={p} /></div>
        <div className="prow-meta">
          <b>{team?.short || '???'}</b>
          <i>·</i>{p.position}
          <i>·</i>{p.own}%
        </div>
      </div>
      {right ?? (
        <>
          <span className="chip blue num">{fmt1(p.ep)}</span>
          <span className="pr num">{money(p.price)}</span>
        </>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ */

// Heuristic chip hints from this draft's own projections. The Solver is the
// authority (it evaluates chips exactly); these flag where a chip looks valuable.
function ChipAdvisor({ draft, proj, byId, players, posOf, updateDraft, entryChips }) {
  const stats = useMemo(() => draft.gws.map((p) => {
    const xiEp = p.xi.reduce((a, id) => a + epOf(proj, id, p.gw), 0)
    const cap = p.xi.reduce((best, id) => {
      const e = epOf(proj, id, p.gw); return e > best.ep ? { id, ep: e } : best
    }, { id: null, ep: 0 })
    const benchEp = p.squad.filter((s) => !p.xi.includes(s.id))
      .reduce((a, s) => a + epOf(proj, s.id, p.gw), 0)
    // Best XI a Free Hit could actually buy this gw: the chip spends the
    // squad's selling value plus the bank, so the comparison has to respect
    // that budget (and the 3-per-club cap) or it advertises a gain you cannot
    // buy — which is also why it could never name the team.
    const budget = p.squad.reduce((a, s) => a + (s.sell ?? byId.get(s.id)?.price ?? 0), 0)
      + (p.bank || 0)
    const pool = players.map((x) => x.id)
    const fh = bestAffordableXI(
      pool, posOf, (id) => epOf(proj, id, p.gw),
      (id) => byId.get(id)?.price ?? 0, (id) => byId.get(id)?.team_id ?? 0, budget)
    const bestEp = fh.xi.reduce((a, id) => a + epOf(proj, id, p.gw), 0)
    return { gw: p.gw, chip: p.chip, xiEp, cap, benchEp, bestEp,
             gap: bestEp - xiEp, fhXi: fh.xi, fhCost: fh.cost, budget }
  }), [draft, proj, players])

  const median = (xs) => { const s = [...xs].sort((a, b) => a - b); return s.length ? s[Math.floor(s.length / 2)] : 0 }
  let hints = []
  if (stats.length) {
    const tc = stats.reduce((b, s) => (s.cap.ep > b.cap.ep ? s : b), stats[0])
    const tcEdge = tc.cap.ep - median(stats.map((s) => s.cap.ep))
    hints.push({ chip: 'triple_captain', gw: tc.gw, score: tcEdge,
      strong: tcEdge >= 1.5, gain: tc.cap.ep,
      text: `${byId.get(tc.cap.id)?.web_name || '?'} ${fmt1(tc.cap.ep)} pts — ${tcEdge >= 0 ? '+' : ''}${fmt1(tcEdge)} vs your typical captain` })
    const bb = stats.reduce((b, s) => (s.benchEp > b.benchEp ? s : b), stats[0])
    hints.push({ chip: 'bench_boost', gw: bb.gw, score: bb.benchEp, strong: bb.benchEp >= 8,
      gain: bb.benchEp, text: `bench projects ${fmt1(bb.benchEp)} pts` })
    const fh = stats.reduce((b, s) => (s.gap > b.gap ? s : b), stats[0])
    hints.push({ chip: 'freehit', gw: fh.gw, score: fh.gap, strong: fh.gap >= 12,
      gain: fh.gap, xi: fh.fhXi, cost: fh.fhCost, budget: fh.budget,
      text: `your XI ${fmt1(fh.xiEp)} vs ${fmt1(fh.bestEp)} best affordable (${fh.gap >= 0 ? '+' : ''}${fmt1(fh.gap)}) — ${money(fh.fhCost)} of ${money(fh.budget)}` })
    const first = stats[0].gap
    const worse = stats.find((s) => s.gap - first >= 4)
    if (worse) {
      hints.push({ chip: 'wildcard', gw: worse.gw, score: worse.gap - first, strong: worse.gap - first >= 8,
        gain: worse.gap - first, text: `gap to the best XI grows by ${fmt1(worse.gap - first)} from GW${worse.gw}` })
    }
  }
  // A chip you have already played is not advice, it is noise: drop the hint
  // rather than let the panel recommend something FPL will not let you do.
  const usable = chipAvailability(entryChips, draft.gws.map((p) => p.gw))
  hints = hints.filter((h) => usable[h.chip]?.usable
    && (!usable[h.chip].known
        || usable[h.chip].windows.some(([x, y]) => h.gw >= x && h.gw <= y)))
  const [showXi, setShowXi] = useState(null)
  const apply = (chip, gw) => updateDraft((d) => {
    for (const p of d.gws) if (p.chip === chip) p.chip = null
    const t = d.gws.find((p) => p.gw === gw)
    if (t) t.chip = chip
    return d
  })
  const active = new Set(draft.gws.filter((p) => p.chip).map((p) => `${p.chip}@${p.gw}`))
  return (
    <>
      <div className="fold-note">Hints from this draft's own projections — the
        Solver is the authority that prices a chip exactly.</div>
      {hints.map((h) => (
        <div key={h.chip}>
          <div className={`advice ${h.strong ? 'strong' : ''}`}>
            <span className="chip gold">{CHIP_SHORT[h.chip]}</span>
            <span className="num" style={{ fontWeight: 800 }}>GW{h.gw}</span>
            <span className="txt">{h.text}</span>
            {h.xi?.length > 0 && (
              <button className="pill-btn" onClick={() => setShowXi(showXi === h.chip ? null : h.chip)}>
                {showXi === h.chip ? 'hide XI' : 'show XI'}
              </button>
            )}
            {active.has(`${h.chip}@${h.gw}`)
              ? <span className="chip green">set</span>
              : <button className="pill-btn" onClick={() => apply(h.chip, h.gw)}>apply</button>}
          </div>
          {showXi === h.chip && h.xi?.length > 0 && (
            <div className="fh-xi">
              {POSITIONS.map((pos) => {
                const ids = h.xi.filter((id) => posOf(id) === pos)
                if (!ids.length) return null
                return (
                  <div key={pos} className="fh-row">
                    <span className="fh-pos">{pos}</span>
                    {ids.sort((a, b) => epOf(proj, b, h.gw) - epOf(proj, a, h.gw)).map((id) => (
                      <span key={id} className="fh-p">
                        {byId.get(id)?.web_name || id}
                        <em>{fmt1(epOf(proj, id, h.gw))}</em>
                        <i>{money(byId.get(id)?.price ?? 0)}</i>
                      </span>
                    ))}
                  </div>
                )
              })}
              <div className="fh-foot">
                XI costs {money(h.cost)} of {money(h.budget)} available (squad value + bank).
                Bench not shown — a legal 15 needs four more, reserved from the budget.
                Estimate only: run the Solver with Free Hit enabled for the exact squad.
              </div>
            </div>
          )}
        </div>
      ))}
      {!hints.length && <div className="ml-note">Nothing worth a chip in this draft's gameweeks.</div>}
    </>
  )
}

/* ------------------------------------------------------------------ */

function TransferModal({ draft, gwIdx, outId, byId, proj, posOf, close, applyTransfer }) {
  const { players } = useStore()
  const [q, setQ] = useState('')
  const plan = draft.gws[gwIdx]
  const out = plan.squad.find((s) => s.id === outId)
  // sell: 0 means the draft predates selling prices — use the current price
  const budget = (plan.bank || 0) + (out?.sell || byId.get(outId)?.price || 0)
  const pos = posOf(outId)
  const owned = new Set(plan.squad.map((s) => s.id))

  const opts = useMemo(() => players
    .filter((p) => p.position === pos && !owned.has(p.id))
    .filter((p) => !q || p.web_name.toLowerCase().includes(q.toLowerCase()) ||
      p.name.toLowerCase().includes(q.toLowerCase()))
    .map((p) => ({ ...p, ep: epOf(proj, p.id, plan.gw), afford: p.price <= budget + 1e-9 }))
    .sort((a, b) => b.ep - a.ep)
    .slice(0, 40), [players, q, pos, plan.gw, proj, budget, owned])

  const outEp = epOf(proj, outId, plan.gw)
  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(10,10,22,0.6)', zIndex: 90 }}
      onClick={close}>
      <div className="panel" onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: 520, margin: '8vh auto', boxShadow: 'var(--shadow)' }}>
        <div className="panel-head">
          Transfer out {byId.get(outId)?.web_name} ({fmt1(outEp)}) · budget {money(budget)}
          <button style={{ marginLeft: 'auto', color: 'var(--muted)' }} onClick={close}>✕</button>
        </div>
        <div style={{ padding: 12 }}>
          <div className="search" style={{ marginBottom: 10 }}>
            🔍<input autoFocus placeholder={`Search ${pos}s…`} value={q}
              onChange={(e) => setQ(e.target.value)} />
          </div>
          <div style={{ maxHeight: 380, overflow: 'auto' }}>
            {opts.map((p) => {
              const d = p.ep - outEp
              return (
                <div key={p.id} className="tm-row" style={{ opacity: p.afford ? 1 : 0.4 }}
                  onClick={() => { if (applyTransfer(outId, p)) close() }}>
                  <PlayerRow p={p} right={(
                    <>
                      <span className={`dv ${d >= 0 ? 'up' : 'down'}`}>{d >= 0 ? '+' : ''}{fmt1(d)}</span>
                      <span className="chip blue num">{fmt1(p.ep)}</span>
                      <span className="pr num">{money(p.price)}</span>
                    </>
                  )} />
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */

function DraftsPanel({ drafts, setDrafts, proj, activeDraftId, setActiveDraftId,
                       gwIdx, setGwIdx, createFromEntry, entry }) {
  const rename = (d) => {
    const name = prompt('Draft name', d.label)
    if (name) setDrafts((ds) => ds.map((x) => (x.id === d.id ? { ...x, label: name } : x)))
  }
  const remove = (d) => {
    if (!confirm(`Delete ${d.label}?`)) return
    setDrafts((ds) => ds.filter((x) => x.id !== d.id))
    if (activeDraftId === d.id) setActiveDraftId(drafts.find((x) => x.id !== d.id)?.id || null)
  }
  const duplicate = (d) => {
    const copy = structuredClone(d)
    copy.id = `d${Date.now()}`
    copy.label = `${d.label} [copy]`
    setDrafts((ds) => [...ds, copy])
  }
  /* Branching is how you actually plan: keep everything up to the gameweek you
     are looking at, then try a different route from there. Duplicating and
     hand-unwinding the tail is the same thing done badly. */
  const branch = (d) => {
    const at = d.id === activeDraftId ? gwIdx : 0
    const copy = structuredClone(d)
    copy.id = `d${Date.now()}`
    copy.label = `${d.label} → GW${d.gws[at]?.gw ?? '?'}`
    // the future is yours to redraw: clear planned moves and chips after the
    // branch point, keeping the squad the route has reached
    for (let i = at; i < copy.gws.length; i++) {
      if (i > at) copy.gws[i].chip = null
      copy.gws[i].transfers_in = []
      copy.gws[i].transfers_out = []
      copy.gws[i].sold = {}
    }
    delete copy.baseline
    copy.source = `branch of ${d.label}`
    setDrafts((ds) => [...ds, copy])
    setActiveDraftId(copy.id)
  }

  // which route is actually ahead, and by how much
  const totals = drafts.map((d) => d.gws.reduce((a, p) => a + gwEV(p, proj), 0))
  const bestTot = totals.length ? Math.max(...totals) : 0
  const activeTot = totals[drafts.findIndex((d) => d.id === activeDraftId)] ?? 0

  const active = drafts.find((d) => d.id === activeDraftId) || drafts[0]

  return (
    <div className="panel drafts-panel">
      <div className="panel-head">
        Drafts
        <span className="dp-actions">
          <button className="pill-btn" onClick={() => branch(active)}
            disabled={!active}
            title="Keep this plan up to the gameweek you are looking at, then try a different route from there">
            ⑂ Branch from GW{active?.gws?.[gwIdx]?.gw ?? '?'}
          </button>
          {entry?.squad && (
            <button className="pill-btn accent" onClick={createFromEntry}
              title="Start a fresh plan from the fifteen you own right now">
              + New draft
            </button>
          )}
        </span>
      </div>
      {drafts.length > 1 && (
        <div className="routes-verdict">
          Best draft projects <b>{fmt1(bestTot)}</b>
          {Math.abs(bestTot - activeTot) >= 0.05 ? (
            <> — the one you are editing is <span className="down">
              {fmt1(activeTot - bestTot)}</span> behind it.</>
          ) : <> — that is the one you are editing.</>}
          <span className="note"> Totals are undecayed projected XI points and
            ignore hits, so compare like with like.</span>
        </div>
      )}
      {drafts.map((d, di) => {
        const evs = d.gws.map((p) => gwEV(p, proj))
        const tot = evs.reduce((a, b) => a + b, 0)
        const dl = baselineDeltas(d, proj)
        const leads = drafts.length > 1 && Math.abs(tot - bestTot) < 1e-9
        return (
          <div key={d.id} className={`draft-row ${d.id === activeDraftId ? 'active' : ''} ${leads ? 'leads' : ''}`}>
            <div className="draft-name" onClick={() => setActiveDraftId(d.id)}
              onDoubleClick={() => rename(d)} title="click to select · double-click to rename">
              {d.label}
              <div className="sub">{d.source === 'solver' ? `obj ${fmt1(d.objective)}` : d.source}</div>
            </div>
            <div className="draft-cells">
              {d.gws.map((p, i) => (
                <button key={p.gw} type="button"
                  className={`draft-cell ${d.id === activeDraftId && i === gwIdx ? 'cur' : ''}`}
                  aria-label={`${d.label}, gameweek ${p.gw}`}
                  onClick={() => { setActiveDraftId(d.id); setGwIdx(i) }}>
                  <div className="ev">{fmt1(evs[i])}
                    {dl && Math.abs(dl[i]) >= 0.05 && (
                      <span className={`dv ${dl[i] > 0 ? 'up' : 'down'}`} style={{ fontSize: 9 }}>
                        {dl[i] > 0 ? '+' : ''}{fmt1(dl[i])}
                      </span>
                    )}
                  </div>
                  <div className="mv">
                    {p.chip ? <span className="chipflag">{CHIP_SHORT[p.chip]} </span> : null}
                    {p.transfers_in.length === 15 ? 'build'
                      : p.transfers_in.length
                        ? `${p.free_used ?? p.transfers_in.length}/${p.transfers_in.length}${p.hits ? ` -${p.hits * 4}` : ''}`
                        : '—'}
                  </div>
                </button>
              ))}
            </div>
            <div className="draft-total">
              <div className="t">{fmt1(tot)}</div>
              <div className="s">
                {leads ? 'best' : drafts.length > 1
                  ? `${fmt1(tot - bestTot)}` : `${d.gws.length} gws`}
              </div>
            </div>
            <div className="draft-tools">
              <button title={`Branch a new draft from GW${d.gws[d.id === activeDraftId ? gwIdx : 0]?.gw ?? ''}`}
                onClick={(e) => { e.stopPropagation(); branch(d) }}>⑂</button>
              <button title="Duplicate this draft"
                onClick={(e) => { e.stopPropagation(); duplicate(d) }}>⧉</button>
              <button className="danger" title="Delete this draft"
                onClick={(e) => { e.stopPropagation(); remove(d) }}>✕</button>
            </div>
          </div>
        )
      })}
      <div className="drafts-foot">
        <span>Saved automatically · double-click a name to rename</span>
        <button className="pill-btn"
          onClick={() => { if (confirm('Delete ALL drafts?')) setDrafts([]) }}>
          Reset all
        </button>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */

function PathsPanel({ draft, byId, gwIdx, setGwIdx, proj }) {
  const nm = (pid) => byId.get(pid)?.web_name || pid
  return (
    <div className="panel route">
      <div className="panel-head">
        The plan — {draft.label}
        <span className="panel-sub">{draft.gws.length} gameweeks · click one to edit it</span>
      </div>
      <div className="route-track">
        {draft.gws.map((p, i) => {
          const moves = p.transfers_out.length
          const build = p.transfers_in.length === 15
          return (
            <button key={p.gw} className={`route-stop ${i === gwIdx ? 'cur' : ''} ${p.chip ? 'chipped' : ''}`}
              onClick={() => setGwIdx(i)}>
              <span className="rs-gw">GW{p.gw}</span>
              {p.chip && <span className="rs-chip">{CHIP_SHORT[p.chip]}</span>}
              <span className="rs-ev">{fmt1(gwEV(p, proj))}</span>
              <span className="rs-moves">
                {build ? <em className="none">squad build</em>
                  : moves === 0 ? <em className="none">roll — no moves</em>
                    : p.transfers_out.map((o, k) => (
                      <em key={o}>
                        <span className="out">{nm(o)}</span>
                        <span className="arrow">→</span>
                        <span className="in">{nm(p.transfers_in[k])}</span>
                      </em>
                    ))}
              </span>
              <span className="rs-foot">
                £{fmt1(p.bank)}m · {p.free_after ?? '–'} FT
                {p.hits ? <b className="hit"> −{p.hits * 4}</b> : null}
              </span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
