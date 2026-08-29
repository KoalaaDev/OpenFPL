import React, { useMemo, useState } from 'react'
import { useStore } from '../store'
import { badgeUrl, fmt1 } from '../util'

/* Solver output: compare the plans, then read one.

   The solver returns one plan per PLAYSTYLE, not N near-identical optima, so
   the job of this view is comparison — what do Win-now, Balanced and Patient
   actually disagree about? Stacked cards made that almost impossible; you had
   to expand each one and hold two squads in your head. Here every plan is a
   column against a shared gameweek rail, so a difference is a horizontal
   glance.

   One number matters and it is easy to get wrong: plans are ranked on
   `total_ep` — undecayed projected XI points, net of hits — because each
   playstyle weights gameweeks with its own decay, which makes `objective`
   incomparable between them. */

const CHIP_SHORT = {
  wildcard: 'WC', freehit: 'FH', bench_boost: 'BB', triple_captain: 'TC',
}
const CHIP_LONG = {
  wildcard: 'Wildcard', freehit: 'Free Hit', bench_boost: 'Bench Boost',
  triple_captain: 'Triple Captain',
}
const POS_ORDER = ['GK', 'DEF', 'MID', 'FWD']
const POS_LABEL = { GK: 'GKP', DEF: 'DEF', MID: 'MID', FWD: 'FWD' }
const ACCENTS = ['#3ddc97', '#3fb8ff', '#b98bff', '#ffb61b', '#ff6d8d']

const money = (v) => (v == null ? '–' : `£${Number(v).toFixed(1)}m`)

function Badge({ teamId, teams }) {
  const code = teams?.[String(teamId)]?.code
  if (!code) return <span className="so-badge-gap" />
  return <img className="so-badge" src={badgeUrl(code)} alt="" loading="lazy"
    onError={(e) => { e.currentTarget.style.visibility = 'hidden' }} />
}

/* one transfer, out -> in. The lists are position-ordered by the solver, so
   pairing them by index is legal and each row is a real single transfer. */
function Move({ out, inn, teams, byId, compact }) {
  const meta = (t) => {
    const p = byId.get(t.player_id)
    return p ? `${p.position} · £${p.price.toFixed(1)}m` : t.position || ''
  }
  const team = (t) => byId.get(t.player_id)?.team_id
  return (
    <div className={`so-move ${compact ? 'compact' : ''}`}>
      <div className="so-side out">
        <span className="so-nm">{out.name}</span>
        {!compact && <span className="so-meta">{meta(out)}</span>}
      </div>
      {!compact && <Badge teamId={team(out)} teams={teams} />}
      <span className="so-arrow">»»</span>
      {!compact && <Badge teamId={team(inn)} teams={teams} />}
      <div className="so-side in">
        <span className="so-nm">{inn.name}</span>
        {!compact && <span className="so-meta">{meta(inn)}</span>}
      </div>
    </div>
  )
}

/* the whole XI+bench, laid out by position — what a chip week actually is */
function SquadBlock({ gw, teams, byId, compact }) {
  const byPos = useMemo(() => {
    const m = { GK: [], DEF: [], MID: [], FWD: [] }
    for (const r of gw.squad || []) (m[r.position] || m.MID).push(r)
    for (const k of POS_ORDER) {
      m[k].sort((a, b) => (b.in_xi ? 1 : 0) - (a.in_xi ? 1 : 0) || b.ep - a.ep)
    }
    return m
  }, [gw])
  return (
    <div className={`so-squad ${compact ? 'compact' : ''}`}>
      {POS_ORDER.map((pos) => (
        <div className="so-squad-row" key={pos}>
          <span className="so-pos">{POS_LABEL[pos]}</span>
          <div className="so-squad-names">
            {byPos[pos].map((r) => (
              <div key={r.player_id}
                className={`so-player ${r.in_xi ? 'is-xi' : 'is-bench'}`}>
                {!compact && <Badge teamId={r.team_id} teams={teams} />}
                <span className="so-nm">
                  {r.name}
                  {r.is_captain && <i className="so-arm" title="captain">C</i>}
                  {r.is_vice && <i className="so-arm v" title="vice">V</i>}
                </span>
                {!compact && <span className="so-meta">{money(r.price)}</span>}
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

/* the rail down the left: one stop per gameweek, chip weeks named */
function Rail({ gws, live }) {
  return gws.map((g, i) => (
    <div className="so-rail" key={g.gw}>
      <span className="so-rail-gw">
        {g.chip ? `${CHIP_SHORT[g.chip]}${g.gw}` : `GW${g.gw}`}
      </span>
      {i === 0 && live && <span className="so-rail-live">live</span>}
      {g.chip && <span className="so-rail-chip">{CHIP_LONG[g.chip]}</span>}
    </div>
  ))
}

function Meta({ g, freeFirst, first }) {
  const ft = freeFirst && first
    ? `${g.n_transfers}/∞`
    : `${Math.round(g.free_used || 0)}/${Math.round((g.free_used || 0) + (g.free_after || 0))}`
  return (
    <span className="so-meta-line">
      {g.chip !== 'freehit' && <span>{ft} FT</span>}
      <span>ITB {money(g.bank)}</span>
      {g.hits > 0 && <span className="so-hit">−{g.hits * 4}</span>}
    </span>
  )
}

export default function SolverOutput({ result, close, addDraft, draftLabel }) {
  const { teams, byId } = useStore()
  const plans = result?.plans || []
  const [tab, setTab] = useState('compare')
  const freeFirst = !!result?.state?.unlimited_transfers

  const best = useMemo(
    () => Math.max(...plans.map((p) => p.total_ep ?? -Infinity)), [plans])

  // the union of gameweeks, so every column shares one rail
  const gws = useMemo(() => {
    const s = new Set()
    for (const p of plans) for (const g of p.per_gw) s.add(g.gw)
    return [...s].sort((a, b) => a - b)
  }, [plans])

  // where the plans actually disagree — the reason to compare at all
  const differs = useMemo(() => {
    const out = {}
    for (const gw of gws) {
      const sigs = plans.map((p) => {
        const g = p.per_gw.find((x) => x.gw === gw)
        if (!g) return ''
        return `${g.chip || ''}|${(g.transfers_in || []).map((t) => t.player_id).sort().join(',')}`
      })
      out[gw] = new Set(sigs).size > 1
    }
    return out
  }, [plans, gws])

  if (!plans.length) return null

  return (
    <div className="so-overlay" onClick={close}>
      <div className="so-modal" onClick={(e) => e.stopPropagation()}>
        <header className="so-top">
          <h2>Transfer Solver</h2>
          <button className="so-x" onClick={close} aria-label="close">✕</button>
        </header>

        <nav className="so-tabs">
          <button className={tab === 'compare' ? 'on' : ''}
            onClick={() => setTab('compare')}>▥ Compare</button>
          {plans.map((p, i) => (
            <button key={i} className={tab === i ? 'on' : ''}
              onClick={() => setTab(i)}
              style={tab === i ? { color: ACCENTS[i % ACCENTS.length] } : undefined}>
              {p.style_label || `Plan ${i + 1}`}
            </button>
          ))}
        </nav>

        <div className="so-body">
          {tab === 'compare' ? (
            <div className="so-compare"
              style={{ gridTemplateColumns: `86px repeat(${plans.length}, minmax(240px, 1fr))` }}>
              <div className="so-cell so-corner" />
              {plans.map((p, i) => (
                <div className="so-cell so-planhead" key={i}
                  style={{ '--pa': ACCENTS[i % ACCENTS.length] }}>
                  <div className="so-planhead-top">
                    <span className="so-planname">
                      PLAN_{String(i + 1).padStart(2, '0')}
                    </span>
                    <span className="so-score">
                      {fmt1(p.total_ep)}
                      <i>pts</i>
                    </span>
                  </div>
                  <div className="so-planhead-sub">
                    <span className="so-tag">{p.style_label || 'plan'}</span>
                    {p.total_ep === best
                      ? <span className="so-tag best">best</span>
                      : <span className="so-delta">{fmt1(p.total_ep - best)}</span>}
                  </div>
                  <button className="so-apply"
                    onClick={() => addDraft(p, i)}>
                    Apply <span>to {draftLabel || 'a new draft'}</span> →
                  </button>
                </div>
              ))}

              {gws.map((gw) => {
                const rows = plans.map((p) => p.per_gw.find((x) => x.gw === gw))
                const anyChip = rows.some((g) => g && g.chip)
                return (
                  <React.Fragment key={gw}>
                    <div className={`so-cell so-railcell ${differs[gw] ? 'differs' : ''}`}>
                      <Rail gws={[rows.find(Boolean)].filter(Boolean)}
                        live={gw === gws[0]} />
                      {differs[gw] && <span className="so-differs" title="the plans disagree here">◆</span>}
                    </div>
                    {rows.map((g, i) => (
                      <div className="so-cell so-gwcell" key={i}
                        style={{ '--pa': ACCENTS[i % ACCENTS.length] }}>
                        {!g ? <span className="so-none">—</span> : (
                          <>
                            <Meta g={g} freeFirst={freeFirst} first={gw === gws[0]} />
                            {g.chip === 'freehit' || g.chip === 'wildcard' ? (
                              <SquadBlock gw={g} teams={teams} byId={byId} compact />
                            ) : (g.transfers_out || []).length ? (
                              (g.transfers_out || []).map((o, k) => (
                                <Move key={k} out={o} inn={g.transfers_in[k]}
                                  teams={teams} byId={byId} compact />
                              ))
                            ) : (
                              <span className="so-none">no transfer</span>
                            )}
                            {anyChip && !g.chip && (
                              <span className="so-chipnote">no chip</span>
                            )}
                          </>
                        )}
                      </div>
                    ))}
                  </React.Fragment>
                )
              })}
            </div>
          ) : (
            <PlanDetail plan={plans[tab]} i={tab} teams={teams} byId={byId}
              freeFirst={freeFirst} best={best} />
          )}
        </div>

        <footer className="so-foot">
          {tab !== 'compare' && (
            <span className="so-note">{plans[tab]?.style_note}</span>
          )}
          <button className="pill-btn" onClick={close}>✕ Close</button>
          {tab !== 'compare' && (
            <button className="pill-btn accent"
              onClick={() => addDraft(plans[tab], tab)}>
              ✓ Apply plan <span style={{ opacity: 0.75 }}>to {draftLabel || 'a new draft'}</span>
            </button>
          )}
        </footer>
      </div>
    </div>
  )
}

function PlanDetail({ plan, i, teams, byId, freeFirst, best }) {
  const acc = ACCENTS[i % ACCENTS.length]
  const hits = plan.per_gw.reduce((a, g) => a + (g.hits || 0), 0)
  const moves = plan.per_gw.reduce((a, g) => a + (g.n_transfers || 0), 0)
  return (
    <div className="so-detail" style={{ '--pa': acc }}>
      <div className="so-banner">
        <span className="so-planname">PLAN_{String(i + 1).padStart(2, '0')}</span>
        <span className="so-sep">/</span>
        <span className="so-score big">{fmt1(plan.total_ep)}<i>pts</i></span>
        {plan.total_ep === best
          ? <span className="so-tag best">best</span>
          : <span className="so-delta">{fmt1(plan.total_ep - best)} vs best</span>}
        <span className="so-banner-meta">
          {moves} transfer{moves === 1 ? '' : 's'}
          {hits ? <> · <span className="so-hit">{hits} hit{hits === 1 ? '' : 's'} −{hits * 4}</span></> : ' · no hits'}
        </span>
      </div>

      <div className="so-detail-head">
        <span />
        <span>FT</span><span>ITB</span>
      </div>

      {plan.per_gw.map((g, k) => (
        <div className={`so-detail-row ${g.chip ? 'chip' : ''}`} key={g.gw}>
          <div className="so-railcell">
            <Rail gws={[g]} live={k === 0} />
          </div>
          <div className="so-detail-body">
            {g.chip === 'freehit' || g.chip === 'wildcard' ? (
              <SquadBlock gw={g} teams={teams} byId={byId} />
            ) : (g.transfers_out || []).length ? (
              (g.transfers_out || []).map((o, j) => (
                <Move key={j} out={o} inn={g.transfers_in[j]} teams={teams} byId={byId} />
              ))
            ) : (
              <span className="so-none">roll the transfer</span>
            )}
            <div className="so-xi">
              XI {fmt1(g.xi_points)} pts · C {g.captain}
              {g.chip && <span className="so-chipflag">{CHIP_LONG[g.chip]}</span>}
            </div>
          </div>
          <div className="so-detail-ft">
            {g.chip === 'freehit' ? '—'
              : freeFirst && k === 0 ? `${g.n_transfers}/∞`
                : `${Math.round(g.free_used || 0)}/${Math.round((g.free_used || 0) + (g.free_after || 0))}`}
          </div>
          <div className="so-detail-itb">
            {money(g.bank)}
            {g.hits > 0 && <span className="so-hit"> −{g.hits * 4}</span>}
          </div>
        </div>
      ))}
    </div>
  )
}
