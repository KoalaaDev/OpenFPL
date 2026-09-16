import React, { useEffect, useState } from 'react'
import { api } from '../api'
import { useStore } from '../store'
import { Empty, Loading } from '../components/States'

/* The operator's desk.

   It used to be six stacked lists — scheduler keys, the model's metadata,
   team news, Friday quotes, predicted XIs and projection movers — with no
   hierarchy, which made the one thing it exists for (is the pipeline healthy
   before a deadline?) as hard to find as everything else. The feeds are the
   reader's business and now live on the **Live** tab, where everyone can see
   them; what is left here is what only an operator can act on:

     * three health checks, each green or not, with the reason
     * the model that is actually serving
     * market coverage per fixture, the failure that cost GW1-4 of 2026-27
     * the running lineup-feed scorecard

   Nothing here changes a projection. */
export default function Deadline() {
  const { isAdmin } = useStore()
  const [d, setD] = useState(null)
  const [err, setErr] = useState(null)
  const [busy, setBusy] = useState(false)

  const load = async (force = false) => {
    setBusy(true)
    try { setD(await api.deadline(force)); setErr(null) } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }
  useEffect(() => { if (isAdmin) load() }, [isAdmin])   // eslint-disable-line react-hooks/exhaustive-deps

  if (!isAdmin) return <div className="panel"><Empty mark="🔒" title="Admins only">Sign in with an admin account to see the operator desk.</Empty></div>
  if (err) return <div className="panel"><Empty mark="!" title="Could not load">{err}</Empty></div>
  if (!d) return <div className="panel"><Loading>Assembling the desk…</Loading></div>

  const dl = d.deadline ? new Date(d.deadline) : null
  const hrs = dl ? (dl - Date.now()) / 36e5 : null
  const r = d.refresh || {}
  const m = d.model || {}
  const mk = d.market || {}
  // a rebuild that changed nothing still returns every player at delta 0.00
  const movers = (d.movers?.rows || []).filter((p) => Math.abs(p.delta) >= 0.05)

  /* the three things that have actually broken here before */
  const projAge = r.proj_updated_at ? (Date.now() / 1000 - r.proj_updated_at) / 3600 : null
  const checks = [
    {
      key: 'refresh',
      label: 'Scheduled refresh',
      ok: !r.last_error && r.last_ok && (Date.now() / 1000 - r.last_ok) < 36 * 3600,
      value: r.last_ok ? `succeeded ${ago(r.last_ok)}` : 'never succeeded',
      detail: r.last_error ? String(r.last_error).slice(0, 200)
        : r.next_run ? `next ${new Date(r.next_run * 1000).toLocaleString()} (${r.next_reason || 'scheduled'})`
          : 'no next run scheduled',
    },
    {
      key: 'proj',
      label: 'Projections',
      ok: projAge != null && projAge < 30,
      value: projAge == null ? 'never built' : `built ${ago(r.proj_updated_at)}`,
      detail: `gameweeks ${(r.projected_gws || []).join(', ') || 'none'}`,
    },
    {
      key: 'market',
      label: 'Market prices',
      // a rejected odds key is skipped silently by the pull: this is the row
      // that makes a whole gameweek of unpriced fixtures visible
      ok: !mk.warning && (mk.counts?.bookmaker ?? 0) > 0,
      value: mk.counts
        ? `${mk.counts.bookmaker ?? 0} bookmaker · ${mk.counts.polymarket ?? 0} Polymarket · ${mk.counts.none ?? 0} unpriced`
        : 'unknown',
      detail: mk.warning || `GW${mk.gw ?? d.gw} fixtures`,
    },
  ]

  return (
    <div className="ops">
      <div className="panel ops-head">
        <div>
          <div className="section-label">Operator desk</div>
          <h2>GW{d.gw}{dl ? ` · deadline ${dl.toLocaleString()}` : ''}</h2>
          <div className="ops-sub">
            {hrs != null && (hrs >= 0 ? `in ${fmtHrs(hrs)}` : `${fmtHrs(-hrs)} ago`)}
            {d.gw_in_progress ? ' · previous gameweek in progress' : ''}
            {' · '}team news, manager quotes and predicted XIs are on the Live tab;
            accuracy over time is on Model
          </div>
        </div>
        <button className="pill-btn" onClick={() => load(true)} disabled={busy}>
          {busy ? <span className="spinner" /> : '⟳'} Recompute
        </button>
      </div>

      <div className="ops-checks">
        {checks.map((c) => (
          <div key={c.key} className={`ops-check ${c.ok ? 'ok' : 'bad'}`}>
            <span className="oc-mark" aria-hidden="true">{c.ok ? '✓' : '!'}</span>
            <div className="oc-text">
              <div className="oc-label">{c.label}</div>
              <div className="oc-value">{c.value}</div>
              <div className="oc-detail">{c.detail}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="ops-grid">
        <div className="panel">
          <div className="panel-head">Serving model</div>
          <div className="dd-kv">
            <Kv k="minutes model" v={m.minutes
              ? `${m.minutes.features} features · trained ${ago(Date.parse(m.minutes.trained_at) / 1000)}`
              : 'not trained'} />
            <Kv k="training seasons" v={m.minutes ? (m.minutes.train_seasons || []).join(', ') : '—'} />
            <Kv k="holdout" v={m.minutes
              ? `${m.minutes.holdout_season} · accuracy ${fmt(m.minutes.holdout_accuracy, 3)}`
              : '—'} />
            <Kv k="role blocks" v={m.minutes
              ? `BBC ${m.minutes.blocks.bbc_role} · Understat ${m.minutes.blocks.understat_line}`
              : '—'} />
            <Kv k="xPts blend" v={m.blend
              ? `weight ${m.blend.xpts_weight} (fitted ${m.blend.fitted_on})` : 'pure OpenFPL'} />
            <Kv k="rates" v={m.rates} />
            <Kv k="jobs running" v={(r.jobs_running || []).join(', ') || 'none'} />
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">Market coverage
            <span className="panel-sub">per fixture, GW{mk.gw ?? d.gw}</span></div>
          <div className="live-list">
            {!(mk.rows || []).length && <div className="dd-empty">No fixtures found.</div>}
            {(mk.rows || []).map((row, i) => (
              <div key={i} className="live-row tight">
                <span className={`presser-tag ${row.source === 'bookmaker' ? 'available'
                  : row.source === 'polymarket' ? 'doubt' : 'out'}`}>{row.source}</span>
                <div className="lr-text">{row.fixture}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="ops-grid">
        <div className="panel">
          <div className="panel-head">Lineup-feed scorecard
            <span className="panel-sub">RotoWire vs the model, ambiguous band</span></div>
          {d.feed_test?.band?.n ? (
            <div className="dd-kv">
              <Kv k="feed accuracy" v={fmt(d.feed_test.band.feed_accuracy, 3)} />
              <Kv k="model accuracy" v={fmt(d.feed_test.band.model_accuracy, 3)} />
              <Kv k="band rows" v={`${d.feed_test.band.n}${d.feed_test.rows_needed ? ` of ${d.feed_test.rows_needed} needed` : ''}`} />
              <Kv k="implied value" v={d.feed_test.band.implied_points_per_season != null
                ? `+${fmt(d.feed_test.band.implied_points_per_season, 0)} pts/season` : '—'} />
              <Kv k="gameweeks scored" v={(d.feed_test.gws || []).join(', ') || '—'} />
              <Kv k="last scored" v={d.feed_test.updated
                ? new Date(d.feed_test.updated).toLocaleString() : '—'} />
            </div>
          ) : (
            <div className="dd-empty">
              Nothing scored yet. Each finished gameweek is scored on the next
              scheduled refresh — no command to run.
            </div>
          )}
        </div>

        <div className="panel">
          <div className="panel-head">Biggest projection moves
            {d.movers.from && <span className="panel-sub">{ago(d.movers.from)} → {ago(d.movers.to)}</span>}</div>
          <div className="live-list">
            {!d.movers.rows.length && <div className="dd-empty">Needs two builds to compare.</div>}
            {d.movers.rows.length > 0 && !movers.length && (
              <div className="dd-empty">Nothing moved by more than 0.05 points
                since the previous build.</div>
            )}
            {movers.map((p) => (
              <div key={p.player_id} className="live-row tight">
                <span className={`num mv ${p.delta >= 0 ? 'up' : 'down'}`}>
                  {p.delta >= 0 ? '+' : ''}{p.delta.toFixed(2)}
                </span>
                <div className="lr-text"><b>{p.name}</b>
                  <span className="muted"> {p.team} · {p.pos}</span></div>
                <span className="num muted">{p.before.toFixed(2)} → {p.after.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

function Kv({ k, v, warn }) {
  return <div className={`dd-kv-row ${warn ? 'warn' : ''}`}><span className="k">{k}</span><span className="v">{v ?? '—'}</span></div>
}
const fmt = (x, d = 2) => (x == null || Number.isNaN(Number(x)) ? '—' : Number(x).toFixed(d))
const fmtHrs = (h) => (h < 1 ? `${Math.round(h * 60)} min` : `${h.toFixed(h < 10 ? 1 : 0)} h`)
function ago(ts) {
  if (!ts) return '—'
  const s = Math.max(0, Date.now() / 1000 - ts)
  if (s < 90) return 'just now'
  if (s < 3600) return `${Math.round(s / 60)} min ago`
  if (s < 86400) return `${Math.round(s / 3600)} h ago`
  return `${Math.round(s / 86400)} d ago`
}
