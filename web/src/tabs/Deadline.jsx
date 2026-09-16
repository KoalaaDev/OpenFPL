import React, { useEffect, useState } from 'react'
import { api } from '../api'
import { useStore } from '../store'
import { Empty, Loading } from '../components/States'

/* Admin only: what the model is thinking before the deadline, and why —
   the live feeds (FPL news, Friday press conferences, RotoWire XIs) beside
   the model's own numbers, the state of the last refresh, and the model
   that is actually serving. Nothing here changes a projection; it is the
   operator's dashboard, so the change of mind is visible before it costs. */
export default function Deadline() {
  const { isAdmin, setToast } = useStore()
  const [d, setD] = useState(null)
  const [err, setErr] = useState(null)
  const [busy, setBusy] = useState(false)

  const load = async (force = false) => {
    setBusy(true)
    try { setD(await api.deadline(force)); setErr(null) } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }
  useEffect(() => { if (isAdmin) load() }, [isAdmin])   // eslint-disable-line react-hooks/exhaustive-deps

  if (!isAdmin) return <div className="panel"><Empty mark="🔒" title="Admins only">Sign in with an admin account to see the deadline desk.</Empty></div>
  if (err) return <div className="panel"><Empty mark="!" title="Could not load">{err}</Empty></div>
  if (!d) return <div className="panel"><Loading>Assembling the deadline desk…</Loading></div>

  const dl = d.deadline ? new Date(d.deadline) : null
  const hrs = dl ? Math.round((dl - Date.now()) / 36e5) : null
  const r = d.refresh || {}
  const m = d.model || {}

  return (
    <div className="dd-grid">
      <div className="panel dd-head">
        <div>
          <div className="section-label">Deadline desk</div>
          <h2>GW{d.gw}{dl ? ` · deadline ${dl.toLocaleString()}` : ''}{hrs != null ? ` · ${hrs >= 0 ? `in ${hrs} h` : `${-hrs} h ago`}` : ''}</h2>
          {d.gw_in_progress && <span className="chip gold">previous gameweek in progress</span>}
        </div>
        <button className="pill-btn" onClick={() => load(true)} disabled={busy}>{busy ? <span className="spinner" /> : '⟳'} Recompute</button>
      </div>

      <div className="panel">
        <div className="panel-head">Refresh & model</div>
        <div className="dd-kv">
          <Kv k="last refresh" v={ago(r.last_run)} />
          <Kv k="last success" v={ago(r.last_ok)} />
          <Kv k="next scheduled" v={r.next_run ? `${new Date(r.next_run * 1000).toLocaleString()} (${r.next_reason || ''})` : '—'} />
          <Kv k="projections built" v={ago(r.proj_updated_at)} />
          <Kv k="projected GWs" v={(r.projected_gws || []).join(', ') || '—'} />
          <Kv k="jobs running" v={(r.jobs_running || []).join(', ') || 'none'} />
          {r.last_error && <Kv k="last error" v={String(r.last_error).slice(0, 160)} warn />}
          <Kv k="minutes model" v={m.minutes ? `${m.minutes.features} features · trained on ${(m.minutes.train_seasons || []).join(', ')} · holdout ${m.minutes.holdout_season} acc ${fmt(m.minutes.holdout_accuracy, 3)} · ${ago(Date.parse(m.minutes.trained_at) / 1000)}` : 'not trained'} />
          <Kv k="role blocks" v={m.minutes ? `BBC ${m.minutes.blocks.bbc_role} · Understat ${m.minutes.blocks.understat_line}` : '—'} />
          <Kv k="xPts blend" v={m.blend ? `weight ${m.blend.xpts_weight} (fitted ${m.blend.fitted_on})` : 'pure OpenFPL'} />
          <Kv k="rates" v={m.rates} />
          {d.market && <Kv k="market prices" v={`GW${d.market.gw}: bookmaker ${d.market.counts?.bookmaker ?? 0} · Polymarket ${d.market.counts?.polymarket ?? 0} · none ${d.market.counts?.none ?? 0}`} warn={!!d.market.warning} />}
          {d.market?.warning && <Kv k="market warning" v={d.market.warning} warn />}
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">FPL team news · last 7 days <span className="chip dim num">{d.news.length}</span></div>
        <div className="dd-list">
          {d.news.length === 0 && <div className="dd-empty">No status changes recorded.</div>}
          {d.news.map((n) => (
            <div key={n.player_id} className="dd-row">
              <span className={`presser-tag ${n.status === 'a' ? 'available' : n.status === 'd' ? 'doubt' : 'out'}`}>{n.status}{n.chance != null ? ` ${Math.round(n.chance * (n.chance <= 1 ? 100 : 1))}%` : ''}</span>
              <b>{n.name}</b><span className="muted">{n.team} · {n.pos}</span>
              <span className="dd-text">{n.news || '—'}</span>
              <span className="dd-when">{ago(Date.parse(n.observed) / 1000)}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">Managers said (BBC, Friday) <span className="chip dim num">{d.pressers.length}</span></div>
        <div className="dd-list">
          {d.pressers.length === 0 && <div className="dd-empty">No press-conference page archived for GW{d.gw} yet — it is collected on the pre-deadline refresh.</div>}
          {d.pressers.map((p, i) => (
            <div key={i} className="dd-row">
              <span className={`presser-tag ${p.cls}`}>{p.cls}</span>
              <b>{p.name}</b><span className="muted">{p.team}</span>
              <span className="dd-text" title={p.snippet}>“{p.phrase}” — {p.snippet}</span>
              <span className="dd-when">{new Date(p.when).toLocaleString(undefined, { weekday: 'short', hour: '2-digit', minute: '2-digit' })}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">Predicted XIs (RotoWire) vs model P(start)
          <span className="chip dim num">{(d.lineups.clubs || []).length} clubs</span>
          {d.feed_test?.pooled?.band && (
            <span className="muted" style={{ marginLeft: 'auto', fontSize: 11 }}>
              feed {fmt(d.feed_test.pooled.band.feed_acc, 2)} vs model {fmt(d.feed_test.pooled.band.model_acc, 2)} in the ambiguous band
            </span>
          )}
        </div>
        {d.lineups.note && <div className="dd-empty">{d.lineups.note}</div>}
        <div className="dd-clubs">
          {(d.lineups.clubs || []).map((c) => (
            <div key={c.team} className="dd-club">
              <div className="dd-club-head"><b>{c.team}</b><span className="muted">{c.n_resolved}/{c.n_named} resolved · {ago(Date.parse(c.observed) / 1000)}</span>
                {c.disagreements > 0 && <span className="chip gold num">{c.disagreements} disagree</span>}</div>
              {c.rows.map((p) => (
                <div key={p.player_id} className={`dd-xi ${p.disagree ? 'dis' : ''} ${p.predicted ? '' : 'sub'}`}>
                  <span>{p.predicted ? '●' : '○'}</span><span className="nm">{p.name}</span>
                  <span className="num">{p.p_start == null ? '—' : p.p_start.toFixed(2)}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">Biggest moves since the previous build
          {d.movers.from && <span className="muted" style={{ marginLeft: 'auto', fontSize: 11 }}>{ago(d.movers.from)} → {ago(d.movers.to)}</span>}</div>
        <div className="dd-list">
          {d.movers.rows.length === 0 && <div className="dd-empty">Needs two builds to compare.</div>}
          {d.movers.rows.map((p) => (
            <div key={p.player_id} className="dd-row">
              <span className={`num ${p.delta >= 0 ? 'up' : 'down'}`} style={{ minWidth: 52 }}>{p.delta >= 0 ? '+' : ''}{p.delta.toFixed(2)}</span>
              <b>{p.name}</b><span className="muted">{p.team} · {p.pos}</span>
              <span className="dd-text num">{p.before.toFixed(2)} → {p.after.toFixed(2)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function Kv({ k, v, warn }) {
  return <div className={`dd-kv-row ${warn ? 'warn' : ''}`}><span className="k">{k}</span><span className="v">{v ?? '—'}</span></div>
}
const fmt = (x, d = 2) => (x == null || Number.isNaN(Number(x)) ? '—' : Number(x).toFixed(d))
function ago(ts) {
  if (!ts) return '—'
  const s = Math.max(0, Date.now() / 1000 - ts)
  if (s < 90) return 'just now'
  if (s < 3600) return `${Math.round(s / 60)} min ago`
  if (s < 86400) return `${Math.round(s / 3600)} h ago`
  return `${Math.round(s / 86400)} d ago`
}
