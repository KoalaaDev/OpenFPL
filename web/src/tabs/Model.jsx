import React, { useEffect, useState } from 'react'
import { api } from '../api'
import { useStore } from '../store'
import { Empty, Loading } from '../components/States'
import { LineChart, StatTile, VIZ, VIZ_NEUTRAL } from '../charts'

/* Admin: is the model getting better, and where does it get its information?

   Every number here is produced automatically by the scheduled refresh —
   the post-mortem and the lineup-feed scorecard run for each finished
   gameweek (`scheduler.score_finished_gameweeks`), so nobody has to open a
   terminal for the page to fill in.

   The one thing the page insists on saying out loud: read the trend, not a
   gameweek. A single gameweek's rank correlation moves by more than any real
   improvement does, which is the standing lesson of this whole project. */
export default function Model() {
  const { isAdmin } = useStore()
  const [d, setD] = useState(null)
  const [err, setErr] = useState(null)
  const [busy, setBusy] = useState(false)

  const load = async (force = false) => {
    setBusy(true)
    try { setD(await api.modelHistory(force)); setErr(null) } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }
  useEffect(() => { if (isAdmin) load() }, [isAdmin])   // eslint-disable-line react-hooks/exhaustive-deps

  if (!isAdmin) return <div className="panel"><Empty mark="🔒" title="Admins only">Sign in with an admin account.</Empty></div>
  if (err) return <div className="panel"><Empty mark="!" title="Could not load">{err}</Empty></div>
  if (!d) return <div className="panel"><Loading>Reading the scorecards…</Loading></div>

  const pm = d.postmortems || []
  const last = pm[pm.length - 1]
  const mean = (k, n = pm.length) => {
    const v = pm.slice(-n).map((r) => r[k]).filter((x) => typeof x === 'number')
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null
  }

  return (
    <div className="ops">
      <div className="panel ops-head">
        <div>
          <div className="section-label">Model history</div>
          <h2>{d.season} · {pm.length} gameweek{pm.length === 1 ? '' : 's'} scored</h2>
          <div className="ops-sub">
            Scored automatically after every gameweek. <b>Read the trend, not a
            gameweek</b> — a single week&apos;s rank correlation moves by more than
            any real improvement does.
          </div>
        </div>
        <button className="pill-btn" onClick={() => load(true)} disabled={busy}>
          {busy ? <span className="spinner" /> : '⟳'} Reload
        </button>
      </div>

      {pm.length === 0 ? (
        <div className="panel">
          <Empty mark="📉" title="No gameweeks scored yet">
            The first scorecard is written after the first gameweek finishes
            and the next scheduled refresh runs.
          </Empty>
        </div>
      ) : (
        <>
          <div className="tile-row">
            <StatTile label="Rank correlation" value={fmt(last.spearman, 3)}
              sub={`season mean ${fmt(mean('spearman'), 3)}`} />
            <StatTile label="Top-20 hits" value={`${last.top20_hits ?? '—'}/20`}
              sub={`season mean ${fmt(mean('top20_hits'), 1)}`} />
            <StatTile label="Captain" value={fmt(last.captain_actual, 0)}
              sub={`best possible ${fmt(last.captain_best, 0)}`} />
            <StatTile label="Predicted vs actual"
              value={last.actual_total ? `${Math.round(100 * last.predicted_total / last.actual_total)}%`
                : '—'}
              sub={`${last.predicted_total ?? '—'} vs ${last.actual_total ?? '—'} pts`} />
          </div>

          <div className="panel">
            <div className="panel-head">Rank quality per gameweek
              <span className="panel-sub">Spearman over all players · higher is better</span></div>
            <div className="chart-wrap">
              <LineChart height={220} fmt={(v) => v.toFixed(3)} series={[
                { name: 'Spearman', color: VIZ[0],
                  points: pm.filter((r) => typeof r.spearman === 'number')
                    .map((r) => ({ x: r.gw, y: r.spearman })) },
              ]} />
            </div>
            <div className="fold-note" style={{ padding: '0 14px 12px' }}>
              GW1 is always the weakest week of a season: every trailing window
              is last season&apos;s, and promoted clubs have no top-flight history
              at all.
            </div>
          </div>

          <div className="ops-grid">
            <div className="panel">
              <div className="panel-head">Captain: picked vs best possible</div>
              <div className="chart-wrap">
                <LineChart height={200} fmt={(v) => v.toFixed(0)} series={[
                  { name: 'Our captain', color: VIZ[0],
                    points: pm.filter((r) => typeof r.captain_actual === 'number')
                      .map((r) => ({ x: r.gw, y: r.captain_actual })) },
                  { name: 'Best possible', color: VIZ_NEUTRAL,
                    points: pm.filter((r) => typeof r.captain_best === 'number')
                      .map((r) => ({ x: r.gw, y: r.captain_best })) },
                ]} />
              </div>
              <div className="fold-note" style={{ padding: '0 14px 12px' }}>
                Nobody catches the dotted line. What matters is the gap to the
                crowd&apos;s captain, measured at +0.7 to +1.1 points a week.
              </div>
            </div>

            <div className="panel">
              <div className="panel-head">Component calibration
                <span className="panel-sub">expected ÷ actual, players who lasted 60+</span></div>
              <div className="chart-wrap">
                <LineChart height={200} fmt={(v) => v.toFixed(2)} series={[
                  ratioSeries(pm, 'goals_ratio', 'Goals', VIZ[0]),
                  ratioSeries(pm, 'assists_ratio', 'Assists', VIZ[1]),
                  ratioSeries(pm, 'cs_ratio', 'Clean sheets', VIZ[2]),
                ].filter(Boolean)} />
              </div>
              <div className="fold-note" style={{ padding: '0 14px 12px' }}>
                1.00 is honest; above it the model expected more than the pitch
                delivered. Conditioning on 60+ minutes selects the surprise
                starters, so read it as a level check, not a rate bias.
              </div>
            </div>
          </div>

          <FeedPanel feed={d.feed} />
          <MinutesPanel m={d.minutes} blend={d.blend} stretch={d.market_stretch} />
          <BacktestPanel rows={d.backtests} />
        </>
      )}
    </div>
  )
}

function ratioSeries(pm, key, name, color) {
  const points = pm.filter((r) => typeof r[key] === 'number').map((r) => ({ x: r.gw, y: r[key] }))
  return points.length ? { name, color, points } : null
}

function FeedPanel({ feed }) {
  if (!feed || !feed.per_gw?.length) {
    return (
      <div className="panel">
        <div className="panel-head">Predicted-lineup feed</div>
        <div className="dd-empty">
          Nothing scored yet. The scorecard is written for each finished
          gameweek on the next scheduled refresh.
        </div>
      </div>
    )
  }
  const b = feed.band || {}
  const scored = feed.per_gw.filter((r) => typeof r.feed === 'number')
  return (
    <div className="panel">
      <div className="panel-head">Predicted-lineup feed vs the model
        <span className="panel-sub">in the 0.30–0.70 band, where minutes value lives</span></div>
      <div className="tile-row inner">
        <StatTile label="Feed accuracy" value={fmt(b.feed_accuracy, 3)}
          sub={b.feed_accuracy_ci95 ? `95% CI ${fmt(b.feed_accuracy_ci95[0], 2)}–${fmt(b.feed_accuracy_ci95[1], 2)}` : ''} />
        <StatTile label="Model accuracy" value={fmt(b.model_accuracy, 3)}
          sub="the bar it has to clear" />
        <StatTile label="Band rows" value={b.n ?? '—'}
          sub={feed.rows_needed ? `${feed.rows_needed} needed to price it` : ''} />
        <StatTile label="Implied value"
          value={b.implied_points_per_season != null ? `+${fmt(b.implied_points_per_season, 0)}` : '—'}
          sub={b.implied_points_ci95
            ? `pts/season · CI ${fmt(b.implied_points_ci95[0], 0)}–${fmt(b.implied_points_ci95[1], 0)}`
            : 'pts/season'} />
      </div>
      {scored.length > 1 && (
        <div className="chart-wrap">
          <LineChart height={200} fmt={(v) => v.toFixed(3)} series={[
            { name: 'Feed', color: VIZ[0], points: scored.map((r) => ({ x: r.gw, y: r.feed })) },
            { name: 'Model', color: VIZ_NEUTRAL, points: scored.map((r) => ({ x: r.gw, y: r.model })) },
          ]} />
        </div>
      )}
      <div className="fold-note" style={{ padding: '0 14px 12px' }}>
        {feed.priceable
          ? 'Enough rows have accumulated to price this feed.'
          : 'Not yet priceable — value is linear in the fraction of the band a feed resolves, so it takes eight to ten gameweeks of band rows before the estimate is tight.'}
        {feed.updated ? ` Last scored ${new Date(feed.updated).toLocaleString()}.` : ''}
      </div>
    </div>
  )
}

function MinutesPanel({ m, blend, stretch }) {
  if (!m) return null
  const max = m.importances?.[0]?.gain || 1
  return (
    <div className="ops-grid">
      <div className="panel">
        <div className="panel-head">Where the minutes model gets its information
          <span className="panel-sub">share of total gain</span></div>
        <div className="imp-blocks">
          {Object.entries(m.blocks || {}).map(([name, v]) => (
            <div key={name} className="imp-block">
              <span className="ib-name">{name}</span>
              <span className="ib-bar"><i style={{ width: `${Math.max(1, v * 100)}%` }} /></span>
              <span className="ib-val">{(v * 100).toFixed(1)}%</span>
            </div>
          ))}
          {!Object.keys(m.blocks || {}).length && (
            <div className="dd-empty">{m.importance_error || 'No importances available.'}</div>
          )}
        </div>
        <div className="fold-note" style={{ padding: '0 14px 12px' }}>
          Trailing minutes dominate, and that is the finding rather than a
          defect: six independent attempts at sharpening this model returned
          ≤0.25% of log-loss each. The residual sits where the manager has not
          decided yet.
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">Top features
          <span className="panel-sub">by total gain</span></div>
        <div className="imp-list">
          {(m.importances || []).slice(0, 14).map((r) => (
            <div key={r.feature} className="imp-row">
              <span className="ir-name">{r.feature}</span>
              <span className="ir-bar"><i style={{ width: `${(r.gain / max) * 100}%` }} /></span>
              <span className="ir-val">{(r.gain * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
        <div className="dd-kv" style={{ borderTop: '1px solid var(--line)' }}>
          <Kv k="features" v={m.features} />
          <Kv k="trained on" v={(m.train_seasons || []).join(', ')} />
          <Kv k="holdout" v={`${m.holdout_season || '—'} · accuracy ${fmt(m.holdout_accuracy, 3)}`} />
          <Kv k="xgboost" v={m.xgboost} />
          <Kv k="xPts blend" v={blend ? `weight ${blend.weight} (fitted ${blend.season})` : 'pure OpenFPL'} />
          <Kv k="market stretch" v={stretch
            ? `b=${fmt(stretch.b, 2)} a=${fmt(stretch.a, 2)} · n=${stretch.n ?? '—'} r=${fmt(stretch.r, 2)}`
            : 'not fitted'} />
        </div>
      </div>
    </div>
  )
}

function BacktestPanel({ rows }) {
  if (!rows?.length) return null
  const cols = [['spearman_played', 'rank (played)'], ['p_at_20', 'prec@20'],
                ['top11', 'top-11 pts/pick'], ['top30', 'top-30'],
                ['captain', 'captain'], ['rmse', 'rmse']]
  return (
    <div className="panel">
      <div className="panel-head">Replayed seasons
        <span className="panel-sub">the engine against the baselines it has to beat</span></div>
      <div className="table-wrap">
        <table className="ptable compact">
          <thead>
            <tr><th>Season</th><th>Model</th>{cols.map(([, l]) => <th key={l} className="num">{l}</th>)}</tr>
          </thead>
          <tbody>
            {rows.flatMap((s) => (s.arms || []).map((a) => (
              <tr key={`${s.season}-${a.arm}`} className={a.arm === 'xpts' ? 'lead' : ''}>
                <td>{a.arm === (s.arms[0] || {}).arm ? s.season : ''}</td>
                <td><b>{ARM[a.arm] || a.arm}</b></td>
                {cols.map(([k]) => <td key={k} className="num">{fmt(a[k], k === 'rmse' ? 3 : 3)}</td>)}
              </tr>
            )))}
          </tbody>
        </table>
      </div>
      <div className="fold-note" style={{ padding: '0 14px 12px' }}>
        Forward-in-time replays, committed. These are the season means any
        change has to beat on a <b>paired</b> test — a mean that looks better
        is not a result until the per-gameweek pairing says so.
      </div>
    </div>
  )
}

const ARM = { xpts: 'xPts engine (shipped)', openfpl: 'OpenFPL', ppg: 'points per game', trail4: 'trailing 4 gws' }
function Kv({ k, v }) {
  return <div className="dd-kv-row"><span className="k">{k}</span><span className="v">{v ?? '—'}</span></div>
}
const fmt = (x, d = 2) => (x == null || Number.isNaN(Number(x)) ? '—' : Number(x).toFixed(d))
