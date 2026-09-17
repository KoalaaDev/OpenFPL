import React, { useEffect, useState } from 'react'
import { api } from '../api'
import { useStore, usePersisted } from '../store'
import { Empty, Loading } from '../components/States'
import { LineChart, StatTile, VIZ, VIZ_NEUTRAL } from '../charts'
import PitchLines from '../components/Pitch'
import { shirtUrl } from '../util'

/* Admin: is the model any good, and how would I know?

   Three views, because they answer three different questions:

   * SEASON RECORD — what the model actually picked before each deadline, and
     how that did against the average manager and against the best team that
     was possible in hindsight. The picks come from the projection the site
     showed before the deadline (LIVE) or, for weeks before those snapshots
     began, a strictly point-in-time replay (REPLAY). Every week says which.
   * ACCURACY — is a projection of 5 worth 5? Rank quality, a calibration
     curve, error by position, and the engine's components against reality.
   * INTERNALS — the minutes model, the feeds, the replayed seasons.

   Everything is produced by the scheduled refresh after each gameweek
   (`scheduler.score_finished_gameweeks` -> `app/modelrecord.py`); nobody
   runs a command for this page to fill in. Read the trend, not a week: four
   gameweeks of anything is mostly noise, and the page says so. */
export default function Model() {
  const { isAdmin } = useStore()
  const [d, setD] = useState(null)
  const [err, setErr] = useState(null)
  const [busy, setBusy] = useState(false)
  const [view, setView] = usePersisted('model.view', 'record')

  const load = async (force = false) => {
    setBusy(true)
    try { setD(await api.modelHistory(force)); setErr(null) } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }
  useEffect(() => { if (isAdmin) load() }, [isAdmin])   // eslint-disable-line react-hooks/exhaustive-deps

  if (!isAdmin) return <div className="panel"><Empty mark="🔒" title="Admins only">Sign in with an admin account.</Empty></div>
  if (err) return <div className="panel"><Empty mark="!" title="Could not load">{err}</Empty></div>
  if (!d) return <div className="panel"><Loading>Reading the scorecards…</Loading></div>

  const rec = d.record || { gws: [] }
  const s = rec.season

  return (
    <div className="ops mdl">
      <div className="panel ops-head">
        <div>
          <div className="section-label">Model</div>
          <h2>{d.season} · {s ? `${s.n} gameweek${s.n === 1 ? '' : 's'} on the record` : 'no gameweeks scored yet'}</h2>
          <div className="ops-sub">
            {s ? <>{s.live_weeks} live, {s.n - s.live_weeks} replayed · </> : null}
            scored automatically after every gameweek · <b>read the trend, not a week</b>
          </div>
        </div>
        <span className="seg mdl-seg" role="tablist">
          {[['record', 'Season record'], ['team', "Model's team"], ['accuracy', 'Accuracy'], ['internals', 'Internals']].map(([k, l]) => (
            <button key={k} role="tab" aria-selected={view === k}
              className={view === k ? 'on' : ''} onClick={() => setView(k)}>{l}</button>
          ))}
        </span>
        <button className="pill-btn" onClick={() => load(true)} disabled={busy}>
          {busy ? <span className="spinner" /> : '⟳'} Reload
        </button>
      </div>

      {view === 'record' && <SeasonRecord rec={rec} />}
      {view === 'team' && <ModelTeam team={d.team} rec={rec} />}
      {view === 'accuracy' && <Accuracy rec={rec} feed={d.feed} />}
      {view === 'internals' && (
        <>
          <MinutesPanel m={d.minutes} blend={d.blend} stretch={d.market_stretch} />
          <BacktestPanel rows={d.backtests} />
        </>
      )}
    </div>
  )
}

/* ================================================================ record == */

const signed = (v, dp = 0) => (v == null ? '—' : `${v > 0 ? '+' : ''}${Number(v).toFixed(dp)}`)
const series = (gws, name, color, get) => ({
  name, color,
  points: gws.map((r) => ({ x: r.gw, y: get(r) })).filter((p) => typeof p.y === 'number'),
})

function SeasonRecord({ rec }) {
  const gws = rec.gws || []
  const s = rec.season
  if (!gws.length) {
    return (
      <div className="panel">
        <Empty mark="📉" title="No gameweeks on the record yet">
          The first entry is written after a gameweek finishes and the next
          scheduled refresh runs.
        </Empty>
      </div>
    )
  }
  const diff = s.model_squad != null && s.average != null ? s.model_squad - s.average : null
  let cm = 0, ca = 0, ch = 0
  const cumulative = gws.map((r) => {
    cm += r.model_squad?.actual || 0
    ca += r.fpl?.average || 0
    ch += r.hindsight_squad?.actual || 0
    return { gw: r.gw, model: cm, avg: ca, hind: ch }
  })
  let cx = 0, cxa = 0
  const cumulativeXi = gws.map((r) => {
    cx += r.model_xi?.actual || 0
    cxa += r.fpl?.average || 0
    return { gw: r.gw, xi: cx, avg: cxa }
  })

  return (
    <>
      <div className="tile-row">
        <StatTile label="Model £100m squad" value={`${Math.round(s.model_squad)} pts`}
          delta={diff != null ? `${signed(diff)} vs average` : null} deltaGood={diff >= 0}
          sub={`average manager ${Math.round(s.average)}`} />
        <StatTile label="Weeks above average" value={`${s.weeks_beat_average} / ${s.n}`}
          sub="the £100m squad vs FPL's average score" />
        <StatTile label="Share of the ceiling" value={s.captured != null ? `${Math.round(s.captured * 100)}%` : '—'}
          sub={`best possible £100m: ${Math.round(s.hindsight_squad)} pts`} />
        <StatTile label="Model captain" value={`${Math.round(s.captain.model)} pts`}
          delta={`${signed(s.captain.model - s.captain.crowd)} vs crowd`}
          deltaGood={s.captain.model >= s.captain.crowd}
          sub={`crowd ${Math.round(s.captain.crowd)} · best ${Math.round(s.captain.best)}`} />
        <StatTile label="Unlimited-budget XI" value={`${Math.round(s.model_xi)} pts`}
          sub={`best possible: ${Math.round(s.hindsight_xi)}`} />
      </div>

      <div className="mdl-grid two">
        <div className="panel">
          <div className="panel-head">Points per gameweek
            <span className="panel-sub">the model's picks, scored on what happened</span></div>
          <div className="chart-wrap">
            <LineChart height={250} fmt={(v) => v.toFixed(0)} series={[
              series(gws, 'Best possible £100m', VIZ_NEUTRAL, (r) => r.hindsight_squad?.actual),
              series(gws, 'Highest manager', '#8a91b8', (r) => r.fpl?.highest),
              series(gws, 'Model £100m squad', VIZ[0], (r) => r.model_squad?.actual),
              series(gws, 'Model XI, no budget', VIZ[2], (r) => r.model_xi?.actual),
              series(gws, 'Average manager', VIZ[1], (r) => r.fpl?.average),
            ]} />
          </div>
          <div className="fold-note mdl-note">
            The £100m squad is fifteen players with a bench, scored with FPL&apos;s
            autosubs and the armband passing to the vice — the fair comparison
            with a manager. It is rebuilt from scratch each week, which is a free
            wildcard every week and flatters it; a manager carries last week&apos;s
            team and pays for changes.
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">Margin over the average manager
            <span className="panel-sub">cumulative points above (or below) FPL&apos;s average</span></div>
          {/* the raw cumulative totals sit on top of each other at this scale;
              the gap between them is the thing worth seeing */}
          <div className="chart-wrap">
            <LineChart height={250} fmt={(v) => `${v >= 0 ? '+' : ''}${v.toFixed(0)}`} series={[
              { name: 'Model £100m squad', color: VIZ[0], points: cumulative.map((c) => ({ x: c.gw, y: c.model - c.avg })) },
              { name: 'Model XI, no budget', color: VIZ[2], points: cumulativeXi.map((c) => ({ x: c.gw, y: c.xi - c.avg })) },
            ]} />
          </div>
          <table className="mdl-table">
            <thead><tr><th>GW</th><th>source</th><th className="num">model</th><th className="num">avg</th>
              <th className="num">±</th><th className="num">best</th><th className="num">share</th></tr></thead>
            <tbody>
              {gws.map((r) => {
                const m = r.model_squad?.actual, a = r.fpl?.average, h = r.hindsight_squad?.actual
                return (
                  <tr key={r.gw}>
                    <td>GW{r.gw}</td>
                    <td><SourceBadge r={r} /></td>
                    <td className="num"><b>{m != null ? Math.round(m) : '—'}</b></td>
                    <td className="num">{a ?? '—'}</td>
                    <td className={`num ${m != null && a != null ? (m >= a ? 'up' : 'down') : ''}`}>
                      {m != null && a != null ? signed(m - a) : '—'}</td>
                    <td className="num">{h != null ? Math.round(h) : '—'}</td>
                    <td className="num">{m != null && h ? `${Math.round((m / h) * 100)}%` : '—'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      <GwPicks gws={gws} />

      <div className="mdl-grid two">
        <CaptainPanel gws={gws} />
        <DeltasPanel gws={gws} />
      </div>
    </>
  )
}

/* ============================================================ the team == */

/* The honest benchmark. The season record rebuilds the best squad every week,
   which is a free wildcard every week; this is one squad, bought before GW1
   and carried the way a manager carries it — a free transfer a week, banked
   up to five, -4 for each extra, sold at FPL's selling price, no chips. */
const formationOf = (xi) => ['DEF', 'MID', 'FWD'].map((p) => xi.filter((x) => x.pos === p).length).join('-')

function ModelTeam({ team, rec }) {
  const weeks = team?.weeks || []
  const [gw, setGw] = usePersisted('model.teamGw', null)
  if (!weeks.length) {
    return (
      <div className="panel">
        <Empty mark="🧮" title={team?.error ? 'Could not read the model\'s team' : 'The model\'s team has not played yet'}>
          {team?.error || 'It is built after each finished gameweek by the scheduled refresh.'}
        </Empty>
      </div>
    )
  }
  const t = team.totals
  const w = weeks.find((x) => x.gw === gw) || weeks[weeks.length - 1]
  const rebuild = Object.fromEntries((rec?.gws || []).map((r) => [r.gw, r.model_squad?.actual]))
  let cn = 0, ca = 0, cr = 0
  const cum = weeks.map((x) => {
    cn += x.net; ca += x.average || 0; cr += rebuild[x.gw] ?? x.net
    return { gw: x.gw, net: cn, avg: ca, rebuild: cr }
  })
  const margin = t.net - t.average
  const last = cum[cum.length - 1]
  const sel = {
    xi: w.xi, bench: w.bench, formation: formationOf(w.xi), actual: w.points,
    projected: w.projected, cost: null,
  }

  return (
    <>
      <div className="tile-row">
        <StatTile label="Model's team" value={`${Math.round(t.net)} pts`}
          delta={`${signed(margin)} vs average`} deltaGood={margin >= 0}
          sub={`average manager ${Math.round(t.average)} · after hits`} />
        <StatTile label="Weeks above average" value={`${t.weeks_above_average} / ${weeks.length}`}
          sub="one squad, carried all season" />
        <StatTile label="Transfers" value={t.transfers}
          sub={`${t.hits} hit${t.hits === 1 ? '' : 's'} taken (−${t.hits * 4} pts)`} />
        <StatTile label="Free transfers now" value={team.state?.ft ?? '—'}
          sub={`bank £${(team.state?.bank ?? 0).toFixed(1)}m`} />
        <StatTile label="Cost of carrying a team" value={signed(last.net - last.rebuild)}
          sub="vs rebuilding a fresh £100m squad every week" />
      </div>

      <div className="mdl-grid two">
        <div className="panel">
          <div className="panel-head">Cumulative margin over the average manager
            <span className="panel-sub">points above FPL&apos;s average, after hits</span></div>
          <div className="chart-wrap">
            <LineChart height={240} fmt={(v) => `${v >= 0 ? '+' : ''}${v.toFixed(0)}`} series={[
              { name: "Model's team (transfers)", color: VIZ[0], points: cum.map((c) => ({ x: c.gw, y: c.net - c.avg })) },
              { name: 'Fresh squad every week', color: VIZ_NEUTRAL, points: cum.map((c) => ({ x: c.gw, y: c.rebuild - c.avg })) },
            ]} />
          </div>
          <div className="fold-note mdl-note">
            Decisions use only what the model knew before each deadline — the
            projection the site showed, or a point-in-time replay where no live
            snapshot exists. The grey line is the season record&apos;s weekly
            rebuild: the gap between the two is what transfer limits cost.
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">Week by week</div>
          <div className="mt-scroll">
            <table className="mdl-table">
              <thead><tr><th>GW</th><th>moves</th><th className="num">FT</th><th className="num">hits</th>
                <th className="num">pts</th><th className="num">avg</th><th className="num">±</th></tr></thead>
              <tbody>
                {weeks.map((x) => (
                  <tr key={x.gw} className={x.gw === w.gw ? 'sel' : ''} onClick={() => setGw(x.gw)}>
                    <td>GW{x.gw}</td>
                    <td className="mt-moves">{x.build ? 'squad built'
                      : x.transfers.length ? x.transfers.map((tr) => `${tr.out.name} → ${tr.in.name}`).join(', ')
                        : <span className="muted">rolled</span>}</td>
                    <td className="num">{x.build ? '—' : x.ft_before}</td>
                    <td className={`num ${x.hits ? 'down' : ''}`}>{x.hits ? `−${x.hits * 4}` : ''}</td>
                    <td className="num"><b>{Math.round(x.net)}</b></td>
                    <td className="num">{x.average ?? '—'}</td>
                    <td className={`num ${x.average != null ? (x.net >= x.average ? 'up' : 'down') : ''}`}>
                      {x.average != null ? signed(x.net - x.average) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">The team, gameweek by gameweek
          <span className="seg mdl-gws">
            {weeks.map((x) => (
              <button key={x.gw} className={x.gw === w.gw ? 'on' : ''} onClick={() => setGw(x.gw)}>GW{x.gw}</button>
            ))}
          </span>
        </div>
        <div className="gwp-sum">
          <span className={`src-badge ${w.source === 'live' ? 'live' : 'replay'}`}>{w.source === 'live' ? 'LIVE' : 'REPLAY'}</span>
          <span>GW{w.gw} · {w.build ? 'opening squad' : `${w.ft_before} free transfer${w.ft_before === 1 ? '' : 's'} available`}
            {' '}· bank £{w.bank.toFixed(1)}m · average manager <b>{w.average ?? '—'}</b></span>
        </div>
        <div className="gwp-pitches">
          <MiniPitch title={`GW${w.gw}${w.hits ? ` · −${w.hits * 4} hit` : ''}`} sel={sel} showEp />
          <TransferList w={w} />
        </div>
      </div>

      {team.next && <NextPlan n={team.next} />}
    </>
  )
}

function TransferList({ w, title }) {
  const { teams } = useStore()
  const row = (p, dir) => {
    const tm = teams[String(p.team_id)]
    return (
      <span className={`mt-player ${dir}`}>
        <img className="mt-shirt" alt="" loading="lazy" src={shirtUrl(tm?.code, p.pos === 'GK')}
          onError={(e) => { e.currentTarget.style.visibility = 'hidden' }} />
        <span className="mt-name">{p.name}</span>
        <span className="mt-meta">{tm?.short ?? ''} · {p.pos} · £{p.price.toFixed(1)}m</span>
      </span>
    )
  }
  return (
    <div className="gwp-col mt-transfers">
      <div className="gwp-head"><b>{title || 'Transfers'}</b>
        <span className="gwp-score muted">
          {w.build ? 'fifteen bought from £100m'
            : `${w.free_used} free${w.hits ? ` · ${w.hits} extra at −4` : ''}`}
        </span>
      </div>
      {w.build ? (
        <div className="dd-empty">The opening squad — no transfers to make before GW1.</div>
      ) : w.transfers.length ? (
        <div className="mt-list">
          {w.transfers.map((tr, i) => (
            <div className="mt-row" key={i}>
              {row(tr.out, 'out')}
              <span className="mt-arrow" aria-hidden="true">→</span>
              {row(tr.in, 'in')}
              <span className="mt-gain" title="projected points over the planning horizon, in minus out">
                {signed(tr.in.ep_horizon - tr.out.ep_horizon, 1)} xP
              </span>
            </div>
          ))}
        </div>
      ) : (
        <div className="dd-empty">No transfers — the free transfer was rolled
          {w.ft_before < 5 ? ` (${Math.min(5, w.ft_before + 1)} next week)` : ''}.</div>
      )}
      {w.captain != null && (
        <div className="mt-cap">Captain <b>{(w.xi.find((p) => p.captain) || {}).name}</b>
          {w.vice != null && <> · vice {(w.xi.find((p) => p.vice) || {}).name}</>}
          {' '}· projected <b>{w.projected}</b></div>
      )}
    </div>
  )
}

function NextPlan({ n }) {
  return (
    <div className="panel">
      <div className="panel-head">Next deadline · GW{n.gw}
        <span className="panel-sub">what the model&apos;s team will do, from the projections on the site now</span></div>
      <div className="gwp-pitches">
        <TransferList w={n} title={n.transfers.length ? 'Planned transfers' : 'Plan'} />
        <div className="gwp-col">
          <div className="gwp-head"><b>Starting XI</b><span className="gwp-form">{formationOf(n.xi)}</span></div>
          <div className="mt-xi">
            {['GK', 'DEF', 'MID', 'FWD'].map((pos) => (
              <div key={pos}><span className="muted">{pos}</span>{' '}
                {n.xi.filter((p) => p.pos === pos).map((p) => (
                  <span key={p.player_id} className="mt-xi-p">{p.name}{p.captain ? ' (C)' : p.vice ? ' (V)' : ''} <b>{p.ep.toFixed(1)}</b></span>
                ))}
              </div>
            ))}
            <div><span className="muted">Bench</span>{' '}
              {n.bench.map((p) => <span key={p.player_id} className="mt-xi-p">{p.name} <b>{p.ep.toFixed(1)}</b></span>)}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function SourceBadge({ r }) {
  const live = r.source === 'live'
  const title = live
    ? `the projection the site showed before the deadline${r.snapshot_built_at
      ? ` (built ${new Date(r.snapshot_built_at * 1000).toLocaleString()})` : ''}`
    : `a point-in-time replay; ${r.availability_at_deadline
      ? 'injury flags as they stood at the deadline'
      : 'NO injury flags — the change log starts after this deadline'}`
  return (
    <span className={`src-badge ${live ? 'live' : 'replay'} ${!live && !r.availability_at_deadline ? 'noavail' : ''}`}
      title={title}>
      {live ? 'LIVE' : 'REPLAY'}{!live && !r.availability_at_deadline ? '*' : ''}
    </span>
  )
}

/* One gameweek at a time: the model's team beside the best team that was
   possible, under the same rules, so the gap is a like-for-like number. */
function GwPicks({ gws }) {
  const [gw, setGw] = usePersisted('model.pickGw', null)
  const [mode, setMode] = usePersisted('model.pickMode', 'squad')
  const r = gws.find((x) => x.gw === gw) || gws[gws.length - 1]
  const mine = mode === 'squad' ? r.model_squad : r.model_xi
  const best = mode === 'squad' ? r.hindsight_squad : r.hindsight_xi
  return (
    <div className="panel">
      <div className="panel-head">The model&apos;s team vs the best possible
        <span className="seg mdl-gws">
          {gws.map((x) => (
            <button key={x.gw} className={x.gw === r.gw ? 'on' : ''} onClick={() => setGw(x.gw)}>GW{x.gw}</button>
          ))}
        </span>
        <span className="seg">
          <button className={mode === 'squad' ? 'on' : ''} onClick={() => setMode('squad')}>£100m squad</button>
          <button className={mode === 'xi' ? 'on' : ''} onClick={() => setMode('xi')}>No budget</button>
        </span>
      </div>
      <div className="gwp-sum">
        <SourceBadge r={r} />
        <span>GW{r.gw} · average manager <b>{r.fpl?.average ?? '—'}</b> · highest <b>{r.fpl?.highest ?? '—'}</b></span>
        {r.live_vs_replay_spearman != null && (
          <span className="muted" title="rank agreement between the live projection and a point-in-time replay of the same week">
            live and replay agree ρ {r.live_vs_replay_spearman.toFixed(2)}
          </span>
        )}
      </div>
      <div className="gwp-pitches">
        <MiniPitch title="Model's picks" sel={mine} showEp />
        <MiniPitch title="Best possible (hindsight)" sel={best} />
      </div>
    </div>
  )
}

function MiniPitch({ title, sel, showEp }) {
  const { teams } = useStore()
  if (!sel) return <div className="gwp-col"><div className="dd-empty">Not available for this week.</div></div>
  const rows = ['GK', 'DEF', 'MID', 'FWD'].map((p) => sel.xi.filter((x) => x.pos === p))
  return (
    <div className="gwp-col">
      <div className="gwp-head">
        <b>{title}</b>
        <span className="gwp-form">{sel.formation}</span>
        <span className="gwp-score"><b>{Math.round(sel.actual)}</b> pts
          {showEp && <span className="muted"> · projected {sel.projected}</span>}
          {sel.cost != null && <span className="muted"> · £{sel.cost}m</span>}
        </span>
      </div>
      <div className="bx-pitch gwp-pitch">
        <PitchLines />
        <div className="bx-rows">
          {rows.map((row, i) => (
            <div className="bx-row" key={i}>
              {row.map((p) => {
                const t = teams[String(p.team_id)]
                const shown = p.captain ? p.pts * 2 : p.pts
                const tone = !showEp ? '' : p.pts >= p.ep + 2 ? 'good' : p.pts <= p.ep - 2 ? 'bad' : ''
                return (
                  <div className={`bx-p ${p.mins === 0 ? 'dnp' : ''}`} key={p.player_id}
                    title={`${p.name} · projected ${p.ep} · scored ${p.pts}${p.mins === 0 ? ' · did not play' : ''}`}>
                    {(p.captain || p.vice) && <span className={`bx-arm ${p.vice ? 'vice' : ''}`}>{p.captain ? 'C' : 'V'}</span>}
                    <img className="bx-shirt" alt="" loading="lazy" src={shirtUrl(t?.code, p.pos === 'GK')}
                      onError={(e) => { e.currentTarget.style.visibility = 'hidden' }} />
                    <span className="bx-name">{p.name}</span>
                    <span className="bx-line">
                      {showEp && <i className="gwp-ep">{p.ep.toFixed(1)}</i>}
                      <b className={`gwp-pts ${tone}`}>{shown}</b>
                    </span>
                  </div>
                )
              })}
            </div>
          ))}
        </div>
      </div>
      {sel.bench && (
        <div className="gwp-bench">
          Bench {sel.bench.map((p) => (
            <span key={p.player_id} className={p.mins === 0 ? 'dnp' : ''}>{p.name} <b>{p.pts}</b></span>
          ))}
        </div>
      )}
    </div>
  )
}

function CaptainPanel({ gws }) {
  return (
    <div className="panel">
      <div className="panel-head">Captaincy
        <span className="panel-sub">the model&apos;s armband vs the crowd&apos;s most-captained player</span></div>
      <table className="mdl-table">
        <thead><tr><th>GW</th><th>Model</th><th className="num">pts</th><th>Crowd</th><th className="num">pts</th><th>Best</th><th className="num">pts</th></tr></thead>
        <tbody>
          {gws.map((r) => {
            const c = r.captain || {}
            const m = c.model?.pts, k = c.crowd?.pts
            return (
              <tr key={r.gw}>
                <td>GW{r.gw}</td>
                <td>{c.model?.name || '—'}</td>
                <td className={`num ${m != null && k != null ? (m > k ? 'up' : m < k ? 'down' : '') : ''}`}><b>{m ?? '—'}</b></td>
                <td>{c.crowd?.name || '—'}</td>
                <td className="num">{k ?? '—'}</td>
                <td className="muted">{c.best?.name}</td>
                <td className="num muted">{c.best?.pts}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <div className="fold-note mdl-note">
        Points shown are the player&apos;s own score, before doubling. Over two
        replayed seasons the model&apos;s captain beat the crowd&apos;s by +0.7 to +1.1
        a week — far too small to see in a handful of gameweeks.
      </div>
    </div>
  )
}

function DeltasPanel({ gws }) {
  // the model XI's biggest surprises across the season, both directions
  const all = gws.flatMap((r) => (r.xi_deltas || []).map((x) => ({ ...x, gw: r.gw })))
  const worst = [...all].sort((a, b) => a.delta - b.delta).slice(0, 6)
  const best = [...all].sort((a, b) => b.delta - a.delta).slice(0, 6)
  const Row = ({ x }) => (
    <div className="live-row tight">
      <span className={`num mv ${x.delta >= 0 ? 'up' : 'down'}`}>{signed(x.delta, 1)}</span>
      <div className="lr-text"><b>{x.name}</b> <span className="muted">GW{x.gw} · {x.pos}</span></div>
      <span className="num muted">{x.ep.toFixed(1)} → {x.pts}</span>
    </div>
  )
  return (
    <div className="panel">
      <div className="panel-head">Biggest surprises in the model&apos;s XI
        <span className="panel-sub">projected → scored</span></div>
      <div className="live-two">
        <div><div className="lt-head up">Delivered more</div>{best.map((x, i) => <Row key={i} x={x} />)}</div>
        <div><div className="lt-head down">Delivered less</div>{worst.map((x, i) => <Row key={i} x={x} />)}</div>
      </div>
    </div>
  )
}

/* ============================================================== accuracy == */

const COMPONENT_LABEL = {
  goals: 'Goals', assists: 'Assists', clean_sheets: 'Clean sheets (GK/DEF)',
  team_clean_sheets: 'Team clean sheets', defcon_crossings: 'DefCon crossings', bonus: 'Bonus points',
}

function Accuracy({ rec, feed }) {
  const gws = rec.gws || []
  const s = rec.season
  if (!gws.length) return <div className="panel"><Empty mark="📉" title="Nothing scored yet" /></div>
  return (
    <>
      <div className="mdl-grid two">
        <div className="panel">
          <div className="panel-head">Rank quality per gameweek
            <span className="panel-sub">Spearman, projection vs points · higher is better</span></div>
          <div className="chart-wrap">
            <LineChart height={220} fmt={(v) => v.toFixed(2)} series={[
              series(gws, 'All players', VIZ[0], (r) => r.accuracy?.spearman),
              series(gws, 'Players who played', VIZ[2], (r) => r.accuracy?.spearman_played),
            ]} />
          </div>
          <div className="fold-note mdl-note">
            &quot;All players&quot; is mostly the model knowing who plays at all.
            &quot;Players who played&quot; is the harder, more useful number: over
            two replayed seasons it sat near 0.38, and a PERFECT minutes model only
            lifts it to about 0.59 — points are that noisy.
          </div>
        </div>
        <Reliability bins={s.calibration} />
      </div>

      <div className="mdl-grid two">
        <ComponentPanel comps={s.components} gws={gws} />
        <PositionPanel byPos={s.by_pos} gws={gws} />
      </div>

      <FeedPanel feed={feed} />
    </>
  )
}

/* Is a projection of 5 worth 5? Each dot is every player-gameweek whose
   projection fell in a band: across, the average projection; up, the average
   score. On the diagonal is honest; above it the model was too cautious. */
function Reliability({ bins }) {
  const pts = (bins || []).filter((b) => b.n > 0)
    .map((b) => ({ ...b, x: b.pred / b.n, y: b.act / b.n }))
  const max = Math.max(8, ...pts.map((p) => Math.max(p.x, p.y))) * 1.05
  const W = 360, H = 260, pad = 34
  const sx = (v) => pad + (v / max) * (W - pad - 12)
  const sy = (v) => H - pad - (v / max) * (H - pad - 12)
  const nmax = Math.max(1, ...pts.map((p) => p.n))
  const ticks = [0, 2, 4, 6, 8, 10].filter((t) => t <= max)
  return (
    <div className="panel">
      <div className="panel-head">Calibration
        <span className="panel-sub">average projection vs average score, pooled</span></div>
      <div className="chart-wrap rel">
        <svg viewBox={`0 0 ${W} ${H}`} className="rel-svg" role="img"
          aria-label="calibration: average projection against average points by projection band">
          {ticks.map((t) => (
            <g key={t}>
              <line x1={sx(0)} x2={sx(max)} y1={sy(t)} y2={sy(t)} className="rel-grid" />
              <text x={pad - 6} y={sy(t) + 3} className="rel-tick" textAnchor="end">{t}</text>
              <text x={sx(t)} y={H - pad + 14} className="rel-tick" textAnchor="middle">{t}</text>
            </g>
          ))}
          <line x1={sx(0)} y1={sy(0)} x2={sx(max)} y2={sy(max)} className="rel-diag" />
          {pts.map((p) => (
            <g key={p.lo}>
              <circle cx={sx(p.x)} cy={sy(p.y)} r={4 + 9 * Math.sqrt(p.n / nmax)} className="rel-dot">
                <title>{`projected ${p.lo}–${p.hi}: ${p.n} player-gameweeks · average projection ${p.x.toFixed(2)} · average score ${p.y.toFixed(2)}`}</title>
              </circle>
            </g>
          ))}
          <text x={W / 2} y={H - 4} className="rel-axis" textAnchor="middle">average projection</text>
          <text x={10} y={H / 2} className="rel-axis" textAnchor="middle" transform={`rotate(-90 10 ${H / 2})`}>average points</text>
        </svg>
      </div>
      <table className="mdl-table">
        <thead><tr><th>projected</th><th className="num">n</th><th className="num">avg proj</th><th className="num">avg pts</th><th className="num">ratio</th></tr></thead>
        <tbody>
          {pts.map((p) => (
            <tr key={p.lo}>
              <td>{p.lo}{p.hi < 99 ? `–${p.hi}` : '+'}</td>
              <td className="num muted">{p.n}</td>
              <td className="num">{p.x.toFixed(2)}</td>
              <td className="num"><b>{p.y.toFixed(2)}</b></td>
              <td className={`num ${Math.abs(p.y / p.x - 1) > 0.25 ? 'warn' : ''}`}>{(p.y / p.x).toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function ComponentPanel({ comps, gws }) {
  const keys = Object.keys(COMPONENT_LABEL).filter((k) => comps?.[k])
  return (
    <div className="panel">
      <div className="panel-head">Components vs what happened
        <span className="panel-sub">expected ÷ actual, pooled over {gws.length} gameweeks</span></div>
      <div className="cmp-list">
        {keys.map((k) => {
          const c = comps[k]
          const ratio = c.act ? c.pred / c.act : null
          const max = Math.max(c.pred, c.act) || 1
          return (
            <div className="cmp-row" key={k}>
              <span className="cmp-name">{COMPONENT_LABEL[k]}</span>
              <span className="cmp-bars">
                <i className="exp" style={{ width: `${(c.pred / max) * 100}%` }} title={`expected ${c.pred}`} />
                <i className="act" style={{ width: `${(c.act / max) * 100}%` }} title={`actual ${c.act}`} />
              </span>
              <span className="cmp-nums">{c.pred.toFixed(0)} / {c.act.toFixed(0)}</span>
              <span className={`cmp-ratio ${ratio != null && Math.abs(ratio - 1) > 0.15 ? 'warn' : ''}`}>
                {ratio != null ? ratio.toFixed(2) : '—'}
              </span>
            </div>
          )
        })}
      </div>
      <div className="cmp-legend"><i className="exp" /> expected <i className="act" /> actual · ratio above 1 = the model expected more</div>
      <table className="mdl-table">
        <thead><tr><th>GW</th>{keys.map((k) => <th key={k} className="num">{COMPONENT_LABEL[k].split(' ')[0]}</th>)}</tr></thead>
        <tbody>
          {gws.map((r) => (
            <tr key={r.gw}>
              <td>GW{r.gw}</td>
              {keys.map((k) => {
                const c = r.components?.[k]
                return <td key={k} className="num">{c ? `${c.pred.toFixed(0)}/${c.act.toFixed(0)}` : '—'}</td>
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="fold-note mdl-note">
        Expected vs actual counts across every player. A ratio flagged amber is
        more than 15% off — but a single high-scoring weekend moves these a lot,
        so treat anything short of ten gameweeks as a hint, not a finding.
      </div>
    </div>
  )
}

function PositionPanel({ byPos, gws }) {
  return (
    <div className="panel">
      <div className="panel-head">Error by position
        <span className="panel-sub">players who played · pooled</span></div>
      <table className="mdl-table">
        <thead><tr><th>Position</th><th className="num">player-gws</th><th className="num">avg error</th><th className="num">bias</th></tr></thead>
        <tbody>
          {['GK', 'DEF', 'MID', 'FWD'].map((p) => {
            const v = byPos?.[p]
            return (
              <tr key={p}>
                <td><b>{p}</b></td>
                <td className="num muted">{v?.n ?? '—'}</td>
                <td className="num">{v?.mae ?? '—'}</td>
                <td className={`num ${v && v.bias < 0 ? 'down' : 'up'}`}>{v ? signed(v.bias, 2) : '—'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <div className="fold-note mdl-note">
        Bias among players who PLAYED is negative by construction: being on the
        pitch is itself good news the projection could not know. Compare
        positions with each other, not with zero.
      </div>
      <table className="mdl-table">
        <thead><tr><th>GW</th><th className="num">top-20 found</th><th className="num">avg error</th><th className="num">bias</th><th className="num">played</th></tr></thead>
        <tbody>
          {gws.map((r) => (
            <tr key={r.gw}>
              <td>GW{r.gw}</td>
              <td className="num"><b>{r.accuracy?.top20_hits}</b>/20</td>
              <td className="num">{r.accuracy?.mae_played}</td>
              <td className="num">{signed(r.accuracy?.bias_played, 2)}</td>
              <td className="num muted">{r.accuracy?.n_played}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
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
