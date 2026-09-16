import React, { useEffect, useState } from 'react'
import { api } from '../api'
import { useStore } from '../store'
import { Empty, Loading } from '../components/States'
import { badgeUrl, fmt1, money } from '../util'
import PitchLines from '../components/Pitch'

/* Live deadline coverage.

   The tab only exists inside its window (24 h before a deadline, 6 h after —
   `app/live.py` decides, this renders whatever it says), so everything here
   can assume it is the thing the reader came for. It reads only what the
   pipeline already archives and changes no projection: FPL's own team-news
   change log, Friday's manager quotes, the predicted XIs beside the model's
   own P(start), what the model changed its mind about since the last build,
   and who is about to move price.

   It refreshes every minute while the deadline is still ahead. After it
   passes the page freezes into a "gameweek under way" state and the countdown
   becomes a clock since. */
export default function Live() {
  const { status, isAdmin } = useStore()
  const [d, setD] = useState(null)
  const [err, setErr] = useState(null)
  const [now, setNow] = useState(Date.now())

  const load = () => api.live().then((v) => { setD(v); setErr(null) })
    .catch((e) => setErr(e.message))
  useEffect(() => { load() }, [])          // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000)
    const r = setInterval(load, 60000)
    return () => { clearInterval(t); clearInterval(r) }
  }, [])                                    // eslint-disable-line react-hooks/exhaustive-deps

  const win = d?.window || status?.live
  if (err) return <div className="panel"><Empty mark="!" title="Could not load the live desk">{err}</Empty></div>
  if (!d) return <div className="panel"><Loading>Going live…</Loading></div>
  // outside the window there is nothing to be live about — unless you are the
  // admin, who needs to be able to look at the desk before it matters
  if (!win || (win.phase === 'idle' && !isAdmin)) {
    return (
      <div className="panel">
        <Empty mark="🛰" title="Nothing live right now">
          This desk opens 24 hours before each deadline and stays up for six
          hours after it{win?.deadline ? ` — next one ${new Date(win.deadline * 1000).toLocaleString()}.` : '.'}
        </Empty>
      </div>
    )
  }

  const preview = win.phase === 'idle'
  const open = win.phase !== 'closed'
  const left = win.deadline * 1000 - now
  const since = now - win.deadline * 1000

  return (
    <div className="live">
      <LiveHero gw={d.gw} open={open} preview={preview} left={left} since={since}
        deadline={win.deadline} built={d.built_at} projected={d.proj_updated_at} />

      <Fixtures rows={d.fixtures || []} />

      <div className="live-grid">
        <TeamNews rows={d.news || []} />
        <Pressers rows={d.pressers || []} gw={d.gw} />
      </div>

      <div className="live-grid">
        <Movers m={d.movers || { rows: [] }} />
        <PriceWatch p={d.prices} open={open} />
      </div>

      <Lineups l={d.lineups || { clubs: [] }} />
      <Picks rows={d.picks || []} gw={d.gw} />
    </div>
  )
}

/* ------------------------------------------------------------------ */

function LiveHero({ gw, open, preview, left, since, deadline, built, projected }) {
  // the run-up as a bar: empty a day out, full at the deadline
  const gone = open ? Math.max(0, Math.min(1, 1 - left / (24 * 3600 * 1000))) : 1
  return (
    <div className={`live-hero ${open ? 'open' : 'over'}`}>
      <div className="lh-top">
        <span className={`live-badge ${preview ? 'over' : open ? '' : 'over'}`}>
          <i className="dot" aria-hidden="true" />
          {preview ? 'PREVIEW · OPENS 24 H OUT' : open ? 'LIVE' : 'DEADLINE PASSED'}
        </span>
        <span className="lh-gw">Gameweek {gw}</span>
        <span className="lh-when">{new Date(deadline * 1000).toLocaleString(undefined,
          { weekday: 'long', hour: '2-digit', minute: '2-digit' })}</span>
      </div>

      <div className="lh-clock">{open ? <Countdown ms={left} /> : <Elapsed ms={since} />}</div>
      <div className="lh-sub">
        {!open
          ? 'the gameweek is under way — nothing can be changed now'
          : preview
            ? 'until the deadline. This desk goes on air for everyone with a day to go.'
            : 'until transfers, captain and chips lock for this gameweek'}
      </div>

      {open && <div className="lh-bar"><div style={{ width: `${gone * 100}%` }} /></div>}

      <div className="lh-foot">
        {projected ? <span>Projections rebuilt {ago(projected)}</span> : null}
        {built ? <span>Refreshed {ago(built)} · updates every minute</span> : null}
      </div>
    </div>
  )
}

const pad = (n) => String(n).padStart(2, '0')

function Countdown({ ms }) {
  const s = Math.max(0, Math.floor(ms / 1000))
  return (
    <>
      <b>{pad(Math.floor(s / 3600))}</b><i>h</i>
      <b>{pad(Math.floor((s % 3600) / 60))}</b><i>m</i>
      <b>{pad(s % 60)}</b><i>s</i>
    </>
  )
}

function Elapsed({ ms }) {
  const s = Math.max(0, Math.floor(ms / 1000))
  return (
    <>
      <b>{pad(Math.floor(s / 3600))}</b><i>h</i>
      <b>{pad(Math.floor((s % 3600) / 60))}</b><i>m</i>
      <i className="ago">ago</i>
    </>
  )
}

/* ------------------------------------------------------------------ */

function Fixtures({ rows }) {
  const { teams } = useStore()
  if (!rows.length) return null
  // grouped by kick-off slot, which is how a gameweek actually reads
  const groups = []
  for (const r of rows) {
    const key = r.kickoff ? new Date(r.kickoff).toLocaleString(undefined,
      { weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }) : 'TBC'
    const g = groups.find((x) => x.key === key)
    if (g) g.rows.push(r)
    else groups.push({ key, rows: [r] })
  }
  /* A scoreboard line per side, crest first: you find your club by its badge
     long before you read its name, and a fixed left edge makes a slot of
     three matches scan as a list. */
  const Side = ({ id, short, score, won }) => {
    const t = teams[String(id)]
    return (
      <div className={`fx-side ${won ? 'won' : ''}`}>
        <img className="fx-crest" src={badgeUrl(t?.code)} alt="" loading="lazy"
          onError={(e) => { e.currentTarget.style.visibility = 'hidden' }} />
        <span className="fx-name">{t?.name || short}</span>
        {score != null && <span className="fx-score">{score}</span>}
      </div>
    )
  }
  return (
    <div className="panel">
      <div className="panel-head">Fixtures <span className="chip dim num">{rows.length}</span></div>
      <div className="fx-slots">
        {groups.map((g) => (
          <div className="fx-slot" key={g.key}>
            <div className="fx-when">{g.key}</div>
            {g.rows.map((r) => (
              <div className={`fx-match ${r.finished ? 'done' : ''}`} key={r.fixture_id}>
                <Side id={r.home_id} short={r.home} score={r.score?.[0]}
                  won={r.score && r.score[0] > r.score[1]} />
                <Side id={r.away_id} short={r.away} score={r.score?.[1]}
                  won={r.score && r.score[1] > r.score[0]} />
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}

function TeamNews({ rows }) {
  return (
    <div className="panel">
      <div className="panel-head">Team news <span className="chip dim num">{rows.length}</span>
        <span className="panel-sub">FPL&apos;s own feed, newest first</span></div>
      <div className="live-list">
        {!rows.length && <div className="dd-empty">No status changes in the last week.</div>}
        {rows.map((n) => (
          <div key={n.player_id} className="live-row">
            <span className={`presser-tag ${n.status === 'a' ? 'available' : n.status === 'd' ? 'doubt' : 'out'}`}>
              {n.status === 'a' ? 'fit' : n.status === 'd' ? 'doubt' : 'out'}
              {n.chance != null ? ` ${Math.round(n.chance * (n.chance <= 1 ? 100 : 1))}%` : ''}
            </span>
            <div className="lr-text">
              <b>{n.name}</b> <span className="muted">{n.team} · {n.pos}</span>
              <div className="lr-note">{n.news || 'no note'}</div>
            </div>
            <span className="dd-when">{ago(Date.parse(n.observed) / 1000)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function Pressers({ rows, gw }) {
  return (
    <div className="panel">
      <div className="panel-head">Managers said <span className="chip dim num">{rows.length}</span>
        <span className="panel-sub">Friday press conferences (BBC)</span></div>
      <div className="live-list">
        {!rows.length && (
          <div className="dd-empty">Nothing archived for GW{gw} yet — the Friday
            page is collected on the pre-deadline refresh.</div>
        )}
        {rows.map((p, i) => (
          <div key={i} className="live-row">
            <span className={`presser-tag ${p.cls}`}>{p.cls}</span>
            <div className="lr-text">
              <b>{p.name}</b> <span className="muted">{p.team}</span>
              <div className="lr-note">{p.snippet}</div>
            </div>
            <span className="dd-when">{p.when ? new Date(p.when).toLocaleString(undefined,
              { weekday: 'short', hour: '2-digit', minute: '2-digit' }) : ''}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function Movers({ m }) {
  // a build that changed nothing still returns every player at delta 0.00,
  // which reads as "the model has gone quiet" only if you know to ignore it
  const rows = (m.rows || []).filter((p) => Math.abs(p.delta) >= 0.05)
  return (
    <div className="panel">
      <div className="panel-head">The model changed its mind
        <span className="panel-sub">since the previous build</span></div>
      <div className="live-list">
        {!m.rows.length && <div className="dd-empty">Needs two builds to compare.</div>}
        {m.rows.length > 0 && !rows.length && (
          <div className="dd-empty">Nothing moved by more than 0.05 points since
            the last build — no new team news has landed.</div>
        )}
        {rows.slice(0, 12).map((p) => (
          <div key={p.player_id} className="live-row tight">
            <span className={`num mv ${p.delta >= 0 ? 'up' : 'down'}`}>
              {p.delta >= 0 ? '+' : ''}{p.delta.toFixed(2)}
            </span>
            <div className="lr-text"><b>{p.name}</b> <span className="muted">{p.team} · {p.pos}</span></div>
            <span className="num muted">{p.before.toFixed(1)} → {p.after.toFixed(1)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function PriceWatch({ p, open }) {
  const { byId } = useStore()
  if (!p?.ok) {
    return (
      <div className="panel">
        <div className="panel-head">Price watch</div>
        <div className="dd-empty">{p?.error || 'No price model for this season yet.'}</div>
      </div>
    )
  }
  const row = (r, dir) => {
    const pl = byId.get(r.player_id)
    return (
      <div key={r.player_id} className="live-row tight">
        <span className={`num mv ${dir}`}>
          {dir === 'up' ? '▲' : '▼'} {Math.round((dir === 'up' ? r.p_rise : r.p_fall) * 100)}%
        </span>
        <div className="lr-text"><b>{pl?.web_name || r.player_id}</b>
          <span className="muted"> {money(r.price)}</span></div>
        <span className="num muted" title="what that move is worth in points if you hold him (the Round 7 conversion)">
          {r.points >= 0 ? '+' : ''}{fmt1(r.points)} pts
        </span>
      </div>
    )
  }
  return (
    <div className="panel">
      <div className="panel-head">Price watch
        <span className="panel-sub">{open ? 'before tonight' : 'this gameweek'}</span></div>
      <div className="live-two">
        <div>
          <div className="lt-head up">Most likely to rise</div>
          {p.risers.slice(0, 6).map((r) => row(r, 'up'))}
        </div>
        <div>
          <div className="lt-head down">Most likely to fall</div>
          {p.fallers.slice(0, 6).map((r) => row(r, 'down'))}
        </div>
      </div>
      <div className="fold-note" style={{ padding: '0 12px 10px' }}>
        A price move is worth about 0.2 points once FPL&apos;s half-the-profit
        sell-on rule is accounted for — a tie-breaker between players you
        already rate equally, not a reason to transfer.
      </div>
    </div>
  )
}

function Lineups({ l }) {
  const { teams } = useStore()
  const [open, setOpen] = useState(false)
  const clubs = l.clubs || []
  const dis = clubs.reduce((a, c) => a + (c.disagreements || 0), 0)
  return (
    <div className="panel">
      <div className="panel-head">Predicted line-ups
        <span className="chip dim num">{clubs.length} clubs</span>
        {dis > 0 && <span className="chip gold num">{dis} disagree with the model</span>}
        <button className="pill-btn" style={{ marginLeft: 'auto' }}
          onClick={() => setOpen((o) => !o)}>{open ? 'hide' : 'show'}</button>
      </div>
      {l.note && <div className="dd-empty">{l.note}</div>}
      {open ? (
        <>
          <div className="lu-legend">
            <span><i className="lu-dot" /> model&apos;s chance he starts</span>
            <span><i className="lu-dot dis" /> the model disagrees with the feed</span>
            <span><i className="lu-dot unk" /> not matched to an FPL player</span>
          </div>
          <div className="lu-grid">
            {clubs.map((c) => <XiCard key={c.team} c={c} team={teams[String(c.team_id)]} />)}
          </div>
        </>
      ) : (
        <div className="fold-note" style={{ padding: '0 12px 12px' }}>
          A predicted XI is somebody&apos;s forecast, not the team sheet — the real
          one lands about an hour before kick-off, after this deadline. Where a
          feed disagrees with the model&apos;s own P(start), both are shown.
        </div>
      )}
    </div>
  )
}

/* One club's predicted eleven, drawn in the shape the feed says they line up
   in — its own pitch positions (DL, DMC, AMR…), not FPL's four labels, which
   would flatten every 4-2-3-1 with an attacking winger into a 4-5-1. Keeper at
   the top, as on the Planner. The model's P(start) sits under each name, and
   anyone the model rates as a likely starter but the feed left out is listed
   underneath, because that is the disagreement worth an eye. */
function XiCard({ c, team }) {
  const bands = []
  for (const x of c.xi || []) {
    (bands[x.band] = bands[x.band] || []).push(x)
  }
  const omitted = (c.rows || []).filter((r) => !r.predicted && r.disagree)
  return (
    <div className="lu-card">
      <div className="lu-head">
        <img className="fx-crest" src={badgeUrl(team?.code)} alt="" loading="lazy"
          onError={(e) => { e.currentTarget.style.visibility = 'hidden' }} />
        <b>{team?.name || c.team}</b>
        {c.formation && <span className="lu-form">{c.formation}</span>}
        <span className="lu-when">{ago(Date.parse(c.observed) / 1000)}</span>
      </div>
      <div className="lu-pitch">
        <PitchLines />
        <div className="lu-rows">
          {bands.filter(Boolean).map((row, i) => (
            <div className="lu-row" key={i}>
              {row.map((x) => (
                <div key={`${x.full_name}-${x.slot}`}
                  className={`lu-p ${x.disagree ? 'dis' : ''} ${x.resolved ? '' : 'unk'}`}
                  title={`${x.full_name} · ${x.position}${x.p_start != null
                    ? ` · model P(start) ${Math.round(x.p_start * 100)}%` : ''}`}>
                  <span className="lu-name">{x.name}</span>
                  <span className="lu-ps">
                    {x.p_start != null ? `${Math.round(x.p_start * 100)}%` : '—'}
                  </span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
      {omitted.length > 0 && (
        <div className="lu-omit">
          <span>Model also rates:</span>
          {omitted.map((r) => (
            <b key={r.player_id}>{r.name} {Math.round((r.p_start || 0) * 100)}%</b>
          ))}
        </div>
      )}
    </div>
  )
}

function Picks({ rows, gw }) {
  if (!rows.length) return null
  return (
    <div className="panel">
      <div className="panel-head">The model&apos;s board — GW{gw}
        <span className="panel-sub">highest projected points</span></div>
      <div className="pick-grid">
        {rows.map((r, i) => (
          <div className={`pick ${i === 0 ? 'cap' : ''}`} key={r.player_id}>
            <span className="pk-rank">{i === 0 ? 'C' : i + 1}</span>
            <div className="pk-text">
              <b>{r.name}</b>
              <span className="muted">{r.team} · {r.pos} · {money(r.price)}</span>
            </div>
            <span className="pk-ep">{fmt1(r.ep)}</span>
          </div>
        ))}
      </div>
      <div className="fold-note" style={{ padding: '0 12px 12px' }}>
        Captaining the model&apos;s top pick was worth about +0.7 to +1.1 points a
        week over the crowd&apos;s choice across two replayed seasons. {rows[0].name} is
        this gameweek&apos;s.
      </div>
    </div>
  )
}

function ago(ts) {
  if (!ts) return ''
  const s = Math.max(0, Date.now() / 1000 - ts)
  if (s < 90) return 'just now'
  if (s < 3600) return `${Math.round(s / 60)} min ago`
  if (s < 86400) return `${Math.round(s / 3600)} h ago`
  return `${Math.round(s / 86400)} d ago`
}
