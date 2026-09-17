import React, { useEffect, useState } from 'react'
import { api } from '../api'
import { useStore } from '../store'
import { Empty, Loading } from '../components/States'
import { badgeUrl, fmt1, money, shirtUrl } from '../util'
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

  /* On a desktop the desk is a dashboard that fits the screen: the model's
     answer on the left (its best XI, then who is moving price), the reasoning
     in the middle (which fixtures are favourable and who to own there), the
     live feed on the right. Every panel scrolls inside itself, so nothing
     pushes the rest below the fold. Narrower screens stack the same panels in
     reading order. */
  return (
    <div className="live">
      <LiveHero gw={d.gw} open={open} preview={preview} left={left} since={since}
        deadline={win.deadline} built={d.built_at} projected={d.proj_updated_at} />

      <div className="live-dash">
        <section className="ld-col ld-a">
          <BestXI xi={d.best_xi} ready={d.components} />
          <PriceWatch p={d.prices} open={open} />
        </section>
        <section className="ld-col ld-b">
          <FixturePicks clubs={d.fixture_picks || []} ready={d.components} gw={d.gw} />
        </section>
        <section className="ld-col ld-c">
          <Fixtures rows={d.fixtures || []} />
          <TeamNews rows={d.news || []} />
          <Pressers rows={d.pressers || []} gw={d.gw} />
          <Movers m={d.movers || { rows: [] }} />
        </section>
      </div>

      <Lineups l={d.lineups || { clubs: [] }} />
    </div>
  )
}

/* ------------------------------------------------------------------ */

function LiveHero({ gw, open, preview, left, since, deadline, built, projected }) {
  // the run-up as a bar: empty a day out, full at the deadline
  const gone = open ? Math.max(0, Math.min(1, 1 - left / (24 * 3600 * 1000))) : 1
  /* One line, not a banner. The clock is the headline; everything else on
     the old hero was a caption that cost 200px of the screen. */
  return (
    <div className={`live-hero ${open ? 'open' : 'over'}`}>
      <span className={`live-badge ${preview || !open ? 'over' : ''}`}>
        <i className="dot" aria-hidden="true" />
        {preview ? 'PREVIEW' : open ? 'LIVE' : 'DEADLINE PASSED'}
      </span>
      <div className="lh-title">
        <b>Gameweek {gw}</b>
        <span>
          {!open ? 'under way — nothing can be changed now'
            : preview ? 'goes on air for everyone 24 h before the deadline'
              : 'until transfers, captain and chips lock'}
        </span>
      </div>
      <div className="lh-clock">{open ? <Countdown ms={left} /> : <Elapsed ms={since} />}</div>
      <div className="lh-meta">
        <span>deadline {new Date(deadline * 1000).toLocaleString(undefined,
          { weekday: 'short', hour: '2-digit', minute: '2-digit' })}</span>
        {projected ? <span>model {ago(projected)}</span> : null}
        {built ? <span>refreshed {ago(built)}</span> : null}
      </div>
      {open && <div className="lh-bar"><div style={{ width: `${gone * 100}%` }} /></div>}
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
  // grouped by kick-off, one line per match, crest first
  const groups = []
  for (const r of rows) {
    const d = r.kickoff ? new Date(r.kickoff) : null
    const key = d ? d.toLocaleString(undefined, { weekday: 'short', day: 'numeric', month: 'short' }) : 'TBC'
    const g = groups.find((x) => x.key === key)
    const row = { ...r, time: d ? d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' }) : '' }
    if (g) g.rows.push(row)
    else groups.push({ key, rows: [row] })
  }
  const Crest = ({ id }) => (
    <img className="fx-crest" src={badgeUrl(teams[String(id)]?.code)} alt="" loading="lazy"
      onError={(e) => { e.currentTarget.style.visibility = 'hidden' }} />
  )
  return (
    <div className="panel ld-fixtures">
      <div className="panel-head">Fixtures <span className="chip dim num">{rows.length}</span></div>
      <div className="ld-scroll fx-list">
        {groups.map((g) => (
          <div key={g.key}>
            <div className="fx-day">{g.key}</div>
            {g.rows.map((r) => (
              <div className={`fx-line ${r.finished ? 'done' : ''}`} key={r.fixture_id}>
                <span className="fx-time">{r.score ? 'FT' : r.time}</span>
                <span className="fx-home"><Crest id={r.home_id} />{teams[String(r.home_id)]?.short || r.home}</span>
                <span className="fx-vs">{r.score ? `${r.score[0]}–${r.score[1]}` : 'v'}</span>
                <span className="fx-away">{teams[String(r.away_id)]?.short || r.away}<Crest id={r.away_id} /></span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */

const TAG = {
  ATT: { label: 'Attack', title: 'most of his projection is goals and assists' },
  DEFCON: { label: 'DefCon', title: 'most of his projection is crossing the defensive-contribution threshold' },
  CS: { label: 'Clean sheet', title: 'most of his projection is a clean sheet' },
  SAVES: { label: 'Saves', title: 'most of his projection is save points' },
}
const pct = (v) => `${Math.round((v || 0) * 100)}%`

/* The model's best legal XI for the gameweek: a keeper, 3-5 defenders, 2-5
   midfielders, 1-3 forwards, no more than three from a club, captain counted
   twice. It is "who the model would start", not a squad it could afford —
   the Planner and Solver are where money is. */
function BestXI({ xi, ready }) {
  const { teams } = useStore()
  return (
    <div className="panel ld-xi">
      <div className="panel-head">Model&apos;s best XI
        {xi && <span className="bx-meta">{xi.formation} · <b>{fmt1(xi.points)}</b> pts · {money(xi.cost)}</span>}
      </div>
      {!xi ? (
        <div className="dd-empty">
          {ready === false
            ? 'The component breakdown is built on the next projection refresh.'
            : 'No legal XI could be built from the current projections.'}
        </div>
      ) : (
        <div className="bx-pitch">
          <PitchLines />
          <div className="bx-rows">
            {xi.rows.map((row, i) => (
              <div className="bx-row" key={i}>
                {row.map((p) => {
                  const t = teams[String(p.team_id)]
                  return (
                    <div className="bx-p" key={p.player_id}
                      title={`${p.name} · ${t?.short || ''} · ${money(p.price)} · ${TAG[p.tag]?.title || ''}`}>
                      {(p.captain || p.vice) && (
                        <span className={`bx-arm ${p.vice ? 'vice' : ''}`}>{p.captain ? 'C' : 'V'}</span>
                      )}
                      <img className="bx-shirt" alt="" loading="lazy"
                        src={shirtUrl(t?.code, p.pos === 'GK')}
                        onError={(e) => { e.currentTarget.style.visibility = 'hidden' }} />
                      <span className="bx-name">{p.name}</span>
                      <span className="bx-line">
                        <b>{fmt1(p.captain ? p.ep * 2 : p.ep)}</b>
                        <i className={`bx-tag t-${p.tag.toLowerCase()}`}>{p.tag === 'DEFCON' ? 'DC' : p.tag === 'ATT' ? 'ATT' : p.tag === 'CS' ? 'CS' : 'SV'}</i>
                      </span>
                    </div>
                  )
                })}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

/* Where to look this gameweek. Clubs are ranked by the market's expected goal
   difference for their fixture — the favourable ones first — and inside each
   club the attacking picks come before the DefCon and clean-sheet ones,
   because an attacking return is the bigger swing. Each player carries the
   one or two numbers that say WHY he is on the list. */
function FixturePicks({ clubs, ready, gw }) {
  const { teams } = useStore()
  return (
    <div className="panel ld-picks">
      <div className="panel-head">Picks by fixture — GW{gw}
        <span className="panel-sub">most favourable first · attack, then DefCon</span></div>
      {!clubs.length ? (
        <div className="dd-empty">
          {ready === false
            ? 'The component breakdown is built on the next projection refresh.'
            : 'No priced fixtures for this gameweek yet.'}
        </div>
      ) : (
        <div className="ld-scroll fp-list">
          {clubs.map((c, i) => {
            const t = teams[String(c.team_id)]
            return (
              <div className="fp-club" key={c.team_id}>
                <div className="fp-head">
                  <span className="fp-rank">{i + 1}</span>
                  <img className="fx-crest" src={badgeUrl(t?.code)} alt="" loading="lazy"
                    onError={(e) => { e.currentTarget.style.visibility = 'hidden' }} />
                  <b>{t?.name || c.team_id}</b>
                  <span className="fp-opp">
                    {c.fixtures.map((f, k) => (
                      <span key={k} className="ep-opp">
                        {k > 0 && <i className="ep-opp-sep">+</i>}
                        {teams[String(f.opp)]?.short || '?'}<b className={f.home ? 'h' : 'a'}>{f.home ? 'H' : 'A'}</b>
                      </span>
                    ))}
                  </span>
                  <span className="fp-stats" title={c.priced ? 'bookmaker-implied' : "no market price — the model's own estimate"}>
                    <span>xG <b>{c.xg.toFixed(2)}</b></span>
                    <span>xGA <b>{c.xga.toFixed(2)}</b></span>
                    <span>CS <b>{pct(c.p_cs)}</b></span>
                  </span>
                </div>
                <PickGroup label="Attack" rows={c.attack} kind="att" />
                <PickGroup label="DefCon & clean sheet" rows={c.defence} kind="def" />
                {c.keeper && (
                  <div className="fp-keeper">
                    Keeper <b>{c.keeper.name}</b> {money(c.keeper.price)} · {fmt1(c.keeper.ep)} pts
                    · CS {pct(c.keeper.p_cs)}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
      <div className="fold-note fp-note">
        Attack: <b>xGI</b> = the model&apos;s expected goals + assists this gameweek.
        DefCon: the chance he crosses the threshold (10 actions for a defender, 12
        for a midfielder), worth 2 points. CS: chance of a clean sheet while he is on.
      </div>
    </div>
  )
}

function PickGroup({ label, rows, kind }) {
  if (!rows?.length) return null
  return (
    <div className={`fp-group ${kind}`}>
      <div className="fp-glabel">{label}</div>
      {rows.map((p) => (
        <div className="fp-row" key={p.player_id}>
          <span className="fp-name">
            <b>{p.name}</b>
            <span className="muted">{p.pos} · {money(p.price)}{p.own != null ? ` · ${p.own}%` : ''}</span>
          </span>
          <span className="fp-chips">
            {kind === 'att' ? (
              <>
                <i className="fp-chip att" title="expected goals + assists this gameweek">xGI {p.xgi.toFixed(2)}</i>
                {p.pk && <i className="fp-chip pk" title="on penalties">PEN</i>}
              </>
            ) : (
              <>
                {p.p_defcon >= 0.15 && (
                  <i className="fp-chip dc" title="chance he crosses the DefCon threshold">DefCon {pct(p.p_defcon)}</i>
                )}
                <i className="fp-chip cs" title="chance of a clean sheet while he is on">CS {pct(p.p_cs)}</i>
              </>
            )}
          </span>
          <span className="fp-ep">{fmt1(p.ep)}</span>
        </div>
      ))}
    </div>
  )
}

function TeamNews({ rows }) {
  return (
    <div className={`panel ld-news ${rows.length ? '' : 'is-empty'}`}>
      <div className="panel-head">Team news <span className="chip dim num">{rows.length}</span>
        <span className="panel-sub">FPL&apos;s own feed, newest first</span></div>
      <div className="live-list ld-scroll">
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
    <div className={`panel ld-pressers ${rows.length ? '' : 'is-empty'}`}>
      <div className="panel-head">Managers said <span className="chip dim num">{rows.length}</span>
        <span className="panel-sub">Friday press conferences (BBC)</span></div>
      <div className="live-list ld-scroll">
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
    <div className={`panel ld-movers ${rows.length ? '' : 'is-empty'}`}>
      <div className="panel-head">The model changed its mind
        <span className="panel-sub">since the previous build</span></div>
      <div className="live-list ld-scroll">
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
    <div className="panel ld-prices">
      <div className="panel-head">Price watch
        <span className="panel-sub">{open ? 'before tonight' : 'this gameweek'}</span></div>
      <div className="live-two ld-scroll">
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
      {clubs.some((c) => c.shape_change) && (
        <div className="lu-changes">
          Predicted to change shape:{' '}
          {clubs.filter((c) => c.shape_change).map((c) => (
            <b key={c.team}>{c.team} {c.last_formation} → {c.formation}</b>
          ))}
        </div>
      )}
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

/* One club's expected eleven, drawn in the formation the feed STATES, row by
   row as the feed lists it. (RotoWire's per-player positions looked like a
   formation but were a template — five patterns across 239 lineups — so they
   no longer draw anything.) Most clubs play the same shape every week, so the
   useful signal is a predicted CHANGE from what the club last played, which
   is called out in gold. The model's P(start) sits under each name, and
   anyone it rates as a likely starter that the feed left out is listed
   underneath. */
function XiCard({ c, team }) {
  const omitted = (c.rows || []).filter((r) => !r.predicted && r.disagree)
  return (
    <div className={`lu-card ${c.shape_change ? 'changed' : ''}`}>
      <div className="lu-head">
        <img className="fx-crest" src={badgeUrl(team?.code)} alt="" loading="lazy"
          onError={(e) => { e.currentTarget.style.visibility = 'hidden' }} />
        <b>{team?.name || c.team}</b>
        {c.formation && <span className="lu-form">{c.formation}</span>}
        <span className="lu-when">{ago(Date.parse(c.observed) / 1000)}</span>
      </div>
      <div className="lu-sub">
        {c.shape_change ? (
          <span className="lu-change">shape change · last played {c.last_formation}</span>
        ) : c.last_formation ? (
          <span>same shape as last match</span>
        ) : (
          <span>no previous match on record</span>
        )}
        <span className="lu-src">{c.status === 'confirmed' ? 'confirmed' : 'predicted'} · {c.source}</span>
      </div>
      <div className="lu-pitch">
        <PitchLines />
        <div className="lu-rows">
          {(c.xi_rows || []).map((row, i) => (
            <div className="lu-row" key={i}>
              {row.map((x, j) => (
                <div key={`${x.full_name}-${j}`}
                  className={`lu-p ${x.disagree ? 'dis' : ''} ${x.resolved ? '' : 'unk'}`}
                  title={`${x.full_name}${x.p_start != null
                    ? ` · model P(start) ${Math.round(x.p_start * 100)}%` : ' · not matched to an FPL player'}`}>
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

function ago(ts) {
  if (!ts) return ''
  const s = Math.max(0, Date.now() / 1000 - ts)
  if (s < 90) return 'just now'
  if (s < 3600) return `${Math.round(s / 60)} min ago`
  if (s < 86400) return `${Math.round(s / 3600)} h ago`
  return `${Math.round(s / 86400)} d ago`
}
