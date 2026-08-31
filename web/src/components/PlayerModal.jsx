import React, { useMemo, useState } from 'react'
import Flag from './Flag'
import { useFixtureLookup, useStore } from '../store'
import { Radar, VIZ } from '../charts'
import { availPct, badgeUrl, epColor, epOf, fdrColor, fmt1, money, photoUrl } from '../util'

/* FPL Review-style player card: horizon projections, availability, per-90
   rates and fixture run for one player in the active draft. The EP column is
   the model output (fixture-aware); the per-90 columns are current season
   rates (xG-based once the season has data, else last-season history). */
export default function PlayerModal({ pid, draft, plan, actions, close }) {
  const { byId, teams, proj, projHistory, players, watch,
          setTransferWatch, context } = useStore()
  const fixOf = useFixtureLookup()
  const [fdrMode, setFdrMode] = useState('diff_att')
  const p = byId.get(pid)
  const team = teams[String(p?.team_id)]
  const inXi = plan?.xi?.includes(pid)

  const gws = (draft?.gws || []).map((g) => g.gw)
  const ava = availPct(p)
  // engine xmins (from the projection run) already folds availability in;
  // the app-layer history estimate does not, so scale only the latter
  const engineXm = proj?.players?.[String(pid)]?.xmins
  const xminsBase = engineXm != null ? engineXm / Math.max(0.01, ava / 100) : (p?.xmins ?? 0)

  const rows = useMemo(() => gws.map((gw) => {
    const fixes = fixOf(p?.team_id, gw)
    // the engine now publishes expected minutes per gameweek; fall back to the
    // single-value estimate for caches built before that
    const perGw = proj?.players?.[String(pid)]?.xm?.[String(gw)]
    const xm = perGw != null
      ? perGw * Math.max(1, fixes.length)
      : xminsBase * (ava / 100) * Math.max(1, fixes.length)
    return { gw, fixes, ep: epOf(proj, pid, gw), xm }
  }), [gws.join(','), p, proj, pid, ava, xminsBase, fixOf])

  // model trend: this player's horizon total across projection builds
  const trend = useMemo(() => {
    const pts = []
    for (const s of projHistory || []) {
      let tot = 0, k = 0
      for (const gw of gws) {
        const v = s.gws?.[String(gw)]?.[String(pid)]
        if (v != null) { tot += v; k++ }
      }
      if (k) pts.push(tot)
    }
    return pts
  }, [projHistory, gws.join(','), pid])

  // Percentile WITHIN POSITION. An 0.35 g90 is elite for a defender and
  // ordinary for a forward, so an absolute scale would make every centre-back
  // look useless and every striker look identical. Percentiles also make the
  // axes commensurable, which is the only way a radar means anything.
  const PROFILE_AXES = [
    { key: 'g90', label: 'Goal threat', fmt: (v) => `${v.toFixed(0)}th pct` },
    { key: 'a90', label: 'Creativity', fmt: (v) => `${v.toFixed(0)}th pct` },
    { key: 'xm', label: 'Minutes', fmt: (v) => `${v.toFixed(0)}th pct` },
    { key: 'dc90', label: 'Defensive', fmt: (v) => `${v.toFixed(0)}th pct` },
    { key: 'cs90', label: 'Clean sheets', fmt: (v) => `${v.toFixed(0)}th pct` },
    { key: 'ppm', label: 'Value', fmt: (v) => `${v.toFixed(0)}th pct` },
  ]
  const profile = useMemo(() => {
    if (!p || !players?.length) return null
    const peers = players.filter((q) => q.position === p.position)
    if (peers.length < 8) return null
    const epTot = (q) => {
      const r = proj?.players?.[String(q.id)]
      if (!r?.ep) return 0
      return gws.reduce((a, g) => a + (r.ep[String(g)] ?? r.ep[g] ?? 0), 0)
    }
    const val = (q, key) => {
      if (key === 'xm') return proj?.players?.[String(q.id)]?.xmins ?? q.xmins ?? 0
      if (key === 'ppm') return q.price ? epTot(q) / q.price : 0
      return q[key] ?? 0
    }
    const pct = (key) => {
      const mine = val(p, key)
      const vs = peers.map((q) => val(q, key))
      const below = vs.filter((v) => v < mine).length
      const same = vs.filter((v) => v === mine).length
      return ((below + same / 2) / vs.length) * 100
    }
    const values = PROFILE_AXES.map((a) => pct(a.key))
    return [{ name: p.web_name, color: VIZ[0], values, raw: values }]
  }, [p, players, proj, gws.join(',')])

  const n = rows.length || 1
  const totEp = rows.reduce((a, r) => a + r.ep, 0)
  const totXm = rows.reduce((a, r) => a + r.xm, 0)
  const goals = (p?.g90 || 0) * totXm / 90
  const assists = (p?.a90 || 0) * totXm / 90
  const cs = (p?.cs90 || 0) * totXm / 90
  const ppm = p?.price ? totEp / n / p.price : 0

  if (!p) return null

  return (
    <div className="pmodal-overlay" onClick={close}>
      <div className="pmodal" onClick={(e) => e.stopPropagation()}>
        <div className="head pm-head">
          {/* The cut-out is a BACKGROUND, not a picture of a man in a box: it
              is masked away towards the text and the bottom edge so it
              dissolves into the panel instead of sitting on it, and the club
              badge behind it supplies the colour. Text always wins. */}
          {photoUrl(p.code) && (
            <div className="pm-hero" aria-hidden="true">
              <img className="pm-hero-badge" src={badgeUrl(team?.code)} alt=""
                onError={(e) => { e.currentTarget.style.display = 'none' }} />
              <img className="pm-hero-img" src={photoUrl(p.code)} alt="" loading="lazy"
                onError={(e) => { e.currentTarget.closest('.pm-hero').style.display = 'none' }} />
            </div>
          )}
          <div className="pm-head-text">
            <h2>{p.web_name}</h2>
            <div className="sub">
              {team?.short || '?'} · {p.position} · {money(p.price)}
              <Flag p={p} />
              {p.status && p.status !== 'a' && p.news && (
                <span style={{ color: 'var(--muted)', marginLeft: 6 }}>{p.news}</span>
              )}
            </div>
          </div>
          <button className="close" onClick={close}>✕</button>
        </div>

        {actions && (
          <div className="actions">
            {inXi && <button className="pill-btn" onClick={actions.captain}>Ⓒ Captain</button>}
            {inXi && <button className="pill-btn" onClick={actions.vice}>Ⓥ Vice</button>}
            <button className="pill-btn" onClick={actions.swap}>⇄ Switch</button>
            <button className="pill-btn" style={{ color: 'var(--red)' }}
              onClick={actions.transfer}>✕ Transfer</button>
            {actions.undo && (
              <button className="pill-btn" style={{ color: 'var(--gold)' }}
                onClick={actions.undo}>↶ Undo transfer</button>
            )}
          </div>
        )}

        <div className="pd-grid">
          <div className="statgrid">
            <Tile k={`Points (${n} GW)`} v={fmt1(totEp)} bar={totEp / (6 * n)} />
            <Tile k="Goals" v={goals.toFixed(2)} bar={goals / (0.8 * n)} />
            <Tile k="Assists" v={assists.toFixed(2)} bar={assists / (0.8 * n)} />
            <Tile k="Clean sheets" v={cs.toFixed(2)} bar={cs / (0.6 * n)} />
          </div>
          <div>
            <div className="statgrid" style={{ gridTemplateColumns: '1fr 1fr 1fr', marginBottom: 10 }}>
              <Tile k="Ownership" v={`${(p.own ?? 0).toFixed(1)}%`} />
              <Tile k="Availability" v={`${ava}%`}
                warn={ava < 100} />
              <Tile k="PPM" v={ppm.toFixed(2)} sub="pts / gw / £m" />
            </div>
            <div className="fdr-head">
              <span className="section-label">Fixture difficulty</span>
              <span style={{ marginLeft: 'auto', display: 'flex', gap: 4 }}>
                {[['diff_att', 'Attacking'], ['diff_def', 'Defensive']].map(([v, l]) => (
                  <button key={v}
                    className={`mode ${fdrMode === v ? 'on' : ''}`}
                    onClick={() => setFdrMode(v)}>{l}</button>
                ))}
              </span>
            </div>
            <div className="fdrbar">
              {rows.map((r) => {
                const d = r.fixes.length
                  ? r.fixes.reduce((a, f) => a + (f[fdrMode] ?? f.fdr ?? 3), 0) / r.fixes.length
                  : null
                const c = fdrColor(d)
                return (
                  <div className="col" key={r.gw}>
                    <div className="bar" style={{
                      height: d == null ? 4 : 8 + (d - 1) * 12,
                      background: c.bg,
                    }} title={d == null ? 'blank' : d.toFixed(1)} />
                    <div className="lbl">{r.gw} {r.fixes.map((f) =>
                      f.home ? f.oppShort : f.oppShort.toLowerCase()).join(',') || '–'}</div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>

        <div style={{ padding: '0 20px 16px' }}>
          <table className="pd-table">
            <thead>
              <tr>
                <th className="l">GW</th><th className="l">Opp</th>
                <th>FDR</th><th>Pts</th><th>xMins</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.gw}>
                  <td className="l">{r.gw}</td>
                  <td className="l">{r.fixes.map((f) =>
                    `${f.oppShort}${f.home ? ' (H)' : ' (A)'}`).join(', ') || '–'}</td>
                  <td>{(() => {
                    if (!r.fixes.length) return <span style={{ color: 'var(--muted-2)' }}>–</span>
                    const d = r.fixes.reduce((a, f) => a + (f[fdrMode] ?? f.fdr ?? 3), 0) / r.fixes.length
                    const c = fdrColor(d)
                    return <span className="ep-chip"
                      style={{ background: c.bg, color: c.fg, minWidth: 34 }}>{d.toFixed(1)}</span>
                  })()}</td>
                  <td><span className="ep-chip" style={{ background: epColor(r.ep) }}>{fmt1(r.ep)}</span></td>
                  <td>{Math.round(r.xm)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <TransferWatch pid={pid} p={p} teams={teams} gws={gws}
            watch={watch} setTransferWatch={setTransferWatch} rows={rows} />

          <div className="pd-rates">
            <span className="section-label">Season rates</span>
            <div className="pd-rates-row">
              {[
                ['Avail', `${ava}%`, 'current availability from FPL'],
                ['PK', `${Math.round((p.pk_share || 0) * 100)}%`, 'share of his club’s penalties'],
                ['G90', (p.g90 ?? 0).toFixed(2), 'goals per 90'],
                ['A90', (p.a90 ?? 0).toFixed(2), 'assists per 90'],
                ['DC90', (p.dc90 ?? 0).toFixed(2),
                  p.position === 'DEF' ? 'CBIT per 90' : 'CBIRT per 90'],
                ['CS90', (p.cs90 ?? 0).toFixed(2), 'clean sheets per 90'],
              ].map(([k, v, tip]) => (
                <div key={k} title={tip}><span>{k}</span><b>{v}</b></div>
              ))}
            </div>
            <p className="pd-rates-note">
              Per-90 rates across the season. The opponent is priced separately
              and shows in <b>Pts</b>.
            </p>
          </div>
          {profile && (
            <div className="pm-profile">
              <span className="section-label">
                Profile — percentile among {p.position}s
              </span>
              <Radar axes={PROFILE_AXES} series={profile} size={300} />
            </div>
          )}
          {trend.length >= 2 && (
            <div className="trend-row">
              <span className="section-label">Model trend</span>
              <Spark pts={trend} />
              {(() => {
                const d = trend[trend.length - 1] - trend[trend.length - 2]
                return (
                  <span className={`dv ${d >= 0 ? 'up' : 'down'}`}>
                    {d >= 0 ? '▲ +' : '▼ '}{fmt1(d)} vs previous build
                  </span>
                )
              })()}
              <span style={{ fontSize: 10.5, color: 'var(--muted-2)' }}>
                horizon total over {trend.length} builds
              </span>
            </div>
          )}
          <Dossier ctx={context?.players?.[String(pid)]}
                   club={context?.clubs?.[String(p.team_id)]} />
          {p.recent_mins?.length > 0 && (
            <div className="recent-mins">
              <span className="section-label">Recent minutes</span>
              {p.recent_mins.map((m, i) => (
                <span key={i} className="minchip" style={{
                  background: m >= 60 ? 'rgba(47,214,128,0.16)' : m > 0 ? 'rgba(255,182,27,0.16)' : 'var(--panel-2)',
                  color: m >= 60 ? 'var(--green)' : m > 0 ? 'var(--gold)' : 'var(--muted-2)',
                }}>{m}′</span>
              ))}
              <span style={{ fontSize: 10.5, color: 'var(--muted-2)' }}>
                newest first{p.p_start != null ? ` · ${Math.round(p.p_start * 100)}% to start` : ''}{p.start_rate != null ? ` · started ${Math.round(p.start_rate * 100)}% of last 10` : ''}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

/* "He is off to Spurs" — the model cannot know that, but it can price it.

   No free feed carries transfer rumours: FPL reclassifies a player only after
   the move completes, and prediction markets only cover the superstar tier. So
   the fact comes from you and the consequence comes from the engine, which
   reprojects him onto the destination's fixtures using his own rates. */
function TransferWatch({ pid, p, teams, gws, watch, setTransferWatch, rows }) {
  const entry = watch?.players?.[String(pid)]
  const alt = watch?.alt?.[String(pid)]
  const [open, setOpen] = useState(false)
  const clubs = useMemo(
    () => Object.entries(teams || {})
      .map(([id, t]) => ({ id: Number(id), name: t.name }))
      .sort((a, b) => a.name.localeCompare(b.name)), [teams])

  const now = rows.reduce((a, r) => a + r.ep, 0)
  const after = alt
    ? gws.reduce((a, g) => a + (alt.ep?.[String(g)] ?? 0), 0)
    : null
  const dest = entry?.to_team ? teams?.[String(entry.to_team)]?.name : null

  const rumour = watch?.rumours?.[String(pid)]

  if (!entry && !open) {
    return (
      <div className="tw-row">
        {rumour ? (
          <div className="tw-suggest">
            <span className="tw-tag">rumour</span>
            <span className="tw-sug-text">
              {rumour.leaves_league
                ? <>linked with <b>{rumour.to_club}</b> — would leave the Premier League</>
                : <>linked with <b>{rumour.to_club}</b></>}
              {rumour.probability != null && rumour.probability >= 0
                ? <> · Transfermarkt rates it <b>{rumour.probability}%</b></> : null}
            </span>
            <button className="pill-btn accent"
              onClick={() => setTransferWatch(pid, {
                to_team: rumour.to_team ?? null,
                note: `Transfermarkt: ${rumour.to_club}`,
              })}>
              Apply
            </button>
          </div>
        ) : null}
        <button className="pill-btn" onClick={() => setOpen(true)}
          title="Mark him as leaving and reproject him on the new club's fixtures">
          ⇄ {rumour ? 'Set manually' : 'Transfer rumour?'}
        </button>
      </div>
    )
  }

  return (
    <div className="tw-panel">
      <div className="tw-head">
        <span className="section-label">Reported move</span>
        {entry && (
          <button className="pill-btn" style={{ marginLeft: 'auto' }}
            onClick={() => { setTransferWatch(pid, null); setOpen(false) }}>
            ✕ Clear
          </button>
        )}
      </div>
      <div className="tw-controls">
        <select className="pill-btn" style={{ background: 'var(--panel)', appearance: 'auto' }}
          value={entry?.to_team ?? ''}
          onChange={(e) => setTransferWatch(pid, {
            to_team: e.target.value ? Number(e.target.value) : null,
            note: entry?.note || '',
          })}>
          <option value="">Leaving the Premier League</option>
          {clubs.filter((c) => c.id !== p.team_id).map((c) => (
            <option key={c.id} value={c.id}>Joining {c.name}</option>
          ))}
        </select>
      </div>
      {entry?.to_team && after != null ? (
        <div className="tw-compare">
          <div><span>at {teams?.[String(p.team_id)]?.name}</span><b>{fmt1(now)}</b></div>
          <span className="tw-arrow">→</span>
          <div><span>at {dest}</span><b>{fmt1(after)}</b></div>
          <div className={`tw-delta ${after >= now ? 'up' : 'down'}`}>
            {after >= now ? '+' : ''}{fmt1(after - now)} over {gws.length} GW
          </div>
        </div>
      ) : entry ? (
        <p className="tw-note">
          Out of the Premier League — he scores nothing and must be sold.
          {entry.to_team ? '' : ' Pick a club above to see the reprojection.'}
        </p>
      ) : null}
      <p className="tw-note">
        Fixtures only. His P(start) is earned at his current club; how a move
        changes his role is not something this data can tell you.
      </p>
    </div>
  )
}

function Spark({ pts, w = 120, h = 26 }) {
  const lo = Math.min(...pts), hi = Math.max(...pts)
  const span = hi - lo || 1
  const xs = pts.map((v, i) => [
    (i / Math.max(1, pts.length - 1)) * (w - 4) + 2,
    h - 3 - ((v - lo) / span) * (h - 6),
  ])
  const up = pts[pts.length - 1] >= pts[0]
  return (
    <svg width={w} height={h} className="spark">
      <polyline fill="none" stroke={up ? 'var(--green)' : 'var(--red)'} strokeWidth="1.8"
        points={xs.map(([x, y]) => `${x},${y}`).join(' ')} />
      <circle cx={xs[xs.length - 1][0]} cy={xs[xs.length - 1][1]} r="2.5"
        fill={up ? 'var(--green)' : 'var(--red)'} />
    </svg>
  )
}

function Tile({ k, v, bar, sub, warn }) {
  return (
    <div className="stattile">
      <div className="k">{k}</div>
      <div className="v" style={warn ? { color: 'var(--gold)' } : undefined}>{v}</div>
      {sub && <div className="s">{sub}</div>}
      {bar != null && (
        <div className="tilebar">
          <div style={{ width: `${Math.max(3, Math.min(100, bar * 100))}%` }} />
        </div>
      )}
    </div>
  )
}


/* Transfermarkt dossier — shown BESIDE the recommendation, never inside it.

   Every field here was measured against the minutes model and against decision
   metrics and none of it earned a place in the objective: injury history moves
   no decision (spearman_played -0.0013, and worse in the opening gameweeks),
   and age, market value, transfers and squad competition together are worth
   about 1-2% of log-loss and nothing at all in points per pick.

   That is not the same as being useless to a person. Choosing between two
   players the model rates alike, it matters that one is three hamstrings into
   two years, is 33, is out of contract in June, and signed six weeks ago. So
   it follows the rule the price model and Polymarket already follow: reported
   next to the recommendation, never folded into it. */
function Dossier({ ctx, club }) {
  if (!ctx && !club) return null
  const inj = ctx?.injury
  const cur = inj?.current
  const tr = ctx?.last_transfer
  const eur = (v) => (v == null ? null
    : v >= 1e9 ? `€${(v / 1e9).toFixed(2)}bn`
    : v >= 1e6 ? `€${(v / 1e6).toFixed(v >= 1e7 ? 0 : 1)}m`
    : `€${Math.round(v / 1e3)}k`)
  const day = (d) => (d ? new Date(d).toLocaleDateString(undefined,
    { day: 'numeric', month: 'short', year: '2-digit' }) : null)
  const contractSoon = ctx?.contract_until
    && (new Date(ctx.contract_until) - new Date()) / 86400000 < 365

  const chips = []
  if (ctx?.age != null) chips.push(['age', `${ctx.age}`])
  if (ctx?.detail_position) chips.push(['role', ctx.detail_position])
  if (ctx?.market_value) chips.push(['value', eur(ctx.market_value)])
  if (ctx?.mv_change_365 != null) {
    const v = ctx.mv_change_365
    chips.push(['1yr', `${v >= 0 ? '+' : ''}${Math.round(v * 100)}%`,
      v >= 0.05 ? 'up' : v <= -0.05 ? 'down' : null])
  }
  if (ctx?.contract_until) {
    chips.push(['contract', day(ctx.contract_until), contractSoon ? 'warn' : null])
  }
  if (ctx?.foot && ctx.foot !== 'right') chips.push(['foot', ctx.foot])

  return (
    <div className="tm-dossier">
      <span className="section-label">
        Transfermarkt — context, not a projection
      </span>

      {cur && (
        <div className="tm-alert">
          <span className="tm-tag out">out</span>
          <span>
            <b>{cur.injury || 'Injured'}</b> since {day(cur.since)} ({cur.days_out}d)
            {cur.expected_back
              ? <> · Transfermarkt expects him back <b>{day(cur.expected_back)}</b>
                  <span className="tm-hint"> (their forecast, not a fact)</span></>
              : <> · <span className="tm-hint">no return date given</span></>}
          </span>
        </div>
      )}

      {tr?.new_signing && (
        <div className="tm-alert new">
          <span className="tm-tag new">new</span>
          <span>
            Signed from <b>{tr.from_club || '?'}</b> {tr.days_ago}d ago
            {tr.fee_text ? <> · {tr.fee_text}</> : null}
            <span className="tm-hint">
              {tr.from_pl === false
                ? ' — no Premier League history to trail'
                : tr.from_pl === true
                  ? ' — his trailing form is from another club'
                  : ''}
            </span>
          </span>
        </div>
      )}

      {chips.length > 0 && (
        <div className="tm-chips">
          {chips.map(([k, v, tone]) => (
            <span key={k} className={`tm-chip${tone ? ` ${tone}` : ''}`}>
              <em>{k}</em>{v}
            </span>
          ))}
        </div>
      )}

      {inj && (inj.spells_730 > 0 || inj.days_since_return != null) && (
        <div className="tm-inj">
          <span className="tm-inj-head">Injury record</span>
          <span className="tm-inj-line">
            {inj.spells_365} spell{inj.spells_365 === 1 ? '' : 's'} and{' '}
            <b>{inj.days_365} days</b> out in the last year
            {inj.games_missed_365 > 0
              ? <> · {inj.games_missed_365} game{inj.games_missed_365 === 1 ? '' : 's'} missed</>
              : null}
            {inj.spells_730 !== inj.spells_365
              ? <> · {inj.spells_730} spells in two years</> : null}
            {!cur && inj.days_since_return != null && inj.days_since_return < 400
              ? <> · back <b>{inj.days_since_return}d</b> ago</> : null}
          </span>
          {inj.recurring?.length > 0 && (
            <span className="tm-inj-line warn">
              recurring: {inj.recurring.join(', ')}
            </span>
          )}
          {inj.common?.length > 0 && !inj.recurring?.length && (
            <span className="tm-inj-line muted">
              {inj.common.map(([k, n]) => n > 1 ? `${k} x${n}` : k).join(' · ')}
            </span>
          )}
        </div>
      )}

      {club && (
        <div className="tm-inj-line muted">
          Manager <b>{club.manager}</b> — {club.days_in_post >= 400
            ? `${(club.days_in_post / 365.25).toFixed(1)} yr` 
            : `${club.days_in_post}d`} in post
          {club.new ? <span className="tm-tag new" style={{ marginLeft: 6 }}>new</span> : null}
        </div>
      )}
    </div>
  )
}
