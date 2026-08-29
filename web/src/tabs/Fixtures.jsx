import React, { useMemo, useState } from 'react'
import { useStore } from '../store'
import { badgeUrl, fdrColor } from '../util'

const MODES = [
  ['diff', 'Overall'], ['diff_att', 'Attacking'], ['diff_def', 'Defensive'],
  ['odds', 'Market'],
]

// Difficulty of one gw cell for sorting/averages: multi-fixture gws average,
// blanks count as a hard 4.5 (no fixture = no points).
const cellDiff = (fs, mode) => {
  if (!fs.length) return 4.5
  // `odds` is an OBJECT on the fixture, not a number, so the generic
  // `f[mode]` lookup would add an object into the running total and turn every
  // row average and the whole sort into NaN. Market mode maps the win
  // probability onto the same 1-5 scale; an unpriced fixture contributes
  // nothing and is excluded rather than counted as average.
  if (mode === 'odds') {
    const priced = fs.filter((f) => f.odds || f.market)
    if (!priced.length) return null
    return priced.reduce((a, f) => a + (5 - 4 * (f.odds || f.market).p_win), 0)
      / priced.length
  }
  return fs.reduce((a, f) => a + (f[mode] ?? f.fdr ?? 3), 0) / fs.length
}

export default function Fixtures() {
  const { fixtures, teams, status } = useStore()
  const [span, setSpan] = useState(10)
  const [mode, setMode] = useState('diff')
  // sort: key = 'team' | 'avg' | <gw number>; dir 1 = ascending (easiest first)
  const [sort, setSort] = useState({ key: 'avg', dir: 1 })

  const from = status?.next_gw || 1
  const gws = useMemo(() => {
    let scheduled = status?.scheduled_gws || []
    if (!scheduled.length && fixtures?.grid) {
      // DB not pulled yet — derive the calendar from the live fixture grid
      const s = new Set()
      for (const byGw of Object.values(fixtures.grid)) {
        for (const g of Object.keys(byGw)) s.add(Number(g))
      }
      scheduled = [...s].sort((a, b) => a - b)
    }
    return scheduled.filter((g) => g >= from).slice(0, span)
  }, [status, fixtures, from, span])

  const rows = useMemo(() => {
    if (!fixtures?.grid) return []
    const list = Object.entries(fixtures.grid).map(([tid, byGw]) => {
      const diffs = {}
      let sum = 0, seen = 0
      for (const g of gws) {
        const d = cellDiff(byGw[String(g)] || [], mode)
        diffs[g] = d
        // cellDiff returns null for a gameweek with no market price; averaging
        // only over what is priced beats inventing a 3.0 for the rest
        if (d != null) { sum += d; seen += 1 }
      }
      return { tid: Number(tid), team: teams[tid], byGw, diffs,
               avg: seen ? sum / seen : null }
    })
    const { key, dir } = sort
    list.sort((a, b) => {
      if (key === 'team') {
        return (a.team?.name || '').localeCompare(b.team?.name || '') * dir
      }
      // unpriced sorts last in both directions rather than pretending to be 4.5
      const va = (key === 'avg' ? a.avg : a.diffs[key]) ?? 999
      const vb = (key === 'avg' ? b.avg : b.diffs[key]) ?? 999
      return (va - vb) * dir || (a.team?.name || '').localeCompare(b.team?.name || '')
    })
    return list
  }, [fixtures, teams, gws, mode, sort])

  const clickSort = (key) => setSort((s) =>
    s.key === key ? { key, dir: -s.dir } : { key, dir: 1 })
  const arrow = (key) => (sort.key === key ? (sort.dir > 0 ? ' ▴' : ' ▾') : '')
  const th = (key, label, extra = {}) => (
    <th key={key} className={`num sortable ${sort.key === key ? 'sorted' : ''}`}
      onClick={() => clickSort(key)} style={extra}
      title={key === 'team' ? 'sort A–Z / Z–A'
        : 'click to sort — ▴ easiest first, ▾ hardest first'}>
      {label}{arrow(key)}
    </th>
  )

  const os_ = fixtures?.odds_status
  const marketUseless = mode === 'odds' && os_ && !os_.priced_upcoming

  return (
    <div className="panel">
      {marketUseless && (
        <div className="odds-warn">
          <b>No market prices for any upcoming fixture</b> — {os_.priced_upcoming} of{' '}
          {os_.upcoming} are priced.
          <ul>
            {!os_.key_set && (
              <li>
                <code>ODDS_API_KEY</code> is <b>not visible to the server</b>. The
                Odds API free tier is the only source here covering <i>upcoming</i>
                matches — 500 credits a month, 2 per call.
                <br />
                If you have already set it: <code>setx</code> (and <code>export</code>)
                only reach processes started <i>afterwards</i>, so a server that was
                already running cannot see it. Close this terminal, open a new one,
                restart <code>python -m app</code>, then hit ⟳ Data.
              </li>
            )}
            {os_.key_set && (
              <li>
                <code>ODDS_API_KEY</code> is set but nothing upcoming came back —
                the key may be rejected or out of credits. Check the pull log.
              </li>
            )}
            <li>
              football-data.co.uk{os_.sources?.includes('football-data')
                ? ` supplied ${os_.priced} priced fixture${os_.priced === 1 ? '' : 's'}`
                : ' supplied none'}, but it only publishes matches that have
              already been played — useful for backtests, never for planning.
            </li>
          </ul>
        </div>
      )}
      <div className="filterbar">
        <span className="section-label">Fixture difficulty</span>
        <div style={{ display: 'flex', gap: 4 }}>
          {MODES.map(([v, l]) => (
            <button key={v} className={`pill-btn ${mode === v ? 'accent' : ''}`}
              onClick={() => setMode(v)}
              title={v === 'diff_att' ? 'how hard the opponent is to score against'
                : v === 'diff_def' ? 'how hard it is to keep a clean sheet against the opponent'
                : v === 'odds' ? 'bookmaker win probability, where a price exists'
                : 'opponent overall strength'}>
              {l}
            </button>
          ))}
        </div>
        <span style={{ fontSize: 11.5, color: 'var(--muted-2)' }}>
          1 = easiest · 5 = hardest · click any column to sort
        </span>
        <span style={{ flex: 1 }} />
        <label style={{ fontSize: 12, color: 'var(--muted)', display: 'flex', alignItems: 'center', gap: 8 }}>
          Next
          <select className="pill-btn" value={span} style={{ background: 'var(--panel)', appearance: 'auto' }}
            onChange={(e) => setSpan(Number(e.target.value))}>
            {[5, 8, 10, 15, 20, 38].map((n) => <option key={n} value={n}>{n} GWs</option>)}
          </select>
        </label>
        <button className={`pill-btn ${sort.key === 'avg' ? 'accent' : ''}`}
          onClick={() => clickSort('avg')}>
          {sort.key === 'avg' && sort.dir < 0 ? 'Hardest first' : 'Easiest first'}
          {sort.key === 'avg' ? (sort.dir > 0 ? ' ▴' : ' ▾') : ''}
        </button>
      </div>
      <div className="fdr-table-wrap" style={{ padding: '0 10px 12px' }}>
        <table className="fdr">
          <thead>
            <tr>
              {th('team', 'Team', { textAlign: 'left', paddingLeft: 10 })}
              {gws.map((g) => th(g, `GW${g}`))}
              {th('avg', 'AVG')}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.tid}>
                <td className="team">
                  <img src={badgeUrl(r.team?.code)} alt=""
                    onError={(e) => { e.currentTarget.style.visibility = 'hidden' }} />
                  {r.team?.name || r.tid}
                </td>
                {gws.map((g) => {
                  const fs = r.byGw[String(g)] || []
                  if (!fs.length) {
                    return <td key={g}><div className="fdr-cell fdr-blank">–</div></td>
                  }
                  return (
                    <td key={g} className={sort.key === g ? 'sortcol' : ''}>
                      {fs.map((f, i) => {
                        const opp = teams[String(f.opp)]?.short || '?'
                        // In Market mode a 5.0 is a near-certain loss and a
                        // 1.0 a near-certain win, so the palette keeps meaning.
                        // A fixture nobody has priced renders as UNPRICED -
                        // quietly falling back to FDR here would dress a
                        // difficulty rating up as a market view.
                        // Market mode falls back to the prediction market
                        // when no bookmaker has priced the fixture — they
                        // measure the same thing, so either is a market view.
                        const mkt = f.odds || f.market
                        const unpriced = mode === 'odds' && !mkt
                        const v = mode === 'odds'
                          ? (mkt ? 5 - 4 * mkt.p_win : 3)
                          : (f[mode] ?? f.fdr ?? 3)
                        const c = fdrColor(v)
                        return (
                          <div key={i} className={`fdr-cell ${unpriced ? 'unpriced' : ''}`}
                            style={unpriced
                              ? { ...(i ? { marginTop: 3 } : {}) }
                              : { background: c.bg, color: c.fg, ...(i ? { marginTop: 3 } : {}) }}
                            title={[
                              `${f.home ? 'Home vs' : 'Away at'} ${teams[String(f.opp)]?.name}`,
                              `FPL FDR ${f.fdr ?? '–'}`,
                              // the market's own read, when a bookmaker has
                              // priced it. Shown raw so the judgement is the
                              // user's - the model already uses these itself.
                              f.odds && `market: win ${(f.odds.p_win * 100).toFixed(0)}%`
                                + ` · draw ${(f.odds.p_draw * 100).toFixed(0)}%`
                                + ` · lose ${(f.odds.p_lose * 100).toFixed(0)}%`,
                              f.odds && `implied goals ${f.odds.xg.toFixed(2)}`
                                + ` – ${f.odds.xg_against.toFixed(2)} (${f.odds.source})`,
                              f.market && `prediction market: win `
                                + `${(f.market.p_win * 100).toFixed(0)}%`
                                + ` (${f.market.source}, $${Math.round(f.market.liquidity).toLocaleString()} liquidity)`,
                              f.odds && f.market && `they disagree by `
                                + `${((f.market.p_win - f.odds.p_win) * 100).toFixed(1)} pts`,
                            ].filter(Boolean).join(' · ')}>
                            {f.home ? opp : <span className="away">{opp.toLowerCase()}</span>}
                            <span className="val">
                              {unpriced ? 'no price'
                                : mode === 'odds'
                                  ? `${(mkt.p_win * 100).toFixed(0)}%`
                                  : Number(v).toFixed(1)}
                            </span>
                            {f.odds && <span className="odds-dot" title="bookmaker price available" />}
                            {/* the two markets parting company by more than a
                                few points is the only part of this worth an
                                eye — agreement is the normal case */}
                            {f.odds && f.market
                              && Math.abs(f.market.p_win - f.odds.p_win) >= 0.04 && (
                              <span className="pm-diverge"
                                title={`prediction market ${(f.market.p_win * 100).toFixed(0)}%`
                                  + ` vs bookmaker ${(f.odds.p_win * 100).toFixed(0)}%`}>
                                {f.market.p_win > f.odds.p_win ? '▲' : '▼'}
                              </span>
                            )}
                          </div>
                        )
                      })}
                    </td>
                  )
                })}
                <td>
                  {r.avg == null
                    ? <div className="fdr-cell unpriced"><span className="val">–</span></div>
                    : (() => { const c = fdrColor(r.avg); return (
                      <div className="fdr-cell" style={{ background: c.bg, color: c.fg }}>
                        {r.avg.toFixed(2)}
                      </div>
                    ) })()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
