import React, { useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import { useStore } from '../store'
import Flag from '../components/Flag'
import PlayerModal from '../components/PlayerModal'
import { badgeUrl } from '../util'
import { Empty, ErrorState, TableSkeleton } from '../components/States'

/* Who is about to rise or fall - and what that is actually worth.

   The model discriminates strongly: held out forward in time, the top-10
   ranked risers actually rose 67% (2024-25) / 75% (2025-26) of the time
   against a 2% base rate. That hit rate is seductive, which is exactly why
   every row also carries the points conversion. A rise is realised only on
   sale, FPL hands back the purchase price plus HALF the profit, and that half
   then buys a marginally better squad for whatever season is left - worth
   0.163 points per £1m per gameweek. The strongest riser in a week comes to
   about 0.2 points.

   So this is a tie-breaker between transfers you already rate, and the page is
   built to say so rather than to let price masquerade as a strategy. */

function PctBar({ v, color }) {
  return (
    <div className="prc-bar">
      <div style={{ width: `${Math.round(v * 100)}%`, background: color }} />
    </div>
  )
}

export default function Prices() {
  const { byId, teams } = useStore()
  const [data, setData] = useState(null)
  const [err, setErr] = useState(null)
  const [modalPid, setModalPid] = useState(null)
  const [side, setSide] = useState('risers')

  useEffect(() => {
    api.prices(40).then(setData).catch((e) => setErr(e.message))
  }, [])

  const rows = useMemo(() => {
    const src = (side === 'risers' ? data?.risers : data?.fallers) || []
    return src
      .map((r) => {
        const p = byId.get(r.player_id)
        return { ...r, p, team: teams[String(p?.team_id)] }
      })
      .filter((r) => r.p)
  }, [data, side, byId, teams])

  if (err) {
    return (
      <div className="panel">
        <ErrorState title="Could not load price predictions">{err}</ErrorState>
      </div>
    )
  }
  if (!data) {
    return (
      <div className="panel">
        <div className="panel-head">Predicted price changes</div>
        <TableSkeleton rows={10} />
      </div>
    )
  }
  if (!data.ok) {
    return (
      <div className="panel">
        <Empty mark="£" title="No price model output yet"
          actions={<span className="ml-note">{data.error}</span>}>
          The price model needs this season&apos;s transfer and ownership history.
          Run a data pull (⟳ Data, top right), then reload this tab.
        </Empty>
      </div>
    )
  }

  const top = rows[0]
  return (
    <div>
      <div className="panel" style={{ marginBottom: 14 }}>
        <div className="panel-head">Predicted price changes — after GW{data.gw}</div>
        <div className="prc-explain">
          <p>
            Ranked by P(rise) − P(fall). Held out forward in time, the top-10
            risers actually rose <b>67–75%</b> of the time against a <b>2%</b> base
            rate — the model is strong.
          </p>
          <p>
            What it is <i>worth</i> is a different question, and the answer is
            deliberately small. A rise is realised only on sale, and FPL returns the
            purchase price plus <b>half</b> the profit; that half buys a better squad
            at <b>{data.pts_per_million_per_gw} points per £1m per gameweek</b>, over
            the <b>{data.gws_remaining}</b> gameweeks left.
            {top ? (
              <> The strongest mover here is worth{' '}
                <b>{Math.abs(top.points).toFixed(2)} points</b>.</>
            ) : null}
          </p>
          <p className="ml-note">
            Use it to break ties between transfers you already rate. It is
            deliberately <i>not</i> in the solver&apos;s objective — a 0.2-point term
            has no business reshaping a squad.
          </p>
        </div>
      </div>

      <div className="toolbar" style={{ marginBottom: 10 }}>
        {[['risers', '▲ Rising'], ['fallers', '▼ Falling']].map(([v, l]) => (
          <button key={v} className={`pill-btn ${side === v ? 'accent' : ''}`}
            onClick={() => setSide(v)}>{l}</button>
        ))}
        <span className="ml-note" style={{ marginLeft: 8 }}>
          {rows.length} players · click a row for the full card
        </span>
      </div>

      <div className="panel">
        <div className="ptable-wrap">
        <table className="ptable prc-table">
          <thead>
            <tr>
              <th className="l">Player</th>
              <th>Price</th>
              <th>P(rise)</th>
              <th>P(fall)</th>
              <th>E[move]</th>
              <th>Worth</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.player_id}>
                <td className="l">
                  <div className="pl-cell clickable" role="button" tabIndex={0}
                    onClick={() => setModalPid(r.player_id)}
                    onKeyDown={(e) => { if (e.key === 'Enter') setModalPid(r.player_id) }}>
                    <img src={badgeUrl(r.team?.code)} alt="" loading="lazy"
                      onError={(e) => { e.currentTarget.style.visibility = 'hidden' }} />
                    <div>
                      <div className="nm">
                        {r.p.web_name}
                        <Flag p={{ status: r.p.status, news: r.p.news, chance: r.p.chance }} />
                      </div>
                      <div className="sub">{r.team?.short} · {r.p.position}</div>
                    </div>
                  </div>
                </td>
                <td className="num" style={{ color: 'var(--muted)' }}>
                  £{r.price.toFixed(1)}m
                </td>
                <td>
                  <PctBar v={r.p_rise} color="#199e70" />
                  <span className="prc-n">{(r.p_rise * 100).toFixed(0)}%</span>
                </td>
                <td>
                  <PctBar v={r.p_fall} color="#d95926" />
                  <span className="prc-n">{(r.p_fall * 100).toFixed(0)}%</span>
                </td>
                <td className="num">
                  {r.e_delta >= 0 ? '+' : ''}{(r.e_delta * 10).toFixed(2)}
                  <span className="prc-unit"> tenths</span>
                </td>
                <td className="num" style={{
                  fontWeight: 700,
                  color: r.points >= 0 ? 'var(--text)' : 'var(--red)',
                }}>
                  {r.points >= 0 ? '+' : ''}{r.points.toFixed(2)} pts
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>
      </div>

      {modalPid && (
        <PlayerModal pid={modalPid} close={() => setModalPid(null)} draft={null} />
      )}
    </div>
  )
}
