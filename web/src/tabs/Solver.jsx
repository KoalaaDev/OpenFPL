import React, { useEffect, useMemo, useRef, useState } from 'react'
import { api, pollJob } from '../api'
import { useStore, usePersisted } from '../store'
import RadarLoader from '../components/RadarLoader'
import SolverOutput from '../components/SolverOutput'
import { useDialog } from '../components/Dialog'
import { CHIP_NAME, CHIP_SHORT, chipAvailability, chipNote, fmt1, money,
         planToDraft, withBaseline } from '../util'

const CHIP_DEFS = [
  ['wildcard', 'WC'], ['freehit', 'FH'], ['bench_boost', 'BB'], ['triple_captain', 'TC'],
]

export default function Solver({ goPlanner }) {
  const { form } = useDialog()
  const { status, entryId, entry, players, teams, byId, setDrafts,
          setActiveDraftId, setToast, refreshProjections } = useStore()

  const [horizon, setHorizon] = usePersisted('solver.horizon', 5)
  const [solveFrom, setSolveFrom] = usePersisted('solver.from', null)
  const [ftValue, setFtValue] = usePersisted('solver.ftValue', 1.5)
  const [decay, setDecay] = usePersisted('solver.decay', 0.85)
  // which strategies to solve. Each is a different *preference* — how far
  // ahead to look, whether hits are allowed, and whether it wants points that
  // arrive in bursts or every week — so they are chosen, not counted.
  const [styles, setStyles] = usePersisted('solver.styles',
    ['aggressive', 'balanced', 'conservative'])
  const [advanced, setAdvanced] = usePersisted('solver.advanced', false)
  const [hitCost, setHitCost] = usePersisted('solver.hitCost', 4)
  const [benchWeight, setBenchWeight] = usePersisted('solver.benchWeight', 0.1)
  const [maxTransfers, setMaxTransfers] = usePersisted('solver.maxTransfers', 3)
  const [timeLimit, setTimeLimit] = usePersisted('solver.timeLimit', 60)
  const [keepPerPos, setKeepPerPos] = usePersisted('solver.keepPerPos', 30)
  const [ftOverride, setFtOverride] = usePersisted('solver.ftOverride', '')
  const [useEntry, setUseEntry] = usePersisted('solver.useEntry', true)

  const [chips, setChips] = usePersisted('solver.chips', {})       // name -> {enabled, force}
  // Whether the solve may play chips at all, and how keen it is. Chips used to
  // be opt-in one by one and defaulted to OFF, so an untouched solve never
  // played one and read as a solver that ignores chips.
  //   worth  — every chip you hold, played only when it beats its keep-value
  //   always — every chip you hold, played whenever it adds points this horizon
  //   off    — none (a chip already activated on FPL is still honoured)
  const [chipMode, setChipMode] = usePersisted('solver.chipMode', 'worth')
  const [chipSkip, setChipSkip] = usePersisted('solver.chipSkip', {})   // name -> true
  const bars = status?.chip_reserve_now || null
  const [locked, setLocked] = usePersisted('solver.locked', [])
  const [avoid, setAvoid] = usePersisted('solver.avoid', [])
  const [banned, setBanned] = usePersisted('solver.banned', [])
  const [sellTeams, setSellTeams] = usePersisted('solver.sellTeams', [])
  const [minFt, setMinFt] = usePersisted('solver.minFt', [])       // [{gw, n}]

  const [running, setRunning] = useState(false)
  const [job, setJob] = useState(null)
  const [result, setResult] = useState(null)
  const [showOut, setShowOut] = useState(false)
  const logRef = useRef(null)

  // never earlier than the first open deadline: a gameweek in progress is locked
  const openGw = status?.editable_gw ?? status?.next_gw ?? 1
  const from = Math.max(solveFrom || 0, openGw)
  const gws = useMemo(() =>
    (status?.scheduled_gws || []).filter((g) => g >= from).slice(0, horizon),
    [status, from, horizon])

  // What FPL says this entry still holds. A chip already spent is not a
  // planning option, and a chip already ACTIVATED for the coming deadline is
  // not a choice either — it will be played whatever the solver decides, so
  // it is pinned to the first gameweek with no option value. The backend
  // enforces both; this is the same truth on screen.
  const chipState = useEntry ? entry?.chips : null
  const avail = useMemo(() => chipAvailability(chipState, gws), [chipState, gws])
  const activeChip = chipState?.active || null
  useEffect(() => {
    if (!activeChip || !gws.length) return
    if (chipState?.next_gw && chipState.next_gw !== gws[0]) return
    setChips((s) => (s[activeChip]?.enabled && s[activeChip]?.force === gws[0]
      ? s : { ...s, [activeChip]: { enabled: true, force: gws[0] } }))
  }, [activeChip, chipState?.next_gw, gws])

  const start = async () => {
    if (running) return
    setRunning(true)
    setResult(null)
    setJob({ progress: [], pct: 0 })
    const chipParams = {}
    const reserve = {}
    for (const [name] of CHIP_DEFS) {
      const c = chips[name] || {}
      const a = avail[name]
      if (!a?.usable) continue
      const live = activeChip === name
      const pinned = !!(c.enabled && c.force)
      const auto = chipMode !== 'off' && !chipSkip[name]
      if (!(auto || pinned || live)) continue
      chipParams[name] = { enabled: true, force: live ? gws[0] : (pinned ? c.force : null) }
      if (chipMode === 'always') reserve[name] = 0
    }
    try {
      const { job_id } = await api.solve({
        entry: useEntry ? entryId : null,
        solve_from: from, horizon,
        decay, ft_value: ftValue, hit_cost: hitCost, bench_weight: benchWeight,
        max_transfers: maxTransfers, time_limit: timeLimit,
        keep_per_position: keepPerPos,
        playstyles: styles.length ? styles : ['balanced'],
        n_plans: Math.max(1, styles.length),
        free_transfers: ftOverride === '' ? null : Number(ftOverride),
        chips: chipParams,
        chip_reserve: reserve,
        locked, avoid, banned_teams: banned, sell_teams: sellTeams,
        min_ft: Object.fromEntries(minFt.map((m) => [m.gw, m.n])),
      })
      const res = await pollJob(job_id, (j) => {
        setJob(j)
        requestAnimationFrame(() => {
          logRef.current?.scrollTo(0, logRef.current.scrollHeight)
        })
      })
      setResult(res); setShowOut(true)
      refreshProjections()
      setToast({ kind: 'ok', msg: `Solve finished — ${res.plans.length} plan(s).` })
    } catch (e) {
      setToast({ kind: 'err', msg: `Solve failed: ${e.message}` })
    } finally {
      setRunning(false)
    }
  }

  const addDraft = (plan, i) => {
    setDrafts((ds) => {
      const label = `Solve ${String.fromCharCode(65 + (ds.length % 26))}`
      const d = withBaseline(planToDraft(plan, result, label, `plan ${i + 1}`))
      setActiveDraftId(d.id)
      return [...ds, d]
    })
    setToast({ kind: 'ok', msg: 'Added to Planner as a draft.' })
    goPlanner()
  }

  return (
    <div className="solver-grid">
      {/* ---------------- parameters column ---------------- */}
      <div>
        <div className="panel" style={{ padding: 16, marginBottom: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: 14 }}>
            <span className="section-label">Parameters &amp; decisions</span>
            <span style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--muted)' }}>
              solve from{' '}
              <select className="pill-btn" value={from} style={{ background: 'var(--panel)', appearance: 'auto' }}
                onChange={(e) => setSolveFrom(Number(e.target.value))}>
                {(status?.scheduled_gws || []).slice(0, 20).map((g) =>
                  <option key={g} value={g}>GW {g}</option>)}
              </select>
            </span>
          </div>

          <div className="field">
            <div className="lbl">
              <span className="section-label">Transfer depth</span>
              <span className="hintdot" title="How many gameweeks the optimiser plans over">i</span>
              <span className="big-num" style={{ marginLeft: 'auto' }}>{horizon} <span style={{ fontSize: 12 }}>GWs</span></span>
            </div>
            <div className="slider-row">
              <input type="range" min={1} max={8} value={horizon}
                onChange={(e) => setHorizon(Number(e.target.value))} />
            </div>
            <div style={{ fontSize: 11, color: 'var(--muted-2)', marginTop: 4 }}>
              up to GW {gws[gws.length - 1] ?? '–'} · deeper = slower solve
            </div>
          </div>

          <StylePicker styles={styles} setStyles={setStyles}
            defs={status?.playstyles || []} />

          <button className="pill-btn" style={{ marginTop: 6 }}
            onClick={() => setAdvanced((a) => !a)}>
            Advanced settings {advanced ? '▴' : '▾'}
          </button>
          {advanced && (
            <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap', marginTop: 14 }}>
              <div style={{ flexBasis: '100%', fontSize: 11, color: 'var(--muted-2)' }}>
                Each strategy sets its own horizon weighting, free-transfer value
                and hit policy, so these two are only used when you solve a single
                strategy on its own.
              </div>
              <NumField label="FT value" hint="Bonus points for each banked free transfer at horizon end"
                value={ftValue} step={0.25} onChange={setFtValue} />
              <NumField label="Time decay" hint="Weight per future gameweek (uncertainty discount)"
                value={decay} step={0.05} onChange={(v) => setDecay(Math.min(1, Math.max(0.5, v)))} />
              <NumField label="Hit cost" value={hitCost} step={1} onChange={setHitCost} />
              <NumField label="Bench weight" hint="How much bench points matter (autosub proxy)"
                value={benchWeight} step={0.05} onChange={setBenchWeight} />
              <NumField label="Max transfers / GW" value={maxTransfers} step={1}
                onChange={(v) => setMaxTransfers(Math.max(1, Math.round(v)))} />
              <NumField label="Solver seconds" value={timeLimit} step={15}
                onChange={(v) => setTimeLimit(Math.max(10, Math.round(v)))} />
              <NumField label="Pool / position" hint="Players kept per position in the MILP"
                value={keepPerPos} step={5} onChange={(v) => setKeepPerPos(Math.max(10, Math.round(v)))} />
              <div className="field">
                <div className="lbl"><span className="section-label">FTs override</span></div>
                <input className="num" placeholder="auto" value={ftOverride}
                  onChange={(e) => setFtOverride(e.target.value.replace(/\D/g, ''))}
                  style={{ width: 90, background: 'var(--bg-deep)', border: '1px solid var(--line)',
                           borderRadius: 7, padding: '9px 10px', outline: 'none' }} />
              </div>
              <div className="field">
                <div className="lbl"><span className="section-label">Use entry squad</span></div>
                <button className={`pill-btn ${useEntry ? 'accent' : ''}`}
                  onClick={() => setUseEntry((u) => !u)}>
                  {useEntry ? (entry?.team_name || `#${entryId}`) : 'fresh £100m squad'}
                </button>
              </div>
            </div>
          )}
        </div>

        <div className="panel" style={{ padding: 16 }}>
          <div className="section-label" style={{ color: 'var(--accent)', marginBottom: 10 }}>Chips</div>
          <div className="chip-mode" role="radiogroup" aria-label="chip use">
            {[
              ['off', 'Don’t use chips', 'plan transfers only'],
              ['worth', 'When worth it', 'only if it beats saving the chip'],
              ['always', 'Whenever it helps', 'play any chip that adds points'],
            ].map(([k, l, sub]) => (
              <button key={k} role="radio" aria-checked={chipMode === k}
                className={`cm-opt ${chipMode === k ? 'on' : ''}`} onClick={() => setChipMode(k)}>
                <b>{l}{k === 'worth' ? <em> recommended</em> : null}</b>
                <span>{sub}</span>
              </button>
            ))}
          </div>
          <div className="chipplan">
            {CHIP_DEFS.map(([name, short]) => {
              const c = chips[name] || {}
              const a = avail[name]
              const live = a?.active
              const pin = c.enabled && c.force ? c.force : null   // a stale force from old settings is not a pin
              const included = a?.usable && (live || pin || (chipMode !== 'off' && !chipSkip[name]))
              return (
                <span key={name}
                  className={`chip-btn ${included ? 'on' : ''} ${!a?.usable ? 'spent' : ''}`}
                  title={!a?.usable ? chipNote(a, CHIP_NAME[name])
                    : live ? 'activated on your FPL team — pinned to this gameweek'
                      : chipMode === 'off' ? 'pin a gameweek to force this chip even with chips off'
                        : included ? 'the solver may play this — click to leave it out' : 'left out — click to include'}>
                  <button disabled={!a?.usable || live || chipMode === 'off'}
                    onClick={() => setChipSkip((sk) => ({ ...sk, [name]: !sk[name] }))}>
                    ⚡ {short}
                    {live && <em> live</em>}
                    {!a?.usable && a?.played?.length ? <em> used GW{a.played[a.played.length - 1]}</em> : null}
                  </button>
                  {a?.usable && !live && (
                    <select value={pin || ''} title="force it into a specific gameweek"
                      onChange={(e) => {
                        const gw = e.target.value ? Number(e.target.value) : null
                        setChips((st) => ({ ...st, [name]: { ...c, enabled: !!gw, force: gw } }))
                      }}>
                      <option value="">{chipMode === 'off' ? 'off' : 'any GW'}</option>
                      {gws.filter((g) => !a?.known
                        || a.windows.some(([x, y]) => g >= x && g <= y))
                        .map((g) => <option key={g} value={g}>GW{g}</option>)}
                    </select>
                  )}
                </span>
              )
            })}
          </div>
          <div className="chip-explain">
            {activeChip ? (
              <p><b>{CHIP_NAME[activeChip]} is live</b> on your FPL team for
                GW{chipState?.next_gw ?? gws[0]} — it is pinned there whatever you pick.</p>
            ) : null}
            {chipMode === 'worth' && (
              <p>A chip is only played when this gameweek&apos;s gain beats what the chip
                is worth <i>kept</i> for a better week later
                {bars ? <> — right now about <b>+{bars.triple_captain}</b> for Triple Captain,
                  <b> +{bars.bench_boost}</b> for Bench Boost, <b>+{bars.freehit}</b> for Free Hit
                  and <b>+{bars.wildcard}</b> for Wildcard</> : null}.
                The bar falls as the season runs out. The results say why each chip was held.</p>
            )}
            {chipMode === 'always' && (
              <p>Any chip that adds points inside the horizon gets played — usually the
                first decent week. That spends chips you would likely get more from
                later; use it to see what a chip is worth now.</p>
            )}
            {chipMode === 'off' && (
              <p>Transfers only. Pick a gameweek on a chip to force it anyway.</p>
            )}
          </div>
        </div>
      </div>

      {/* ---------------- constraints + run column ---------------- */}
      <div>
        <div className="panel" style={{ padding: 16, marginBottom: 16 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <PlayerPicker label="Target / hold players" color="green" list={locked}
              setList={setLocked} players={players} byId={byId} exclude={avoid} />
            <PlayerPicker label="Avoid / sell players" color="red" list={avoid}
              setList={setAvoid} players={players} byId={byId} exclude={locked} />
          </div>
          <div style={{ marginTop: 16 }}>
            <div className="lbl" style={{ marginBottom: 8 }}>
              <span className="section-label">📤 Sell out of teams</span>
              <span className="hintdot" title="Players you already own from these teams must be sold by the end of the horizon. The solver picks the cheapest gameweek to do it, so it uses free transfers where it can.">i</span>
            </div>
            <div className="tagbox">
              {!sellTeams.length && <span className="empty">No teams to sell out of.</span>}
              {sellTeams.map((tid) => (
                <span className="tag red" key={tid}>
                  {teams[String(tid)]?.name || tid}
                  <button onClick={() => setSellTeams((b) => b.filter((x) => x !== tid))}>×</button>
                </span>
              ))}
              <select className="pill-btn" value="" style={{ background: 'var(--panel)', appearance: 'auto' }}
                onChange={(e) => {
                  const v = Number(e.target.value)
                  if (v && !sellTeams.includes(v)) setSellTeams((b) => [...b, v])
                }}>
                <option value="">+ add team</option>
                {Object.values(teams).sort((a, b) => a.name.localeCompare(b.name))
                  .map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
              </select>
            </div>
          </div>
          <div style={{ marginTop: 16 }}>
            <div className="lbl" style={{ marginBottom: 8 }}>
              <span className="section-label">🚫 Do not buy from teams</span>
            </div>
            <div className="tagbox">
              {!banned.length && <span className="empty">No teams excluded.</span>}
              {banned.map((tid) => (
                <span className="tag red" key={tid}>
                  {teams[String(tid)]?.name || tid}
                  <button onClick={() => setBanned((b) => b.filter((x) => x !== tid))}>×</button>
                </span>
              ))}
              <select className="pill-btn" value="" style={{ background: 'var(--panel)', appearance: 'auto' }}
                onChange={(e) => {
                  const v = Number(e.target.value)
                  if (v && !banned.includes(v)) setBanned((b) => [...b, v])
                }}>
                <option value="">+ add team</option>
                {Object.values(teams).sort((a, b) => a.name.localeCompare(b.name))
                  .map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
              </select>
            </div>
          </div>
          <div style={{ marginTop: 16 }}>
            <div className="lbl" style={{ marginBottom: 8 }}>
              <span className="section-label">→ Minimum available FTs</span>
            </div>
            <div className="tagbox">
              {!minFt.length && <span className="empty">No FT targets configured.</span>}
              {minFt.map((m, i) => (
                <span className="tag" key={i}>
                  GW{m.gw}: ≥{m.n} FT
                  <button onClick={() => setMinFt((l) => l.filter((_, k) => k !== i))}>×</button>
                </span>
              ))}
              <button className="pill-btn" onClick={async () => {
                const v = await form({
                  title: 'Hold free transfers', confirmLabel: 'Add target',
                  body: 'The solver will keep at least this many free transfers banked after the gameweek\u2019s moves.',
                  fields: [
                    { name: 'gw', label: 'Gameweek', type: 'number', value: String(gws[0] ?? ''),
                      min: gws[0], max: gws[gws.length - 1], hint: `GW${gws[0]}–GW${gws[gws.length - 1]}` },
                    { name: 'n', label: 'Minimum free transfers', type: 'number', value: '2', min: 1, max: 5 },
                  ],
                  validate: (x) => {
                    const g = Number(x.gw), n = Number(x.n)
                    if (!gws.includes(g)) return `Pick a gameweek between ${gws[0]} and ${gws[gws.length - 1]}.`
                    if (!(n >= 1 && n <= 5)) return 'Between 1 and 5 free transfers.'
                    return null
                  },
                })
                if (v) setMinFt((l) => [...l, { gw: Number(v.gw), n: Number(v.n) }])
              }}>+ add target</button>
            </div>
          </div>
        </div>

        <div className="panel" style={{ padding: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <div style={{ fontSize: 12, color: 'var(--muted)' }}>
              {useEntry
                ? (entry?.squad
                  ? <>Optimising <b>{entry.team_name}</b> · {money(entry.bank)} itb · {entry.free_transfers} FT</>
                  : <>Pre-season: building a fresh £100m squad for <b>{entry?.team_name || `#${entryId}`}</b></>)
                : 'Building a fresh £100m squad'}
              {' '}· GW{gws[0]}–{gws[gws.length - 1]}
            </div>
            <button className="pill-btn accent" style={{ marginLeft: 'auto', padding: '12px 22px', fontSize: 13 }}
              onClick={start} disabled={running}>
              {running ? <span className="spinner" /> : '⚡'} START SOLVE
            </button>
          </div>

          {(running || job?.progress?.length > 0) && (
            <div style={{ marginTop: 14 }}>
              {/* A solve is 30-60s of MILP with nothing to preview, which is
                  exactly the wait the radar is for. It carries the real
                  progress line rather than spinning decoratively. */}
              {running && (
                <RadarLoader inline size={116} label="Solving"
                  sub={(job?.progress || []).slice(-1)[0]?.msg || 'Setting up…'} />
              )}
              <div className="progressbar" style={{ marginBottom: 8 }}>
                <div style={{ width: `${Math.round((job?.pct || 0) * 100)}%` }} />
              </div>
              <div className="solve-log" ref={logRef}>
                {(job?.progress || []).map((p, i, arr) => (
                  <div key={i} className={i === arr.length - 1 ? 'last' : ''}>{p.msg}</div>
                ))}
              </div>
            </div>
          )}

          {result && !showOut && (
            <div className="so-reopen">
              <span>
                {result.plans.length} plan{result.plans.length === 1 ? '' : 's'} ready
              </span>
              <button className="pill-btn accent" onClick={() => setShowOut(true)}>
                ▥ Open results
              </button>
            </div>
          )}
        </div>
      </div>
      {result && showOut && (
        <SolverOutput result={result} close={() => setShowOut(false)}
          addDraft={(plan, k) => { addDraft(plan, k); setShowOut(false) }}
          draftLabel={null} />
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ */

/* The strategy is the first real decision on this tab, so it is a choice
   between three described things rather than a "how many plans?" number.
   The definitions come from the engine (status.playstyles) so the screen and
   the solver can never drift apart. */
function StylePicker({ styles, setStyles, defs }) {
  const list = defs.length ? defs : [{ key: 'balanced', label: 'Balanced', note: '' }]
  const toggle = (k) => setStyles((cur) => {
    const has = cur.includes(k)
    // never solve nothing: the last one on stays on
    if (has && cur.length === 1) return cur
    return has ? cur.filter((x) => x !== k) : [...cur, k]
  })
  return (
    <div className="field">
      <div className="lbl">
        <span className="section-label">Strategy</span>
        <span className="hintdot" title="One plan is solved per strategy selected. They differ in what they prefer, never in the rules — a -4 always costs 4.">i</span>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--muted-2)' }}>
          {styles.length} plan{styles.length === 1 ? '' : 's'}
        </span>
      </div>
      <div className="style-picker">
        {list.map((d) => {
          const on = styles.includes(d.key)
          return (
            <button key={d.key} className={`style-card ${on ? 'on' : ''}`}
              onClick={() => toggle(d.key)} title={d.detail || d.note}>
              <span className="sc-head">
                <span className="sc-tick" aria-hidden="true">{on ? '✓' : ''}</span>
                {d.label}
              </span>
              <span className="sc-note">{d.note}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

function NumField({ label, hint, value, step, onChange }) {
  return (
    <div className="field">
      <div className="lbl">
        <span className="section-label">{label}</span>
        {hint && <span className="hintdot" title={hint}>i</span>}
      </div>
      <div className="numctl">
        <button onClick={() => onChange(Math.round((value - step) * 100) / 100)}>−</button>
        <input className="num" value={value}
          onChange={(e) => { const v = Number(e.target.value); if (!Number.isNaN(v)) onChange(v) }} />
        <button onClick={() => onChange(Math.round((value + step) * 100) / 100)}>+</button>
      </div>
    </div>
  )
}

function PlayerPicker({ label, color, list, setList, players, byId, exclude }) {
  const [q, setQ] = useState('')
  const opts = useMemo(() => {
    if (!q) return []
    const lq = q.toLowerCase()
    return players
      .filter((p) => !list.includes(p.id) && !exclude.includes(p.id))
      .filter((p) => p.web_name.toLowerCase().includes(lq) || p.name.toLowerCase().includes(lq))
      .sort((a, b) => b.own - a.own)
      .slice(0, 8)
  }, [q, players, list, exclude])

  return (
    <div>
      <div className="lbl" style={{ marginBottom: 8 }}>
        <span className="section-label" style={{ color: color === 'green' ? 'var(--green)' : 'var(--red)' }}>
          {color === 'green' ? '🔒' : '⛔'} {label}
        </span>
      </div>
      <div className="typeahead">
        <div className="search" style={{ minWidth: 0, marginBottom: 8 }}>
          🔍<input placeholder="Add player…" value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        {opts.length > 0 && (
          <div className="ta-list">
            {opts.map((p) => (
              <div key={p.id} className="ta-item"
                onClick={() => { setList((l) => [...l, p.id]); setQ('') }}>
                <div>
                  <div style={{ fontWeight: 700 }}>{p.web_name}</div>
                  <div className="sub">{p.position} · own {p.own}%</div>
                </div>
                <span className="pr num">£{p.price.toFixed(1)}m</span>
              </div>
            ))}
          </div>
        )}
      </div>
      <div className="tagbox">
        {!list.length && <span className="empty">No players selected.</span>}
        {list.map((pid) => (
          <span className={`tag ${color}`} key={pid}>
            {byId.get(pid)?.web_name || pid}
            <button onClick={() => setList((l) => l.filter((x) => x !== pid))}>×</button>
          </span>
        ))}
      </div>
    </div>
  )
}

function PlanCard({ plan, i, addDraft, byId, freeFirst }) {
  const [open, setOpen] = useState(i === 0)
  const nm = (t) => t.name || byId.get(t.player_id)?.web_name || t.player_id
  return (
    <div className="plan-card">
      <div className="head" onClick={() => setOpen((o) => !o)} style={{ cursor: 'pointer' }}>
        <span className="chip pink num">#{i + 1}</span>
        {plan.style_label || `Plan ${i + 1}`}
        {/* total_ep is undecayed and net of hits, so it compares fairly across
            playstyles; obj is not comparable (each style weights gws its own way) */}
        <span className="num" style={{ color: 'var(--muted)', fontSize: 12 }}>
          {plan.total_ep != null ? `${fmt1(plan.total_ep)} pts` : `obj ${fmt1(plan.objective)}`}
        </span>
        {plan.style_note && (
          <span style={{ color: 'var(--muted-2)', fontSize: 11, fontStyle: 'italic',
                         marginLeft: 8, maxWidth: 320, overflow: 'hidden',
                         textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                title={plan.style_note}>
            {plan.style_note}
          </span>
        )}
        <span style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          <button className="pill-btn accent"
            onClick={(e) => { e.stopPropagation(); addDraft(plan, i) }}>
            + Add to Planner
          </button>
          <span style={{ color: 'var(--muted-2)' }}>{open ? '▴' : '▾'}</span>
        </span>
      </div>
      {open && plan.style_detail && (
        <div className="plan-note style">{plan.style_detail}</div>
      )}
      {open && freeFirst && (
        <div className="plan-note">
          Pre-deadline: GW{plan.per_gw[0]?.gw} moves are free (FPL's unlimited transfers
          before the first deadline) — hits and free-transfer limits apply from the next GW.
        </div>
      )}
      {open && plan.per_gw.map((g, gi) => (
        <div className="plan-gw" key={g.gw}>
          <span className="g">GW{g.gw}</span>
          <div>
            {g.chip && <span className="chip gold" style={{ marginRight: 8 }}>{CHIP_SHORT[g.chip]}</span>}
            {freeFirst && gi === 0 && g.transfers_in.length > 0 && (
              <span className="chip green" style={{ marginRight: 8 }}>free rebuild</span>
            )}
            {g.transfers_out.length === 0 && !g.chip && (
              <span style={{ color: 'var(--muted-2)' }}>roll</span>
            )}
            {/* A Free Hit is not a transfer, so transfers_in/out are empty by
                design. fh_out -> fh_in is what the chip actually does. */}
            {(g.fh_out || []).map((o, k) => (
              <div key={`fh${k}`} style={{ marginBottom: 2 }}>
                <span style={{ color: 'var(--muted)' }}>{nm(o)}</span>
                <span style={{ color: 'var(--gold, var(--accent))', margin: '0 8px' }}>⇢</span>
                <span style={{ fontWeight: 600 }}>{nm(g.fh_in[k])}</span>
              </div>
            ))}
            {g.transfers_out.map((o, k) => (
              <div key={k} style={{ marginBottom: 2 }}>
                <span style={{ color: 'var(--muted)' }}>{nm(o)}</span>
                <span style={{ color: 'var(--accent)', margin: '0 8px' }}>→</span>
                <span style={{ fontWeight: 600 }}>{nm(g.transfers_in[k])}</span>
              </div>
            ))}
            <div style={{ fontSize: 11, color: 'var(--muted-2)', marginTop: 3 }}>
              XI {fmt1(g.xi_points)} pts · C {g.captain}
            </div>
            {g.chip === 'freehit' && (
              <div style={{ fontSize: 11, color: 'var(--muted-2)', marginTop: 2 }}>
                XI: {(g.squad || []).filter((r) => r.in_xi)
                  .map((r) => r.name + (r.is_captain ? ' (C)' : '')).join(', ')}
              </div>
            )}
          </div>
          <span className="meta">
            £{fmt1(g.bank)}m · {g.free_after}FT
            {g.hits ? <span style={{ color: 'var(--red)' }}> · −{g.hits * 4} hit</span> : ''}
          </span>
        </div>
      ))}
    </div>
  )
}
