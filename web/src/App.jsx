import React, { useEffect, useState } from 'react'
import { useStore, usePersisted } from './store'
import Planner from './tabs/Planner'
import Projections from './tabs/Projections'
import Fixtures from './tabs/Fixtures'
import MiniLeague from './tabs/MiniLeague'
import Solver from './tabs/Solver'
import Prices from './tabs/Prices'
import Deadline from './tabs/Deadline'
import Live from './tabs/Live'
import Model from './tabs/Model'
import ErrorBoundary from './components/ErrorBoundary'
import MyTeamModal from './components/MyTeamModal'
import AccountMenu from './components/AccountMenu'
import { BrandMark, Wordmark } from './components/Brand'
import { BootLoader } from './components/RadarLoader'
import { API_VERSION, api, pollJob } from './api'

const TABS = [
  ['Planner', '⚽'], ['Projections', '📈'], ['Fixtures', '🗓'],
  ['Prices', '💷'], ['Mini League', '🏆'], ['Solver', '🧪'],
]

export default function App() {
  const [tab, setTab] = usePersisted('tab', 'Planner')
  // Tabs used to be swapped with `tab === 'X' && <X/>`, which UNMOUNTS the
  // old one and throws away everything it held - most painfully a solve that
  // took a minute to run. A tab is now mounted the first time it is opened
  // and then merely hidden, so results, filters and scroll position survive
  // switching. Unvisited tabs are still never mounted, so nothing fetches
  // league or projection data until it is actually asked for.
  const [visited, setVisited] = useState({ Planner: true, [tab]: true })
  const openTab = (t) => { setVisited((v) => (v[t] ? v : { ...v, [t]: true })); setTab(t) }
  const { booted, status, setStatus, entryId, setEntryId, entry, toast, setToast,
          refreshProjections, isAdmin } = useStore()
  /* The Live desk is not a permanent tab: it appears 24 h before a deadline
     and stays six hours past it, then goes away (`app/live.py` owns that
     window, `status.live` reports it). A tab that is only there when there is
     something to do reads as a signal; one that is always there is furniture. */
  const live = status?.live
  // admins keep it on screen outside the window so the desk can be checked
  // (and fixed) on a Tuesday rather than an hour before a deadline
  const liveOn = !!live && (live.phase !== 'idle' || isAdmin)
  const tabs = [
    ...(liveOn ? [['Live', '🔴']] : []),
    ...TABS,
    ...(isAdmin ? [['Deadline', '🛰'], ['Model', '📉']] : []),
  ]
  // a window that opens while you are sitting on the page still gets mounted
  useEffect(() => {
    if (liveOn) setVisited((v) => (v.Live ? v : { ...v, Live: true }))
  }, [liveOn])
  const [refreshing, setRefreshing] = useState(false)
  const [teamModal, setTeamModal] = useState(false)

  // the Google round-trip lands back here with ?login=…
  useEffect(() => {
    const q = new URLSearchParams(window.location.search)
    const r = q.get('login')
    if (!r) return
    if (r === 'failed') setToast({ kind: 'err', msg: 'Google sign-in failed — please try again.' })
    if (r === 'cancelled') setToast({ kind: 'info', msg: 'Sign-in cancelled.' })
    window.history.replaceState({}, '', window.location.pathname)
  }, [setToast])

  useEffect(() => {
    if (tab === 'Live' && !liveOn) setTab('Planner')
  }, [tab, liveOn])                  // eslint-disable-line react-hooks/exhaustive-deps

  const doRefresh = async () => {
    if (refreshing) return
    setRefreshing(true)
    setToast({ kind: 'info', msg: 'Refreshing data and re-running the model…' })
    try {
      const { job_id } = await api.refresh()
      await pollJob(job_id, (j) => {
        const last = j.progress[j.progress.length - 1]
        if (last) setToast({ kind: 'info', msg: last.msg })
      })
      setToast({ kind: 'ok', msg: 'Data refreshed and projections rebuilt.' })
      api.status().then(setStatus)
      refreshProjections()
    } catch (e) {
      setToast({ kind: 'err', msg: `Refresh failed: ${e.message}` })
    } finally {
      setRefreshing(false)
    }
  }

  const stale = status && status.api_version !== API_VERSION
  const autoBusy = (status?.jobs_running || []).includes('refresh')

  if (!booted) {
    return <BootLoader sub={status ? 'Loading players and fixtures…' : 'Contacting the lab…'} />
  }

  return (
    <>
      {stale && (
        <div className="stale-banner">
          ⚠ The server is running an older build than this page (API {status.api_version || 'unknown'}
          vs {API_VERSION}). Reload in a minute — if it persists the server needs a restart.
        </div>
      )}
      <header className="topnav">
        <div className="brand">
          <BrandMark />
          <Wordmark />
        </div>
        <nav className="tabs">
          {tabs.map(([t]) => (
            <button key={t}
              className={`tab ${tab === t ? 'active' : ''} ${t === 'Live' ? `live-tab ${live.phase}` : ''}`}
              title={t === 'Live' && live.phase === 'idle' ? 'Preview — visible to admins only until 24 h before the deadline' : undefined}
              onClick={() => openTab(t)}>
              {t === 'Live' && <i className="live-dot" aria-hidden="true" />}
              {/* the phase is carried by the dot's colour and the tooltip; a
                  longer label pushed the first tab off a crowded strip */}
              {t === 'Live' && live.phase === 'closed' ? 'Live · over' : t}
            </button>
          ))}
        </nav>
        <div className="right">
          <button className={`pill-btn squad-btn ${entry?.squad ? '' : 'accent'}`}
            title="import or enter your current 15" onClick={() => setTeamModal(true)}>
            {entry?.squad ? '✓ squad' : '⚠ set my team'}
          </button>
          <EntryBox entryId={entryId} setEntryId={setEntryId} entry={entry} />
          {isAdmin && (
            <button className="pill-btn admin-only" onClick={doRefresh} disabled={refreshing || autoBusy}
              title="pull data + rebuild projections now (admin)">
              {refreshing || autoBusy ? <span className="spinner" /> : '⟳'} Refresh
            </button>
          )}
          {status && (
            <span className="chip dim num gw-chip" title={freshnessTitle(status)}>
              GW{status.next_gw}{autoBusy ? ' ·' : ''}
            </span>
          )}
          <AccountMenu />
        </div>
      </header>

      <main className={`page ${tab === 'Mini League' || tab === 'Live' ? 'wide' : ''}`}>
        {/* the prompt belongs where a squad is needed, not on a read-only or
            operator tab */}
        {!entryId && !['Fixtures', 'Prices', 'Live', 'Deadline', 'Model'].includes(tab) && (
          <Welcome setEntryId={setEntryId} openTeam={() => setTeamModal(true)} />
        )}
        {visited.Live && liveOn && <Pane name="Live" on={tab === 'Live'}><Live /></Pane>}
        {visited.Planner && <Pane name="Planner" on={tab === 'Planner'}><Planner /></Pane>}
        {visited.Projections && <Pane name="Projections" on={tab === 'Projections'}><Projections /></Pane>}
        {visited.Fixtures && <Pane name="Fixtures" on={tab === 'Fixtures'}><Fixtures /></Pane>}
        {visited.Prices && <Pane name="Prices" on={tab === 'Prices'}><Prices /></Pane>}
        {visited['Mini League'] && <Pane name="Mini League" on={tab === 'Mini League'}><MiniLeague /></Pane>}
        {visited.Solver && (
          <Pane name="Solver" on={tab === 'Solver'}>
            <Solver goPlanner={() => openTab('Planner')} />
          </Pane>
        )}
        {visited.Deadline && isAdmin && <Pane name="Deadline" on={tab === 'Deadline'}><Deadline /></Pane>}
        {visited.Model && isAdmin && <Pane name="Model" on={tab === 'Model'}><Model /></Pane>}
        <footer className="site-foot">
          <span><b>FPLabs</b> by KoalaaDev · models refresh automatically{status?.proj_updated_at ? ` · last run ${ago(status.proj_updated_at)}` : ''}</span>
          <span className="muted">Built on the open OpenFPL research models. Not affiliated with the Premier League.
            {' '}<a href="/privacy">Privacy</a> · <a href="/terms">Terms</a></span>
        </footer>
      </main>

      <BottomNav tabs={tabs} tab={tab} openTab={openTab} />

      {teamModal && <MyTeamModal close={() => setTeamModal(false)} />}

      {toast && (
        <div className="toast">
          {toast.kind === 'ok' && <span className="ok">✓</span>}
          {toast.kind === 'err' && <span style={{ color: 'var(--red)', fontWeight: 800 }}>✕</span>}
          {toast.kind === 'info' && <span className="spinner" />}
          <span>{toast.msg}</span>
          <button className="close" onClick={() => setToast(null)}>×</button>
        </div>
      )}
    </>
  )
}

function freshnessTitle(status) {
  const parts = [`next gameweek: ${status.next_gw}`]
  if (status.proj_updated_at) parts.push(`projections built ${ago(status.proj_updated_at)}`)
  const ar = status.auto_refresh
  if (ar?.enabled && ar.next_run) parts.push(`next auto-refresh ${new Date(ar.next_run * 1000).toLocaleString()}`)
  return parts.join(' · ')
}

function ago(ts) {
  const s = Math.max(0, Date.now() / 1000 - ts)
  if (s < 90) return 'just now'
  if (s < 3600) return `${Math.round(s / 60)} min ago`
  if (s < 86400) return `${Math.round(s / 3600)} h ago`
  return `${Math.round(s / 86400)} d ago`
}

/* A phone bottom bar holds five things legibly. With the Live desk open and
   an admin signed in there are nine, and nine `flex: 1` items at 400px is
   44px each — labels truncate to "Proje…" and the bar stops being navigation.
   Four primary destinations plus More, which opens a sheet with the rest;
   whichever tab you are on is always one of the five, so the bar never shows
   you standing somewhere it cannot indicate. */
const BN_PRIMARY = ['Live', 'Planner', 'Projections', 'Solver']

function BottomNav({ tabs, tab, openTab }) {
  const [more, setMore] = useState(false)
  const byName = Object.fromEntries(tabs.map((t) => [t[0], t]))
  let primary = BN_PRIMARY.filter((n) => byName[n]).map((n) => byName[n]).slice(0, 4)
  let rest = tabs.filter((t) => !primary.includes(t))
  // never hide where you actually are
  if (rest.some((t) => t[0] === tab)) {
    const here = rest.find((t) => t[0] === tab)
    primary = [...primary.slice(0, 3), here]
    rest = tabs.filter((t) => !primary.includes(t))
  }
  const short = (t) => (t === 'Mini League' ? 'League' : t === 'Projections' ? 'Points' : t)
  return (
    <>
      <nav className="bottomnav" aria-label="sections">
        {primary.map(([t, icon]) => (
          <button key={t} className={`bn-tab ${tab === t ? 'active' : ''}`}
            aria-current={tab === t ? 'page' : undefined}
            onClick={() => openTab(t)}>
            <span className="bn-icon" aria-hidden="true">{icon}</span>
            <span className="bn-label">{short(t)}</span>
          </button>
        ))}
        {rest.length > 0 && (
          <button className={`bn-tab ${more ? 'active' : ''}`} aria-expanded={more}
            onClick={() => setMore(true)}>
            <span className="bn-icon" aria-hidden="true">⋯</span>
            <span className="bn-label">More</span>
          </button>
        )}
      </nav>

      {more && (
        <div className="bn-sheet-backdrop" onClick={() => setMore(false)}>
          <div className="bn-sheet" onClick={(e) => e.stopPropagation()}
            role="dialog" aria-label="More sections">
            <div className="bn-grip" aria-hidden="true" />
            {rest.map(([t, icon]) => (
              <button key={t} className={`bn-sheet-item ${tab === t ? 'active' : ''}`}
                onClick={() => { openTab(t); setMore(false) }}>
                <span aria-hidden="true">{icon}</span>{t}
              </button>
            ))}
            <button className="bn-sheet-close" onClick={() => setMore(false)}>Close</button>
          </div>
        </div>
      )}
    </>
  )
}

// keeps a tab alive but out of the way; `hidden` would also stop layout but
// display:none is what lets a re-shown tab keep its scroll position.
// The boundary is per pane so a render error is contained to the tab that
// caused it instead of unmounting the whole app (see ErrorBoundary).
function Pane({ on, name, children }) {
  return (
    <div style={{ display: on ? 'contents' : 'none' }}>
      <ErrorBoundary where={name}>{children}</ErrorBoundary>
    </div>
  )
}

/* No team id yet: the first thing a new visitor sees. There is no default
   manager on a public planner — the projections are the same for everyone,
   the squad is yours. */
function Welcome({ setEntryId, openTeam }) {
  const [val, setVal] = useState('')
  return (
    <div className="welcome panel">
      <div className="welcome-copy">
        <h2>Your team, the lab's numbers.</h2>
        <p>Enter your FPL team ID to load your squad, plan transfers and run the solver.
          You can find it in the URL of your Points page on fantasy.premierleague.com:
          <span className="num"> …/entry/<b>1234567</b>/event/…</span></p>
      </div>
      <form className="welcome-form" onSubmit={(e) => {
        e.preventDefault()
        const n = parseInt(val, 10)
        if (n > 0) setEntryId(n)
      }}>
        <input inputMode="numeric" pattern="[0-9]*" className="num" placeholder="team id" value={val}
          onChange={(e) => setVal(e.target.value.replace(/\D/g, ''))} />
        <button className="pill-btn accent" type="submit" disabled={!val}>Load my team</button>
        <button className="pill-btn" type="button" onClick={openTeam}>No id — enter squad</button>
      </form>
    </div>
  )
}

function EntryBox({ entryId, setEntryId, entry }) {
  const [editing, setEditing] = useState(false)
  const [val, setVal] = useState('')
  if (editing) {
    return (
      <form onSubmit={(e) => {
        e.preventDefault()
        const n = parseInt(val, 10)
        if (n > 0) setEntryId(n)
        setEditing(false)
      }}>
        <input autoFocus className="num entry-input" value={val} inputMode="numeric"
          onChange={(e) => setVal(e.target.value.replace(/\D/g, ''))}
          onBlur={() => setEditing(false)}
          placeholder="team id" />
      </form>
    )
  }
  return (
    <button className="pill-btn entry-btn" title="change FPL team id"
      onClick={() => { setVal(String(entryId || '')); setEditing(true) }}>
      <span className="entry-ico">👤</span>
      <span className="entry-lbl">{entry?.team_name || (entryId ? `#${entryId}` : 'team id')}</span>
    </button>
  )
}
