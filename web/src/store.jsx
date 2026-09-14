import React, {
  createContext, useCallback, useContext, useEffect, useMemo, useRef, useState,
} from 'react'
import { api } from './api'

const Ctx = createContext(null)
const ENTRY_KEY = 'fplabs.entry'

// The team id a visitor last used, remembered in this browser. Signed-in
// accounts also keep it server-side (prefs), so it follows them to a phone.
const readLocalEntry = () => {
  try {
    const v = parseInt(localStorage.getItem(ENTRY_KEY) || '', 10)
    return v > 0 ? v : null
  } catch { return null }
}
const writeLocalEntry = (v) => {
  try {
    if (v) localStorage.setItem(ENTRY_KEY, String(v))
    else localStorage.removeItem(ENTRY_KEY)
  } catch { /* private mode */ }
}

export function StoreProvider({ children }) {
  const [status, setStatus] = useState(null)
  const [auth, setAuth] = useState(null)             // /api/auth/me
  const [playersDoc, setPlayersDoc] = useState(null)
  const [fixtures, setFixtures] = useState(null)
  const [proj, setProj] = useState(null)
  const [projHistory, setProjHistory] = useState([])
  const [draftsDoc, setDraftsDoc] = useState({ drafts: [] })
  const [watch, setWatch] = useState({ players: {}, alt: {} })
  // Transfermarkt context: injuries, age, contract, value, manager.
  // Decoration — never feeds a projection, so a failure is silent.
  const [context, setContext] = useState({ players: {}, clubs: {} })
  const [entryId, setEntryIdState] = useState(readLocalEntry)
  const [entry, setEntry] = useState(null)
  const [toast, setToast] = useState(null)
  const saveTimer = useRef(null)
  // Every control a visitor touches — tab, filters, solver knobs, the draft
  // and gameweek they were looking at — lives here and autosaves (debounced)
  // to the account's prefs, so nothing is lost between visits or devices.
  const [ui, setUiState] = useState({})
  const uiTimer = useRef(null)
  const setUi = useCallback((patch) => {
    setUiState((u) => {
      const next = { ...u, ...(typeof patch === 'function' ? patch(u) : patch) }
      clearTimeout(uiTimer.current)
      uiTimer.current = setTimeout(() => api.savePrefs({ ui: next }).catch(() => {}), 700)
      return next
    })
  }, [])

  const refreshProjections = useCallback(() => {
    api.projections().then(setProj).catch(() => {})
    api.projectionHistory().then((h) => setProjHistory(h.snapshots || [])).catch(() => {})
  }, [])

  const refreshAuth = useCallback(() => api.me().then((a) => {
    setAuth(a)
    // an account's remembered team wins over this browser's, when it has one
    const remembered = a?.prefs?.entry_id
    if (remembered) { setEntryIdState(remembered); writeLocalEntry(remembered) }
    if (a?.prefs?.ui && typeof a.prefs.ui === 'object') setUiState(a.prefs.ui)
    return a
  }).catch(() => setAuth({ user: null })), [])

  const loadUserDocs = useCallback((prefUi) => {
    api.transferWatch().then(setWatch).catch(() => {})
    api.drafts().then((d) => {
      setDraftsDoc(d)
      const want = prefUi?.activeDraftId
      setActiveDraftIdState((cur) => {
        const ids = (d.drafts || []).map((x) => x.id)
        if (want && ids.includes(want)) return want
        return ids.includes(cur) ? cur : (ids[0] ?? null)
      })
    }).catch(() => {})
  }, [])
  // the draft being worked on is itself a remembered choice
  const [activeDraftId, setActiveDraftIdState] = useState(null)
  const setActiveDraftId = useCallback((v) => {
    setActiveDraftIdState(v)
    if (typeof v !== 'function') setUi({ activeDraftId: v })
  }, [setUi])

  useEffect(() => {
    api.status().then(setStatus)
      .catch(() => setToast({ kind: 'err', msg: 'Backend unreachable — the server is down or restarting.' }))
    // prefs first, so the remembered draft/tab/filters are known before the
    // documents that depend on them arrive
    refreshAuth().then((a) => loadUserDocs(a?.prefs?.ui)).catch(() => loadUserDocs())
    api.players().then(setPlayersDoc).catch(() => {})
    api.fixtures().then(setFixtures).catch(() => {})
    refreshProjections()
    api.context().then(setContext).catch(() => {})
  }, [refreshProjections, refreshAuth, loadUserDocs])

  // Projections are rebuilt by the server on its own schedule; poll cheaply
  // so a page left open picks up the overnight refresh.
  useEffect(() => {
    const t = setInterval(() => {
      api.status().then((s) => {
        setStatus((prev) => {
          if (prev && prev.proj_updated_at !== s.proj_updated_at) refreshProjections()
          return s
        })
      }).catch(() => {})
    }, 5 * 60 * 1000)
    return () => clearInterval(t)
  }, [refreshProjections])

  const setEntryId = useCallback((v) => {
    const n = v ? parseInt(v, 10) : null
    setEntryIdState(n > 0 ? n : null)
    writeLocalEntry(n > 0 ? n : null)
    api.savePrefs({ entry_id: n > 0 ? n : null }).catch(() => {})
  }, [])

  const refreshEntry = useCallback(() => {
    if (!entryId) { setEntry(null); return }
    api.entry(entryId).then(setEntry).catch(() => setEntry(null))
  }, [entryId])

  useEffect(() => { refreshEntry() }, [refreshEntry])

  const signOut = useCallback(async ({ deleteAccount = false } = {}) => {
    try { await (deleteAccount ? api.deleteAccount() : api.logout()) } catch { /* the cookie is gone either way */ }
    setEntryIdState(null); writeLocalEntry(null); setEntry(null)
    setDraftsDoc({ drafts: [] }); setActiveDraftIdState(null); setUiState({})
    setWatch({ players: {}, alt: {} })
    const a = await refreshAuth()
    loadUserDocs(a?.prefs?.ui)
  }, [refreshAuth, loadUserDocs])

  // debounced autosave of drafts
  const setDrafts = useCallback((updater, { save = true } = {}) => {
    setDraftsDoc((doc) => {
      const drafts = typeof updater === 'function' ? updater(doc.drafts) : updater
      const next = { ...doc, drafts }
      if (save) {
        clearTimeout(saveTimer.current)
        saveTimer.current = setTimeout(() => api.saveDrafts(next).catch(() => {}), 800)
      }
      return next
    })
  }, [])

  /* Mark a player as leaving (optionally naming the destination) and the
     engine reprojects him onto that club's fixtures. FPL only reclassifies a
     player once the transfer completes, so until then he is projected on a run
     he will never play — the fact has to come from you, off the news. */
  const setTransferWatch = useCallback(async (pid, entryDoc) => {
    const players = { ...(watch.players || {}) }
    if (entryDoc) players[String(pid)] = entryDoc
    else delete players[String(pid)]
    setWatch((w) => ({ ...w, players }))          // optimistic
    try {
      setWatch(await api.saveTransferWatch({ players }))
    } catch {
      api.transferWatch().then(setWatch).catch(() => {})
    }
  }, [watch.players])

  const byId = useMemo(() => {
    const m = new Map()
    for (const p of playersDoc?.players || []) m.set(p.id, p)
    return m
  }, [playersDoc])

  const teams = playersDoc?.teams || {}

  // The app used to render tabs against nulls while the first fetches were in
  // flight, so the first second looked like an empty product. `booted` is what
  // the boot screen waits on: status and players are the two payloads every
  // tab needs before anything it draws means anything.
  const booted = !!(status && playersDoc && auth)
  // The first gameweek a manager can still change. While a gameweek is in
  // progress (deadline passed, matches unfinished) the model's next_gw is
  // that gameweek, but nothing can be changed for it any more: every tab
  // plans from the next open deadline instead.
  const editableGw = status?.editable_gw ?? status?.next_gw ?? null

  const user = auth?.user || null
  const value = {
    booted,
    status, setStatus, editableGw,
    ui, setUi,
    auth, user, isAdmin: !!user?.is_admin, refreshAuth, signOut,
    plan: auth?.plan || 'pro', entitlements: auth?.entitlements || null,
    players: playersDoc?.players || [], byId, teams,
    events: playersDoc?.events || [],
    fixtures, proj, projHistory, refreshProjections,
    drafts: draftsDoc.drafts || [], setDrafts,
    activeDraftId, setActiveDraftId,
    entryId, setEntryId, entry, refreshEntry,
    watch, setTransferWatch,
    context,
    toast, setToast,
  }
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export const useStore = () => useContext(Ctx)

/* A piece of UI state that is remembered on the account: reads like
   useState, writes through the debounced prefs autosave. */
export function usePersisted(key, initial) {
  const { ui, setUi } = useStore()
  const value = ui[key] === undefined ? initial : ui[key]
  const set = useCallback((v) => setUi((u) => {
    const cur = u[key] === undefined ? initial : u[key]
    return { [key]: typeof v === 'function' ? v(cur) : v }
  }), [key, setUi, initial])
  return [value, set]
}

// convenience: what does team X play in gw G? -> [{opp, home, fdr}]
export function useFixtureLookup() {
  const { fixtures, teams } = useStore()
  return useCallback((teamId, gw) => {
    const cell = fixtures?.grid?.[String(teamId)]?.[String(gw)] || []
    return cell.map((f) => ({
      ...f,
      oppShort: teams[String(f.opp)]?.short || '?',
    }))
  }, [fixtures, teams])
}
