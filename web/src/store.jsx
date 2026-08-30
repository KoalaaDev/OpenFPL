import React, {
  createContext, useCallback, useContext, useEffect, useMemo, useRef, useState,
} from 'react'
import { api } from './api'

const Ctx = createContext(null)

export function StoreProvider({ children }) {
  const [status, setStatus] = useState(null)
  const [playersDoc, setPlayersDoc] = useState(null)
  const [fixtures, setFixtures] = useState(null)
  const [proj, setProj] = useState(null)
  const [projHistory, setProjHistory] = useState([])
  const [draftsDoc, setDraftsDoc] = useState({ drafts: [] })
  const [activeDraftId, setActiveDraftId] = useState(null)
  const [watch, setWatch] = useState({ players: {}, alt: {} })
  const [entryId, setEntryId] = useState(null)
  const [entry, setEntry] = useState(null)
  const [toast, setToast] = useState(null)
  const saveTimer = useRef(null)

  const refreshProjections = useCallback(() => {
    api.projections().then(setProj).catch(() => {})
    api.projectionHistory().then((h) => setProjHistory(h.snapshots || [])).catch(() => {})
  }, [])

  useEffect(() => {
    api.status().then((s) => {
      setStatus(s)
      setEntryId((e) => e ?? s.default_entry)
    }).catch(() => setToast({ kind: 'err', msg: 'Backend unreachable — is `python -m app` running?' }))
    api.players().then(setPlayersDoc).catch(() => {})
    api.fixtures().then(setFixtures).catch(() => {})
    refreshProjections()
    api.transferWatch().then(setWatch).catch(() => {})
    api.drafts().then((d) => {
      setDraftsDoc(d)
      if (d.drafts?.length) setActiveDraftId(d.drafts[0].id)
    }).catch(() => {})
  }, [refreshProjections])

  const refreshEntry = useCallback(() => {
    if (!entryId) return
    api.entry(entryId).then(setEntry).catch(() => setEntry(null))
  }, [entryId])

  useEffect(() => { refreshEntry() }, [refreshEntry])

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
  const setTransferWatch = useCallback(async (pid, entry) => {
    const players = { ...(watch.players || {}) }
    if (entry) players[String(pid)] = entry
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
  const booted = !!(status && playersDoc)

  const value = {
    booted,
    status, setStatus,
    players: playersDoc?.players || [], byId, teams,
    events: playersDoc?.events || [],
    fixtures, proj, projHistory, refreshProjections,
    drafts: draftsDoc.drafts || [], setDrafts,
    activeDraftId, setActiveDraftId,
    entryId, setEntryId, entry, refreshEntry,
    watch, setTransferWatch,
    toast, setToast,
  }
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export const useStore = () => useContext(Ctx)

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
