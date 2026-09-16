// must match app/services.py API_VERSION — mismatch means the running
// `python -m app` predates this build and needs a restart
export const API_VERSION = '2026-09-16.1'

const j = async (r) => {
  if (!r.ok) {
    let msg = `${r.status}`
    try { msg = (await r.json()).detail || msg } catch { /* ignore */ }
    if (r.status === 429) msg = 'Too many requests — give it a moment.'
    if (r.status === 402) msg = `${msg} (Pro)`
    throw new Error(msg)
  }
  return r.json()
}

// Every state-changing call carries the two things a cross-site form cannot:
// a JSON content type and a custom header. The backend refuses mutations
// without them, which (with SameSite cookies) is the whole CSRF defence.
const send = (url, method, body) =>
  fetch(url, {
    method,
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'fetch' },
    body: JSON.stringify(body ?? {}),
  }).then(j)

const get = (url) => fetch(url, { credentials: 'same-origin' }).then(j)

export const api = {
  // account
  me: () => get('/api/auth/me'),
  logout: () => send('/api/auth/logout', 'POST'),
  deleteAccount: () => send('/api/auth/me', 'DELETE'),
  loginUrl: (next = '/') => `/api/auth/google/start?next=${encodeURIComponent(next)}`,
  prefs: () => get('/api/prefs'),
  savePrefs: (doc) => send('/api/prefs', 'PUT', doc),

  status: () => get('/api/status'),
  players: () => get('/api/players'),
  fixtures: () => get('/api/fixtures'),
  prices: (limit = 30) => get(`/api/prices?limit=${limit}`),
  context: () => get('/api/context'),
  projections: () => get('/api/projections'),
  projectionHistory: () => get('/api/projections/history'),
  buildProjections: (gws, force = false) => send('/api/projections/build', 'POST', { gws, force }),
  pull: () => send('/api/pull', 'POST', {}),
  refresh: () => send('/api/refresh', 'POST', {}),
  deadline: (force = false) => get(`/api/admin/deadline${force ? '?force=1' : ''}`),
  live: () => get('/api/live'),
  entry: (id) => get(`/api/entry/${id}`),
  league: (id, { gw, limit } = {}) => {
    const q = new URLSearchParams()
    if (gw) q.set('gw', gw)
    if (limit) q.set('limit', limit)
    const qs = q.toString()
    return get(`/api/league/${id}${qs ? `?${qs}` : ''}`)
  },
  solve: (params) => send('/api/solve', 'POST', params),
  job: (id) => get(`/api/jobs/${id}`),
  myTeam: () => get('/api/myteam'),
  saveMyTeam: (doc) => send('/api/myteam', 'PUT', doc),
  clearMyTeam: () => send('/api/myteam', 'DELETE'),
  pasteMyTeam: (entry, payload) => send('/api/myteam/paste', 'POST', { entry, payload }),
  importMyTeam: (entry, cookie) => send('/api/myteam/import', 'POST', { entry, cookie }),
  transferWatch: () => get('/api/transferwatch'),
  saveTransferWatch: (doc) => send('/api/transferwatch', 'PUT', doc),
  drafts: () => get('/api/drafts'),
  saveDrafts: (doc) => send('/api/drafts', 'PUT', doc),
}

export async function pollJob(id, onProgress, intervalMs = 1200) {
  for (;;) {
    const job = await api.job(id)
    if (onProgress) onProgress(job)
    if (job.status === 'done') return job.result
    if (job.status === 'error') throw new Error(job.error || 'job failed')
    await new Promise((res) => setTimeout(res, intervalMs))
  }
}
