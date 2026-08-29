import { epOf } from './util'

/* Team DNA — one definition, used by both Mini League and Planner.

   These six axes describe the *shape* of a squad rather than its quality, so
   two managers on the same points can still look completely different. The
   definitions live here rather than inside either tab because the whole value
   of the chart is that "my Attack" means the same thing everywhere it appears;
   two copies would drift the first time one was tweaked.

   Mini League scales these against the league (who is the most template of
   THIS group). The Planner cannot — comparing exactly two squads by min-max
   would pin one axis at each end regardless of how small the real difference
   was, and make a one-player transfer look like a personality transplant. So
   the Planner scales against the fixed domains below, which are absolute and
   let a small change look small. */

export const DNA_AXES = [
  { key: 'firepower', label: 'Firepower', fmt: (v) => v.toFixed(1),
    domain: [0, 90], hint: 'projected starting XI points this gameweek' },
  { key: 'template', label: 'Template', fmt: (v) => `${v.toFixed(0)}%`,
    domain: [0, 55], hint: 'average global ownership of the 15' },
  { key: 'premium', label: 'Premiums', fmt: (v) => `${v.toFixed(0)}%`,
    domain: [15, 45], hint: 'share of squad value in the 3 costliest players' },
  { key: 'attack', label: 'Attack', fmt: (v) => `${v.toFixed(0)}%`,
    domain: [30, 85], hint: 'share of expected points from MID and FWD' },
  { key: 'bench', label: 'Bench', fmt: (v) => `${v.toFixed(0)}%`,
    domain: [0, 30], hint: 'share of expected points sitting on the bench' },
  { key: 'spread', label: 'Spread', fmt: (v) => v.toFixed(1),
    domain: [5, 12], hint: 'distinct clubs in the squad — how concentrated it is' },
]

/* players: [{ id, benched }]  — benched is optional and only affects `bench` */
export function dnaOf(players, { byId, proj, gw }) {
  const ps = players.map((r) => ({ ...r, s: byId.get(r.id) })).filter((r) => r.s)
  if (!ps.length) return null
  const prices = ps.map((r) => r.s.price).sort((a, b) => b - a)
  const totVal = prices.reduce((a, b) => a + b, 0) || 1
  const eps = ps.map((r) => ({ ...r, ep: gw ? epOf(proj, r.id, gw) : 0 }))
  const epSum = eps.reduce((a, r) => a + r.ep, 0) || 1
  const clubs = new Set(ps.map((r) => r.s.team_id))
  return {
    firepower: eps.filter((r) => !r.benched).reduce((a, r) => a + r.ep, 0),
    template: ps.reduce((a, r) => a + (r.s.own || 0), 0) / ps.length,
    premium: ((prices[0] || 0) + (prices[1] || 0) + (prices[2] || 0)) / totVal * 100,
    attack: eps.filter((r) => ['MID', 'FWD'].includes(r.s.position))
      .reduce((a, r) => a + r.ep, 0) / epSum * 100,
    bench: eps.filter((r) => r.benched).reduce((a, r) => a + r.ep, 0) / epSum * 100,
    spread: clubs.size,
  }
}

/* absolute 0-100 for the radar, clamped so an outlier cannot blow the shape */
export const dnaScaled = (m) =>
  DNA_AXES.map((a) => {
    const [lo, hi] = a.domain
    const v = ((m[a.key] - lo) / (hi - lo)) * 100
    return Math.max(3, Math.min(100, v))
  })

export const dnaRaw = (m) => DNA_AXES.map((a) => m[a.key])
