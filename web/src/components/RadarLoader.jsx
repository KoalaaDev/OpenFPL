import React, { useEffect, useRef, useState } from 'react'

/* An "analysing" radar, for waits that are long and shapeless.

   The radar is this app's signature chart — Team DNA, the player profile — so
   using it as the loading state makes the wait look like the product rather
   than like a generic spinner. It belongs on the boot sequence, a solve, a
   projections build, a data pull: waits measured in seconds where there is no
   layout to preview. Tables keep their skeletons, which show the SHAPE of what
   is coming and stop the page jumping when it lands; a radar there would just
   flash and shift things.

   The shape is driven by summed sine waves at incommensurable frequencies, so
   it never visibly repeats and reads as live measurement instead of a loop.
   Under `prefers-reduced-motion` it settles into a single static polygon —
   still on-brand, no animation. */

const AXES = 6
const RINGS = [0.25, 0.5, 0.75, 1]

/* Timing. The first version morphed every vertex continuously with summed
   sines, and it read as a blob drifting inside the grid rather than as
   measurement — because nothing on screen CAUSED the movement.

   Instruments read as sample-and-hold: a value jumps when it is read, then
   sits still until it is read again. So each axis is now driven BY the sweep.
   A vertex re-samples at the instant the sweep line crosses it, eases to the
   new reading in a fraction of a second, and then holds until the next pass.

   One full revolution is 4.8s and the axes are 60 degrees apart, so exactly
   one vertex ticks every 0.8s — the cadence asked for, but arrived at from the
   geometry, so the sweep and the readings can never drift out of step. */
const REV_SECONDS = 4.8
const SWEEP_DEG_PER_S = 360 / REV_SECONDS      // 75 deg/s -> a tick every 0.8s
const SETTLE = 0.34 / REV_SECONDS              // ~340ms of travel, then hold
const V_MIN = 0.34
const V_MAX = 0.95

// deterministic per (axis, revolution): the same vertex never re-reads the
// same value twice in a row, and the sequence does not repeat visibly
const reading = (i, rev) => {
  const h = Math.sin(i * 127.1 + rev * 311.7) * 43758.5453
  return V_MIN + (h - Math.floor(h)) * (V_MAX - V_MIN)
}
const easeOut = (x) => 1 - Math.pow(1 - x, 3)

export default function RadarLoader({
  size = 132, label = 'Analysing', sub = null, inline = false,
}) {
  const [t, setT] = useState(0)
  const raf = useRef(0)
  const uid = useRef(Math.random().toString(36).slice(2, 8)).current
  const reduced = typeof window !== 'undefined'
    && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

  useEffect(() => {
    if (reduced) return undefined
    let start = null
    const step = (now) => {
      if (start == null) start = now
      setT((now - start) / 1000)
      raf.current = requestAnimationFrame(step)
    }
    raf.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf.current)
  }, [reduced])

  const c = size / 2
  const R = c - 10
  const angle = (i) => (Math.PI * 2 * i) / AXES - Math.PI / 2
  const pt = (i, v) => [c + Math.cos(angle(i)) * R * v, c + Math.sin(angle(i)) * R * v]

  const sweepDeg = t * SWEEP_DEG_PER_S

  /* How far through its own cycle each axis is, measured from the moment the
     sweep last crossed it. Axis i sits at (60i - 90) degrees from the +x axis,
     which is where the sweep line starts. */
  const cycle = (i) => {
    const raw = (sweepDeg - (60 * i - 90)) / 360
    const rev = Math.floor(raw)
    return { rev, frac: raw - rev }
  }

  const value = (i) => {
    if (reduced) return 0.62
    const { rev, frac } = cycle(i)
    const from = reading(i, rev - 1)
    const to = reading(i, rev)
    // travel briefly after the sweep passes, then hold at the new reading
    return from + (to - from) * easeOut(Math.min(1, frac / SETTLE))
  }

  const poly = Array.from({ length: AXES }, (_, i) => pt(i, value(i)))
  const ring = (f) => Array.from({ length: AXES }, (_, i) => pt(i, f).join(',')).join(' ')
  const sweep = sweepDeg % 360

  return (
    <div className={`radar-loader ${inline ? 'inline' : ''}`}
      role="status" aria-live="polite" aria-label={label}>
      <svg viewBox={`0 0 ${size} ${size}`} width={size} height={size} aria-hidden="true">
        <defs>
          <linearGradient id={`rl-sweep-${uid}`} x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="var(--accent)" stopOpacity="0" />
            <stop offset="100%" stopColor="var(--accent)" stopOpacity="0.5" />
          </linearGradient>
          {/* the rings are hexagons but the wedge is a circular sector, so it
              bled past the grid on the flats; clip it to the outer hexagon */}
          <clipPath id={`rl-clip-${uid}`}>
            <polygon points={ring(1)} />
          </clipPath>
        </defs>

        {RINGS.map((f) => (
          <polygon key={f} points={ring(f)} fill="none"
            stroke="var(--line)" strokeWidth={f === 1 ? 1.3 : 1}
            opacity={f === 1 ? 1 : 0.55} />
        ))}
        {Array.from({ length: AXES }, (_, i) => {
          const [x, y] = pt(i, 1)
          return <line key={i} x1={c} y1={c} x2={x} y2={y}
            stroke="var(--line)" strokeWidth="1" opacity="0.4" />
        })}

        {/* The sweep, as a computed wedge rather than an SVG arc: the arc's
            large-arc/sweep flags took the long way round and the tail bulged
            outside the chart. Sampling the edge is unambiguous. */}
        {!reduced && (
          <g clipPath={`url(#rl-clip-${uid})`}
            style={{ transform: `rotate(${sweep}deg)`, transformOrigin: 'center' }}>
            <polygon
              points={[`${c},${c}`].concat(
                Array.from({ length: 9 }, (_, k) => {
                  const a = (-Math.PI / 3) * (k / 8)      // 0 -> -60 degrees
                  return `${c + Math.cos(a) * R},${c + Math.sin(a) * R}`
                })).join(' ')}
              fill={`url(#rl-sweep-${uid})`} />
            <line x1={c} y1={c} x2={c + R} y2={c}
              stroke="var(--accent)" strokeWidth="1.4" opacity="0.9" />
          </g>
        )}

        <polygon points={poly.map((p) => p.join(',')).join(' ')}
          fill="var(--accent)" fillOpacity="0.16"
          stroke="var(--accent)" strokeWidth="2" strokeLinejoin="round" />
        {poly.map(([x, y], i) => {
          // a brief flare exactly as this vertex is re-read, so the eye is
          // told which reading just landed rather than watching everything move
          const f = reduced ? 1 : cycle(i).frac
          const fresh = Math.max(0, 1 - f / (SETTLE * 2.4))
          return (
            <g key={i}>
              {fresh > 0.01 && (
                <circle cx={x} cy={y} r={2.6 + 5 * fresh}
                  fill="var(--accent)" opacity={0.28 * fresh} />
              )}
              <circle cx={x} cy={y} r={2.6 + 1.1 * fresh} fill="var(--accent)" />
            </g>
          )
        })}
      </svg>
      {label ? <div className="rl-label">{label}</div> : null}
      {sub ? <div className="rl-sub">{sub}</div> : null}
    </div>
  )
}

/* Full-screen version for the one wait the user cannot do anything during:
   the app fetching everything it needs before any tab can render. */
export function BootLoader({ sub }) {
  return (
    <div className="boot-loader">
      <RadarLoader size={168} label="OpenFPL" sub={sub || 'Loading squad, fixtures and projections…'} />
    </div>
  )
}
