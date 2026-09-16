import React from 'react'

/* A real pitch, drawn rather than suggested.

   The old planner put the squad on a rounded purple rectangle with two
   decorative lines. It read as a UI panel, which is exactly why nothing on
   it looked clickable: there was no scene, so the cards floated.

   This is the markings of an actual football pitch, seen from above, with
   the goal the keeper defends at the TOP (the rows run GK -> DEF -> MID ->
   FWD down the page, which is the direction of attack). It is one SVG with
   `preserveAspectRatio="none"` so it stretches to whatever the panel gives
   it — the circles are therefore ellipses in the markup, which is what keeps
   them looking circular relative to the pitch at any panel shape.

   Purely decorative: `aria-hidden`, no pointer events, so every click still
   lands on a player card. */
export default function PitchLines() {
  return (
    <svg className="pitch-lines" viewBox="0 0 100 150" preserveAspectRatio="none"
      aria-hidden="true" focusable="false">
      {/* touchlines */}
      <rect x="3" y="3" width="94" height="144" />
      {/* halfway line + centre circle */}
      <line x1="3" y1="75" x2="97" y2="75" />
      <ellipse cx="50" cy="75" rx="15" ry="13" />
      <circle className="spot" cx="50" cy="75" r="0.9" />

      {/* defended goal (top) */}
      <rect x="27.5" y="3" width="45" height="22" />
      <rect x="39.5" y="3" width="21" height="8.5" />
      <circle className="spot" cx="50" cy="17" r="0.9" />
      <path d="M 38.5 25 A 15 13 0 0 0 61.5 25" />
      <rect className="goal" x="42" y="0.6" width="16" height="2.6" />

      {/* attacking goal (bottom) */}
      <rect x="27.5" y="125" width="45" height="22" />
      <rect x="39.5" y="138.5" width="21" height="8.5" />
      <circle className="spot" cx="50" cy="133" r="0.9" />
      <path d="M 38.5 125 A 15 13 0 0 1 61.5 125" />
      <rect className="goal" x="42" y="146.8" width="16" height="2.6" />

      {/* corner arcs */}
      <path d="M 3 7 A 4 3.5 0 0 0 7 3" />
      <path d="M 93 3 A 4 3.5 0 0 0 97 7" />
      <path d="M 3 143 A 4 3.5 0 0 1 7 147" />
      <path d="M 93 147 A 4 3.5 0 0 1 97 143" />
    </svg>
  )
}
