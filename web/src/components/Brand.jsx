import React from 'react'

/* The FPLabs mark: a flask over a pitch line — the lab, in the game's own
   colours. One SVG, drawn once, so the header, the boot screen and the
   favicon all agree. */
export function BrandMark({ size = 30 }) {
  return (
    <svg className="brand-mark" width={size} height={size} viewBox="0 0 64 64" aria-hidden="true">
      <defs>
        <linearGradient id="fpl-g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#8b7bff" />
          <stop offset="1" stopColor="#3ddc97" />
        </linearGradient>
      </defs>
      <rect x="2" y="2" width="60" height="60" rx="16" fill="url(#fpl-g)" />
      <path d="M26 14h12v14l10 18c1.6 2.9-.4 6-3.5 6h-25c-3.1 0-5.1-3.1-3.5-6l10-18V14z"
        fill="#0b0e1a" opacity="0.92" />
      <path d="M24 38h16l5 9H19z" fill="url(#fpl-g)" />
      <circle cx="30" cy="42" r="1.8" fill="#0b0e1a" />
      <circle cx="35" cy="45" r="1.3" fill="#0b0e1a" />
      <rect x="23" y="12" width="18" height="4" rx="2" fill="#0b0e1a" opacity="0.92" />
    </svg>
  )
}

export function Wordmark({ compact = false }) {
  return (
    <span className="wordmark">
      <span className="wm-name">FPLabs</span>
      {!compact && <span className="wm-by">by KoalaaDev</span>}
    </span>
  )
}
