import React from 'react'
import { usePersisted } from '../store'

/* A foldable panel.

   The planner's right column used to be four panels stacked open at once —
   add-player, chip advisor, routes and the expanded path — about 1,400px of
   controls with nothing saying which mattered. Everything below the two that
   do (your routes, and what the model suggests) now folds, remembers whether
   you opened it, and states in its header what it is for. */
export default function Section({ id, title, hint, badge, defaultOpen = false, children }) {
  const [open, setOpen] = usePersisted(`sec.${id}`, defaultOpen)
  return (
    <div className={`fold ${open ? 'open' : ''}`}>
      <button className="fold-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="fold-caret" aria-hidden="true">›</span>
        <span className="fold-title">{title}</span>
        {hint && <span className="fold-hint">{hint}</span>}
        {badge != null && <span className="fold-badge">{badge}</span>}
      </button>
      {open && <div className="fold-body">{children}</div>}
    </div>
  )
}
