import React from 'react'

/* Shared loading / empty / error states.

   Every tab had improvised its own, so the same situation looked different
   depending where you hit it — a bare sentence here, a spinner there, a panel
   with 60px of padding somewhere else. Consistency in the boring states is
   most of what separates a tool that feels finished from one that feels
   half-built, because these are the screens you see first and see often.

   Each takes a title and a hint, because "nothing here" without a reason is
   the least useful thing an interface can say. */

export function Empty({ mark = '◦', title, children, actions }) {
  return (
    <div className="empty-state">
      <div className="es-mark">{mark}</div>
      <h3>{title}</h3>
      {children ? <p>{children}</p> : null}
      {actions ? <div className="es-actions">{actions}</div> : null}
    </div>
  )
}

/* A spinner says "wait"; a skeleton says "wait, and here is the shape of what
   is coming" — which also stops the layout jumping when the data lands. */
export function TableSkeleton({ rows = 8 }) {
  return (
    <div aria-busy="true" aria-live="polite">
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="skeleton sk-row"
          style={{ opacity: 1 - i * (0.55 / rows) }} />
      ))}
    </div>
  )
}

export function Loading({ children = 'Loading…' }) {
  return (
    <div className="empty-state" aria-busy="true" aria-live="polite">
      <span className="spinner" />
      <p style={{ marginTop: 4 }}>{children}</p>
    </div>
  )
}

export function ErrorState({ title = 'Something went wrong', children, actions }) {
  return (
    <div className="empty-state">
      <div className="es-mark es-err">!</div>
      <h3>{title}</h3>
      {children ? <p>{children}</p> : null}
      {actions ? <div className="es-actions">{actions}</div> : null}
    </div>
  )
}
