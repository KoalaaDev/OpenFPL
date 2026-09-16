import React from 'react'

/* One broken panel must not take the page down.

   The chip advisor called `POS_ORDER.map` on an object. React's default
   response to a render error is to unmount the entire tree, so a single
   wrong line turned "show XI" into a **blank white screen** with the squad,
   the drafts and the unsaved plan all gone from view. That is the worst
   possible failure mode for a planner, and it is entirely avoidable: the
   boundary keeps the rest of the app alive and shows what broke, where.

   Wrapped around each tab, so a failure is contained to the tab that caused
   it and the others stay usable. */
export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { err: null }
  }

  static getDerivedStateFromError(err) {
    return { err }
  }

  componentDidCatch(err, info) {
    // the console is where a developer will look; the panel is for the user
    console.error(`[${this.props.where || 'app'}]`, err, info?.componentStack)
  }

  render() {
    if (!this.state.err) return this.props.children
    return (
      <div className="panel crash">
        <div className="crash-mark" aria-hidden="true">✕</div>
        <h3>{this.props.where ? `${this.props.where} hit a problem` : 'Something went wrong'}</h3>
        <p>
          The rest of the app is still working — switch tabs and come back, or
          reload. Nothing you have saved is affected.
        </p>
        <pre>{String(this.state.err?.message || this.state.err)}</pre>
        <div className="crash-actions">
          <button className="pill-btn accent" onClick={() => this.setState({ err: null })}>
            Try again
          </button>
          <button className="pill-btn" onClick={() => window.location.reload()}>
            Reload the page
          </button>
        </div>
      </div>
    )
  }
}
