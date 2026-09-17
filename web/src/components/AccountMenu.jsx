import React, { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import { useStore } from '../store'
import { useDialog } from './Dialog'

/* Sign in with Google, or the signed-in account: plan badge, team id, sign
   out. Anonymous visitors lose nothing by not signing in — their drafts and
   squad live in this browser's session — the account only makes those follow
   them to another device. That is said on the menu, because a login wall on
   a planner is how you lose the visitor. */
export default function AccountMenu() {
  const { auth, user, plan, entryId, setEntryId, signOut, setToast, status } = useStore()
  const { confirm } = useDialog()
  const [open, setOpen] = useState(false)
  const ref = useRef(null)

  useEffect(() => {
    if (!open) return undefined
    const onDoc = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [open])

  const google = auth?.google_login
  const pro = plan === 'pro'

  // Not signed in and sign-in is available: a plain, visible button. A
  // login hidden behind an avatar dropdown is a login nobody finds.
  if (!user && google) {
    return (
      <a className="pill-btn accent signin-btn" href={api.loginUrl(window.location.pathname)}
        title="Sign in with Google to keep your drafts and squad on any device">
        <GoogleG /> <span className="signin-lbl">Sign in</span>
      </a>
    )
  }

  return (
    <div className="acct" ref={ref}>
      <button className={`acct-btn ${user ? 'in' : ''}`} onClick={() => setOpen((o) => !o)}
        title={user ? user.email : 'account'}>
        {user?.picture
          ? <img src={user.picture} alt="" referrerPolicy="no-referrer" />
          : <span className="acct-initial">{user ? (user.name || user.email || '?')[0].toUpperCase() : '👤'}</span>}
        {user && <span className="acct-name">{(user.name || user.email).split(' ')[0]}</span>}
        {status?.plans_enforced && <span className={`plan-badge ${pro ? 'pro' : ''}`}>{pro ? 'PRO' : 'FREE'}</span>}
      </button>

      {open && (
        <div className="dd-menu acct-menu">
          {user ? (
            <>
              <div className="acct-row">
                <div className="acct-who">
                  <b>{user.name || 'Signed in'}</b>
                  <span>{user.email}</span>
                </div>
              </div>
              <div className="acct-row muted">
                Team ID: <b className="num">{entryId ? `#${entryId}` : 'not set'}</b>
                {entryId && (
                  <button className="link" onClick={() => { setEntryId(null); setOpen(false) }}>forget</button>
                )}
              </div>
              {status?.plans_enforced && (
                <div className="acct-row">
                  <span className={`plan-badge ${pro ? 'pro' : ''}`}>{pro ? 'PRO' : 'FREE'}</span>
                  {!pro && <button className="pill-btn accent" onClick={() => setToast({ kind: 'info', msg: 'Pro is not on sale yet — everything is free while FPLabs is in beta.' })}>Upgrade</button>}
                </div>
              )}
              {user.is_admin && <div className="acct-row muted">admin · can refresh data by hand</div>}
              <div className="acct-row">
                <button className="pill-btn" onClick={() => { setOpen(false); signOut() }}>Sign out</button>
                <button className="link danger" onClick={async () => {
                  setOpen(false)
                  const ok = await confirm({
                    title: 'Delete your FPLabs account?',
                    body: 'Your drafts, saved squad, transfer watch and settings are deleted with it. This cannot be undone.',
                    confirmLabel: 'Delete account', danger: true,
                  })
                  if (ok) {
                    signOut({ deleteAccount: true })
                    setToast({ kind: 'ok', msg: 'Account deleted.' })
                  }
                }}>Delete account</button>
              </div>
              <div className="acct-row muted legal-links">
                <a href="/privacy">Privacy</a> · <a href="/terms">Terms</a>
              </div>
            </>
          ) : (
            <>
              <div className="acct-row">
                <div className="acct-who">
                  <b>Save your work</b>
                  <span>Drafts and your squad are kept in this browser for 30 days.
                    Sign in to keep them on your account and pick up on any device.
                    We store only your name, email and avatar — see the <a href="/privacy">privacy policy</a>.</span>
                </div>
              </div>
              <div className="acct-row">
                {google ? (
                  <a className="google-btn" href={api.loginUrl(window.location.pathname)}>
                    <GoogleG /> Sign in with Google
                  </a>
                ) : (
                  <span className="muted" style={{ fontSize: 12 }}>
                    Sign-in is not set up on this server yet.
                  </span>
                )}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}

function GoogleG() {
  return (
    <svg width="18" height="18" viewBox="0 0 48 48" aria-hidden="true">
      <path fill="#EA4335" d="M24 9.5c3.5 0 6.6 1.2 9 3.5l6.7-6.7C35.6 2.6 30.2 0 24 0 14.6 0 6.5 5.4 2.6 13.3l7.8 6C12.3 13.5 17.7 9.5 24 9.5z" />
      <path fill="#4285F4" d="M46.5 24.5c0-1.6-.1-3.1-.4-4.5H24v9h12.7c-.6 3-2.3 5.5-4.8 7.2l7.5 5.8c4.4-4 7.1-10 7.1-17.5z" />
      <path fill="#FBBC05" d="M10.4 28.7c-.5-1.5-.8-3-.8-4.7s.3-3.2.8-4.7l-7.8-6C.9 16.5 0 20.1 0 24s.9 7.5 2.6 10.7l7.8-6z" />
      <path fill="#34A853" d="M24 48c6.2 0 11.6-2 15.4-5.6l-7.5-5.8c-2.1 1.4-4.8 2.2-7.9 2.2-6.3 0-11.7-4-13.6-9.7l-7.8 6C6.5 42.6 14.6 48 24 48z" />
    </svg>
  )
}
