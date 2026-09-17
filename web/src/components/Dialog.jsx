import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'

/* In-app confirmations and small forms, in place of window.confirm/prompt.

   The browser's own dialogs look like a different product, cannot say which
   choice is destructive, block the whole tab, and on a phone sometimes
   silently refuse to open. This is one dialog for the whole app, called like
   the thing it replaces but awaited:

     const { confirm, form } = useDialog()
     if (await confirm({ title: 'Delete Draft A?', danger: true })) ...
     const v = await form({ title: 'Rename', fields: [{ name: 'label', value }] })

   `confirm` resolves true/false; `form` resolves the values object, or null
   when cancelled. Escape and the backdrop cancel; Enter submits. */
const DialogCtx = createContext(null)

export function DialogProvider({ children }) {
  const [dlg, setDlg] = useState(null)       // { kind, opts, resolve }

  const confirm = useCallback((opts) => new Promise((resolve) => {
    setDlg({ kind: 'confirm', opts, resolve })
  }), [])
  const form = useCallback((opts) => new Promise((resolve) => {
    setDlg({ kind: 'form', opts, resolve })
  }), [])

  const close = (value) => {
    if (dlg) dlg.resolve(value)
    setDlg(null)
  }

  return (
    <DialogCtx.Provider value={{ confirm, form }}>
      {children}
      {dlg && <DialogView dlg={dlg} close={close} />}
    </DialogCtx.Provider>
  )
}

export function useDialog() {
  const ctx = useContext(DialogCtx)
  if (!ctx) throw new Error('useDialog must be used inside <DialogProvider>')
  return ctx
}

function DialogView({ dlg, close }) {
  const { kind, opts } = dlg
  const cancelValue = kind === 'confirm' ? false : null
  const fields = opts.fields || []
  const [values, setValues] = useState(() =>
    Object.fromEntries(fields.map((f) => [f.name, f.value ?? ''])))
  const [error, setError] = useState(null)
  const primary = useRef(null)
  const firstInput = useRef(null)

  useEffect(() => {
    (firstInput.current || primary.current)?.focus()
    firstInput.current?.select?.()
    const onKey = (e) => { if (e.key === 'Escape') close(cancelValue) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])                                     // eslint-disable-line react-hooks/exhaustive-deps

  const submit = (e) => {
    e?.preventDefault()
    if (kind === 'confirm') { close(true); return }
    const msg = opts.validate?.(values)
    if (msg) { setError(msg); return }
    close(values)
  }

  return (
    <div className="dlg-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) close(cancelValue) }}>
      <form className={`dlg ${opts.danger ? 'danger' : ''}`} role="dialog" aria-modal="true"
        aria-labelledby="dlg-title" onSubmit={submit}>
        <h3 id="dlg-title">{opts.title}</h3>
        {opts.body && <div className="dlg-body">{opts.body}</div>}
        {fields.map((f, i) => (
          <label key={f.name} className="dlg-field">
            <span>{f.label}</span>
            <input ref={i === 0 ? firstInput : undefined} type={f.type || 'text'}
              inputMode={f.type === 'number' ? 'numeric' : undefined}
              min={f.min} max={f.max} maxLength={f.maxLength}
              value={values[f.name]}
              onChange={(e) => { setError(null); setValues((v) => ({ ...v, [f.name]: e.target.value })) }} />
            {f.hint && <small>{f.hint}</small>}
          </label>
        ))}
        {error && <div className="dlg-error">{error}</div>}
        <div className="dlg-actions">
          <button type="button" className="pill-btn" onClick={() => close(cancelValue)}>
            {opts.cancelLabel || 'Cancel'}
          </button>
          <button ref={primary} type="submit" className={`pill-btn ${opts.danger ? 'danger-btn' : 'accent'}`}>
            {opts.confirmLabel || (kind === 'confirm' ? 'Confirm' : 'Save')}
          </button>
        </div>
      </form>
    </div>
  )
}
