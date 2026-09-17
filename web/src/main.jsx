import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import { StoreProvider } from './store'
import { DialogProvider } from './components/Dialog'
import './theme.css'
import './solver.css'
import './brand.css'

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <StoreProvider>
      <DialogProvider>
        <App />
      </DialogProvider>
    </StoreProvider>
  </React.StrictMode>,
)
