import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../app/static',
    // keep previous hashed bundles: a browser holding a cached index.html
    // still asks for the old script, and deleting it made the page blank
    emptyOutDir: false,
  },
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8410',
    },
  },
})
