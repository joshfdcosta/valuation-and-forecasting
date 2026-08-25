import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// GitHub Pages serves a project site from /<repo>/, not /. Set via env in CI
// only, so local dev keeps serving from root.
const base = process.env.GH_PAGES_BASE || '/'

// The frontend calls /api/* and Vite forwards to the FastAPI dev server, so
// the browser only ever sees one origin.
export default defineConfig({
  base,
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
