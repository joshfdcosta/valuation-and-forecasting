import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The frontend calls /api/* and Vite forwards to the FastAPI dev server, so
// the browser only ever sees one origin.
export default defineConfig({
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
