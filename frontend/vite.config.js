import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Allow JSX syntax in .js files (avoids renaming all source files to .jsx)
  esbuild: {
    include: /\.(jsx?|tsx?)$/,
    exclude: [],
    loader: 'jsx',
  },

  // Serve at /app/ to match FastAPI's static mount point
  base: '/app/',
  build: {
    outDir: 'dist',
    // Content-hash filenames are the default; this is explicit for clarity
    rollupOptions: {
      output: {
        entryFileNames: 'assets/[name].[hash].js',
        chunkFileNames: 'assets/[name].[hash].js',
        assetFileNames: 'assets/[name].[hash][extname]',
      },
    },
  },
  server: {
    // Proxy API calls to the FastAPI server during local dev
    proxy: {
      '/ask': 'http://localhost:8000',
      '/auth': 'http://localhost:8000',
      '/chat': 'http://localhost:8000',
      '/admin': { target: 'http://localhost:8000', changeOrigin: false },
      '/ingest-file': 'http://localhost:8000',
      '/ingest-jobs': 'http://localhost:8000',
      '/feedback': 'http://localhost:8000',
      '/drug-price': 'http://localhost:8000',
      '/db': 'http://localhost:8000',
      '/uploads': 'http://localhost:8000',
    },
  },
})
