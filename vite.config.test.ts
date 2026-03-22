import { defineConfig } from 'vite';

export default defineConfig({
  root: 'examples/basic',
  base: './',
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  server: {
    fs: {
      allow: ['../..'],
    },
    headers: {
      // Required for SharedArrayBuffer (WASM threads)
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  build: {
    target: 'esnext',
  },
});
