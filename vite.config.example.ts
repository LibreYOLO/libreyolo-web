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
  },
  build: {
    target: 'esnext',
  },
});
