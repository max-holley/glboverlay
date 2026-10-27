import { defineConfig } from 'vite';

export default defineConfig({
  base: '/',
  build: { target: 'esnext', chunkSizeWarningLimit: 2000 },
  // The tracking worker loads MediaPipe's ES-module WASM loader with import(), so it must be a module worker.
  worker: { format: 'es' },
});
