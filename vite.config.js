import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  base: "/junction2024/",
  resolve: {
    alias: {
      // Alias pdfjs-dist module paths
      'pdfjs-dist/build/pdf': resolve(__dirname, 'node_modules/pdfjs-dist/build/pdf'),
      'pdfjs-dist/build/pdf.worker.min': resolve(__dirname, 'node_modules/pdfjs-dist/build/pdf.worker.min.js'),
    },
  },
});
