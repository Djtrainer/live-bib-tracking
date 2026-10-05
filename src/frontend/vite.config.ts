import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // Get backend URL from environment variable, fallback to localhost
  const apiBaseUrl = process.env.VITE_API_BASE_URL || 'http://localhost:8001';
  const wsBaseUrl = process.env.VITE_WS_BASE_URL || 'ws://localhost:8001';

  return {
    server: {
      host: "::",
      port: process.env.PORT ? parseInt(process.env.PORT) : 8080,
      proxy: {
        // Proxy all /api requests to the FastAPI backend
        // xfwd: the API treats a request from this machine with no
        // forwarding header as the detector and asks it for no login. This
        // dev server listens on every interface, so without the header
        // anyone on the network would be the detector through it.
        '/api': {
          target: apiBaseUrl,
          changeOrigin: true,
          xfwd: true,
        },
        // Proxy WebSocket connections
        '/ws': {
          target: wsBaseUrl,
          ws: true,
          changeOrigin: true,
          xfwd: true,
        },
      },
    },
    build: {
      outDir: 'dist',
    },
    plugins: [
      react(),
      mode === 'development' &&
      componentTagger(),
    ].filter(Boolean),
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
  };
});
