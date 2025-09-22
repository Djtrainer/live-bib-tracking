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
      port: 5173,
      proxy: {
        // Proxy all /api requests to the FastAPI backend
        '/api': {
          target: apiBaseUrl,
          changeOrigin: true,
        },
        // Proxy WebSocket connections
        '/ws': {
          target: wsBaseUrl,
          ws: true,
          changeOrigin: true,
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
