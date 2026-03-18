import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const proxyTarget = process.env.FRONTEND_PROXY_TARGET || "http://127.0.0.1:8000";
const usePolling = process.env.CHOKIDAR_USEPOLLING === "true";
const pollingInterval = Number(process.env.CHOKIDAR_INTERVAL || "300");

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    watch: {
      usePolling,
      interval: pollingInterval
    },
    proxy: {
      "/api": {
        target: proxyTarget,
        changeOrigin: true
      },
      "/health": {
        target: proxyTarget,
        changeOrigin: true
      }
    }
  }
});
