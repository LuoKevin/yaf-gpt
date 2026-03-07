import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(import.meta.url);

const viteRoot = path.dirname(require.resolve("vite/package.json"));
const viteBin = path.join(viteRoot, "bin", "vite.js");
const args = process.argv.slice(2);
const esbuildBinary = require.resolve("esbuild/bin/esbuild");

const child = spawn(process.execPath, [viteBin, ...args], {
  stdio: "inherit",
  env: {
    ...process.env,
    ESBUILD_BINARY_PATH: esbuildBinary
  }
});

child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }

  process.exit(code ?? 0);
});
