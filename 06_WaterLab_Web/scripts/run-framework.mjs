import { spawnSync } from "node:child_process";
import {existsSync} from "node:fs";
import { fileURLToPath } from "node:url";
import { readExecutionProfile } from "./execution-profile.mjs";

const [command, ...args] = process.argv.slice(2);
if (!["dev", "build"].includes(command)) throw new Error("Expected dev or build.");
// Generate research inputs from the shared workbook before dev or build.
if (!process.env.WATERLAB_RESEARCH_SYNCED) {
  const input = fileURLToPath(new URL("../../data/需水预测模型对照.xlsx", import.meta.url));
  if (existsSync(input)) {
    const sync = spawnSync(process.execPath, [fileURLToPath(new URL("./sync-research-input.mjs", import.meta.url)), input], {stdio:"inherit"});
    if (sync.error || sync.status !== 0) throw sync.error || new Error("论文SSP输入同步失败");
  } else console.warn("论文输入Excel不存在；请在论文SSP页面上传原模型工作簿。");
}
const managedLinux = readExecutionProfile() === "managed-linux";

if (managedLinux && command === "build") {
  const result = spawnSync("bash", [
    fileURLToPath(new URL("./build-verified.sh", import.meta.url)), ...args,
  ], { stdio: "inherit" });
  if (result.error) throw result.error;
  process.exit(result.status ?? 1);
}

// Import in this process so the preview owner retains its PID and signals.
const cli = new URL(managedLinux
  ? "../node_modules/vite/bin/vite.js"
  : "../node_modules/vinext/dist/cli.js", import.meta.url);
process.argv = [process.execPath, fileURLToPath(cli), command,
  ...(!managedLinux && command === "dev" ? ["--port", "5173"] : []), ...args];
await import(cli.href);
