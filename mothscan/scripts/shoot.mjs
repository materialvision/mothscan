#!/usr/bin/env node
/**
 * Deterministic screenshot of the mothscan viewer.
 *
 * `--screenshot` fires as soon as the page load event settles, which is long
 * before a full-deployment spectrogram (tens of MB) has decoded and painted.
 * This waits for the canvas to actually contain the spectrogram before
 * capturing, so the image is representative.
 *
 *   node scripts/shoot.mjs <url> <outFile> [waitMs]
 */
import { spawn } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import { setTimeout as sleep } from "node:timers/promises";

const URL_ = process.argv[2] || "http://127.0.0.1:8770/";
const OUT = process.argv[3] || "data/shots/viewer.png";
const EXTRA_WAIT = Number(process.argv[4] || 2500);
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 9333;
const W = Number(process.env.SHOT_W || 1700);
const H = Number(process.env.SHOT_H || 1050);

const proc = spawn(
  CHROME,
  [
    "--headless=new", "--disable-gpu", "--no-sandbox", "--no-first-run",
    "--hide-scrollbars", "--force-device-scale-factor=1",
    `--remote-debugging-port=${PORT}`,
    "--user-data-dir=/tmp/mothscan-shoot",
    `--window-size=${W},${H}`,
    "about:blank",
  ],
  { stdio: ["ignore", "ignore", "ignore"] },
);

async function findTarget() {
  for (let i = 0; i < 80; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      const p = list.find((t) => t.type === "page");
      if (p) return p;
    } catch {}
    await sleep(400);
  }
  throw new Error("devtools never came up");
}

try {
  const target = await findTarget();
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((res, rej) => {
    ws.addEventListener("open", res);
    ws.addEventListener("error", rej);
  });
  let id = 0;
  const pending = new Map();
  ws.addEventListener("message", (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) {
      const { resolve, reject } = pending.get(m.id);
      pending.delete(m.id);
      m.error ? reject(new Error(JSON.stringify(m.error))) : resolve(m.result);
    }
  });
  const send = (method, params = {}) =>
    new Promise((resolve, reject) => {
      const i = ++id;
      pending.set(i, { resolve, reject });
      ws.send(JSON.stringify({ id: i, method, params }));
    });
  const evaluate = async (expression) => {
    const r = await send("Runtime.evaluate", { expression, returnByValue: true });
    return r.result?.value;
  };

  await send("Runtime.enable");
  await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride", {
    width: W, height: H, deviceScaleFactor: 1, mobile: false,
  });
  await send("Page.navigate", { url: URL_ });

  const specOk = `(() => {
    const cvs = [...document.querySelectorAll('.lane canvas')];
    if (!cvs.length) return -1;
    let total = 0;
    for (const cv of cvs) {
      if (!cv.width || !cv.height) continue;
      const d = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
      for (let gy = 0; gy < 14; gy++) {
        const y = Math.floor((gy + 0.5) / 14 * cv.height);
        for (let gx = 0; gx < 64; gx++) {
          const x = Math.floor((gx + 0.5) / 64 * cv.width);
          const i = (y * cv.width + x) * 4;
          if (d[i] + d[i+1] + d[i+2] > 24) total++;
        }
      }
    }
    return total;
  })()`;

  let n = -1;
  for (let i = 0; i < 90; i++) {
    n = await evaluate(specOk);
    if (n > 200) break;
    await sleep(1000);
  }
  console.log(`spectrogram lit samples: ${n} (across all lanes)`);
  await sleep(EXTRA_WAIT);

  const shot = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false });
  await writeFile(OUT, Buffer.from(shot.data, "base64"));
  console.log(`wrote ${OUT}`);
  ws.close();
} finally {
  proc.kill("SIGKILL");
}
