#!/usr/bin/env node
/**
 * Focused test: switching the audible site while playing must not stop the audio.
 *
 * Assigning `src` on a media element always pauses it, so the player has to
 * resume explicitly on the new stream. This test plays a site, switches to
 * another one the way the UI does, and asserts that audio is still advancing on
 * the new stream - including that the playhead stayed on the same hour.
 *
 *   node scripts/verify_playback_switch.mjs [baseUrl]
 */
import { spawn } from "node:child_process";
import { setTimeout as sleep } from "node:timers/promises";

const BASE = process.argv[2] || "http://127.0.0.1:8770";
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 9444;
const PROFILE = "/tmp/mothscan-switchtest";

const proc = spawn(CHROME, [
  "--headless=new", "--disable-gpu", "--no-sandbox", "--no-first-run",
  "--disable-extensions", "--autoplay-policy=no-user-gesture-required",
  "--hide-scrollbars", `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${PROFILE}`, "--window-size=1500,950", "about:blank",
], { stdio: ["ignore", "ignore", "ignore"] });

const results = [];
function check(name, ok, detail = "") {
  results.push({ name, ok, detail });
  console.log(`  [${ok ? "PASS" : "FAIL"}] ${name}${detail ? "  — " + detail : ""}`);
}

let exitCode = 0;
try {
  let target = null;
  for (let i = 0; i < 60 && !target; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      target = list.find((t) => t.type === "page") || null;
    } catch {}
    if (!target) await sleep(500);
  }
  if (!target) throw new Error("devtools never came up");

  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((res, rej) => {
    ws.addEventListener("open", res);
    ws.addEventListener("error", rej);
  });
  let id = 0;
  const pending = new Map();
  const jsErrors = [];
  ws.addEventListener("message", (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) {
      const { resolve, reject } = pending.get(m.id);
      pending.delete(m.id);
      m.error ? reject(new Error(JSON.stringify(m.error))) : resolve(m.result);
    }
    if (m.method === "Runtime.exceptionThrown")
      jsErrors.push(m.params.exceptionDetails.exception?.description ||
                   m.params.exceptionDetails.text);
  });
  const send = (method, params = {}) =>
    new Promise((resolve, reject) => {
      const i = ++id;
      pending.set(i, { resolve, reject });
      ws.send(JSON.stringify({ id: i, method, params }));
    });
  const ev = async (expression) => {
    const r = await send("Runtime.evaluate",
      { expression, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails)
      throw new Error("page exception: " + JSON.stringify(r.exceptionDetails).slice(0, 300));
    return r.result.value;
  };

  await send("Runtime.enable");
  await send("Page.enable");
  await send("Page.navigate", { url: BASE });

  console.log(`testing playback continuity on ${BASE}`);
  // Wait for the viewer to have loaded all lanes.
  let ready = false;
  for (let i = 0; i < 60 && !ready; i++) {
    await sleep(1000);
    ready = await ev("!!(S.meta && S.sites.length >= 2 && " +
                     "S.sites.every(s => S.site[s] && S.site[s].slots.length > 100))")
      .catch(() => false);
  }
  check("viewer ready with multiple sites", ready === true);

  const sites = await ev("JSON.stringify(S.sites)");
  check("at least two sites to switch between",
        JSON.parse(sites).length >= 2, sites);

  // ---- start playing the first site ----
  const started = await ev(`(async () => {
    const a = document.getElementById('audio');
    a.muted = true;
    a.currentTime = 600;                      // an hour well inside the deployment
    try { await a.play(); } catch (e) { return {err: e.message}; }
    await new Promise(r => setTimeout(r, 1500));
    return {err: null, playing: !a.paused, t: a.currentTime, site: S.active,
            flagged: S.playing};
  })()`);
  check("playback starts on the first site",
        started.err === null && started.playing && started.t > 600,
        `site=${started.site} t=${started.t?.toFixed(2)}s flagged=${started.flagged}`);

  // ---- switch site the way the UI button does ----
  const switched = await ev(`(async () => {
    const a = document.getElementById('audio');
    const per = S.meta.slot_seconds;
    const beforeSlot = a.currentTime / per;
    const beforeSite = S.active;
    const tBefore = a.currentTime;
    const other = S.sites.find(s => s !== S.active);
    setActiveSite(other, true);               // exactly what the lane button calls
    await new Promise(r => setTimeout(r, 4000));   // allow load + resume
    const tA = a.currentTime;
    await new Promise(r => setTimeout(r, 1500));
    const tB = a.currentTime;
    return {
      beforeSite, other, activeNow: S.active,
      src: a.src.split('/').pop(),
      beforeSlot, afterSlot: a.currentTime / per,
      playing: !a.paused && !a.ended,
      flagged: S.playing,
      advanced: tB - tA,
      duration: a.duration, readyState: a.readyState,
    };
  })()`);
  check("switching site does not stop the audio",
        switched.playing && switched.advanced > 0.3,
        `${switched.beforeSite} -> ${switched.activeNow} (${switched.src}); ` +
        `playing=${switched.playing}, advanced ${switched.advanced?.toFixed(2)}s, ` +
        `readyState=${switched.readyState}`);
  check("playhead stayed on the same wall-clock hour",
        Math.abs(switched.afterSlot - switched.beforeSlot) < 0.05,
        `${switched.beforeSlot?.toFixed(2)} -> ${switched.afterSlot?.toFixed(2)} slot`);
  check("transport flag agrees with the element",
        switched.flagged === switched.playing,
        `S.playing=${switched.flagged}, element playing=${switched.playing}`);

  // ---- switching back must also keep playing ----
  const back = await ev(`(async () => {
    const a = document.getElementById('audio');
    const back = S.sites.find(s => s !== S.active);
    setActiveSite(back, true);
    await new Promise(r => setTimeout(r, 4000));
    const t0 = a.currentTime;
    await new Promise(r => setTimeout(r, 1500));
    return {active: S.active, playing: !a.paused, advanced: a.currentTime - t0};
  })()`);
  check("switching back also keeps playing",
        back.playing && back.advanced > 0.3,
        `now ${back.active}, advanced ${back.advanced?.toFixed(2)}s`);

  // ---- a paused site can be resumed from its own lane button ----
  const resume = await ev(`(async () => {
    const a = document.getElementById('audio');
    a.pause();
    await new Promise(r => setTimeout(r, 400));
    const pausedFlag = S.playing;
    setActiveSite(S.active, true);            // click the active lane again
    await new Promise(r => setTimeout(r, 1500));
    return {pausedFlag, playing: !a.paused, flagged: S.playing};
  })()`);
  check("clicking the active site resumes it",
        resume.pausedFlag === false && resume.playing === true,
        `paused flag was ${resume.pausedFlag}, now playing=${resume.playing}`);

  check("no JS errors", jsErrors.length === 0, jsErrors.slice(0, 2).join(" | "));

  await ev("document.getElementById('audio').pause()").catch(() => {});
  ws.close();
} catch (err) {
  console.error("test error:", err.message);
  exitCode = 1;
} finally {
  proc.kill("SIGKILL");
}

const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length) {
  exitCode = 1;
  for (const f of failed) console.log(`  FAILED: ${f.name} — ${f.detail}`);
}
process.exit(exitCode);
