#!/usr/bin/env node
/**
 * Headless behavioural verification of the mothscan viewer.
 *
 * Screenshots prove the pixels; this proves the *behaviour*: that the page
 * throws no JS errors, that every site lane renders, that hovering resolves a
 * source file and its absolute hour, that switching site keeps the playhead on
 * the same wall-clock hour, and that every export returns a real file.
 *
 *   node scripts/verify_ui.mjs [baseUrl]
 */
import { spawn } from "node:child_process";
import { setTimeout as sleep } from "node:timers/promises";
import { readdirSync, rmSync } from "node:fs";

const BASE = process.argv[2] || "http://127.0.0.1:8770";
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 9222;

function launch() {
  const proc = spawn(
    CHROME,
    [
      "--headless=new", "--disable-gpu", "--no-sandbox", "--no-first-run",
      "--disable-extensions", "--autoplay-policy=no-user-gesture-required",
      `--remote-debugging-port=${PORT}`,
      "--user-data-dir=/tmp/mothscan-verify",
      "--window-size=1600,1000",
      "about:blank",
    ],
    { stdio: ["ignore", "pipe", "pipe"] },
  );
  proc.stderr.on("data", () => {});
  return proc;
}

async function cdpTarget() {
  for (let i = 0; i < 60; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      const page = list.find((t) => t.type === "page");
      if (page) return page;
    } catch {}
    await sleep(500);
  }
  throw new Error("chrome devtools endpoint never came up");
}

class Session {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    ws.addEventListener("message", (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result);
      }
    });
  }
  send(method, params = {}) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }
  async eval(expression) {
    const r = await this.send("Runtime.evaluate", {
      expression, awaitPromise: true, returnByValue: true,
    });
    if (r.exceptionDetails) {
      throw new Error("page exception: " +
        JSON.stringify(r.exceptionDetails).slice(0, 400));
    }
    return r.result.value;
  }
}

const results = [];
function check(name, ok, detail = "") {
  results.push({ name, ok, detail });
  console.log(`  [${ok ? "PASS" : "FAIL"}] ${name}${detail ? "  — " + detail : ""}`);
}

/**
 * Count lit samples across every lane canvas on a 2-D grid.
 * A wide spectrogram has far more columns than the canvas, so a fixed 1-D
 * stride would alias onto a single (possibly dark) column.
 */
const PROBE_LANES = `(() => {
  const out = [];
  for (const cv of document.querySelectorAll('.lane canvas')) {
    if (!cv.width || !cv.height) { out.push(-1); continue; }
    const d = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
    let n = 0;
    for (let gy = 0; gy < 14; gy++) {
      const y = Math.floor((gy + 0.5) / 14 * cv.height);
      for (let gx = 0; gx < 64; gx++) {
        const x = Math.floor((gx + 0.5) / 64 * cv.width);
        const i = (y * cv.width + x) * 4;
        if (d[i] + d[i+1] + d[i+2] > 24) n++;
      }
    }
    out.push(n);
  }
  return out;
})()`;

const proc = launch();
let exitCode = 0;
try {
  const target = await cdpTarget();
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((res, rej) => {
    ws.addEventListener("open", res);
    ws.addEventListener("error", rej);
  });
  const s = new Session(ws);
  const consoleErrors = [];
  const failedRequests = [];
  ws.addEventListener("message", (ev) => {
    const m = JSON.parse(ev.data);
    if (m.method === "Log.entryAdded" && m.params.entry.level === "error")
      consoleErrors.push(m.params.entry.text);
    if (m.method === "Runtime.exceptionThrown")
      consoleErrors.push(m.params.exceptionDetails.exception?.description ||
                         m.params.exceptionDetails.text);
    if (m.method === "Network.loadingFailed")
      failedRequests.push(m.params.errorText);
  });
  await s.send("Runtime.enable");
  await s.send("Page.enable");
  await s.send("Log.enable");
  await s.send("Network.enable");

  // Purge anything a previous run left behind *before* loading the page: a
  // stale `verify_tmp` set is newer than the real one and would be auto-loaded,
  // making every later check measure the wrong preview.
  const dataDir = new URL("../../data/", import.meta.url).pathname;
  const purge = (sub, test) => {
    try {
      for (const f of readdirSync(dataDir + sub)) {
        if (test(f)) rmSync(dataDir + sub + f, { force: true, recursive: true });
      }
    } catch { /* directory may not exist yet */ }
  };
  // Must complete BEFORE the page loads: a leftover test set is newer than the
  // real one and the viewer would auto-load it, invalidating every later check.
  // `verify_tmp*` are this test's own renders; `scan_<stamp>*` are timestamped
  // renders left by earlier runs before named renders existed. Either would be
  // newer than the canonical `scan` set and auto-load first.
  purge("previews/", (f) => f.startsWith("verify_tmp") ||
                             /^scan_\d{8}-\d{6}/.test(f));
  purge("exports/", (f) => /^(bundle_|points_\d{8}-|cuepoints_\d{8}-|max_notes_\d{8}-)/.test(f));

  console.log(`verifying ${BASE}`);
  await s.send("Page.navigate", { url: BASE });
  await sleep(5000);

  check("page loaded", /mothscan/.test(await s.eval("document.title")));

  const state = await s.eval(`fetch('/api/state').then(r => r.json())`);
  check("index has points", state.index.total_points > 0,
        `${state.index.total_points} points`);
  check("a preview set is available", state.previews.length > 0,
        state.previews.map((p) => p.name).join(", "));

  const nSites = await s.eval("S.sites.length");
  check("viewer loaded the multi-site set", nSites >= 2 &&
        (await s.eval("S.meta.name")) === "scan",
        `set=${await s.eval("S.meta.name")}, ${nSites} sites`);

  const laneCount = await s.eval("document.querySelectorAll('.lane').length");
  check("one lane per site rendered", laneCount === nSites, `${laneCount} lanes`);

  // Wait for every lane's slot table and spectrogram bitmap.
  let lit = [];
  for (let i = 0; i < 60; i++) {
    lit = await s.eval(PROBE_LANES);
    if (Array.isArray(lit) && lit.length === nSites && lit.every((n) => n > 100)) break;
    await sleep(1000);
  }
  check("every lane's spectrogram drawn",
        Array.isArray(lit) && lit.length === nSites && lit.every((n) => n > 100),
        JSON.stringify(lit) + " of 896 per lane");

  const slotCounts = await s.eval(
    "S.sites.map(s => S.site[s].slots.length).join(',')");
  check("every site's slot table loaded",
        String(slotCounts).split(",").every((n) => +n > 100),
        `slots per site: ${slotCounts}`);

  // ---- alignment: the same wall-clock hour must map to one stream offset ----
  const align = await s.eval(`(() => {
    const sites = S.sites;
    const total = S.site[sites[0]].slots.length;
    let checked = 0, bad = 0, lenMismatch = 0;
    for (let i = 0; i < total; i++) {
      const g = S.site[sites[0]].slots[i].grid;
      const offs = new Set();
      for (const s of sites) {
        const sl = S.site[s].slots[i];
        if (!sl) { lenMismatch++; continue; }
        if (sl.grid !== g) { bad++; continue; }
        offs.add(sl.stream_s);
      }
      checked++;
      if (offs.size !== 1) bad++;
    }
    const origins = new Set(sites.map(s => S.meta.sites[s].grid_origin));
    return {checked, bad, lenMismatch, origins: [...origins],
            durations: sites.map(s => S.meta.sites[s].duration_s)};
  })()`);
  check("all sites share one grid origin and stream offsets",
        align.bad === 0 && align.lenMismatch === 0 && align.checked > 100 &&
        align.origins.length === 1,
        `${align.checked} hours checked, ${align.bad} mismatches, origin ${align.origins}`);

  check("aligned streams have identical lengths",
        new Set(align.durations).size === 1,
        JSON.stringify(align.durations));

  // ---- hover resolves source file + absolute hour ----
  const box = await s.eval(`(() => {
    const r = document.querySelector('.lane canvas').getBoundingClientRect();
    return {x: r.left + r.width * 0.5, y: r.top + r.height * 0.5};
  })()`);
  await s.send("Input.dispatchMouseEvent",
               { type: "mouseMoved", x: box.x, y: box.y, buttons: 0 });
  await sleep(700);
  const hoverText = await s.eval(
    "document.getElementById('ptab').innerText.replace(/\\s+/g,' ')");
  check("hover resolves source file and absolute hour",
        /WAV|no recording/i.test(hoverText) && /UTC/.test(hoverText),
        hoverText.slice(0, 130));

  // ---- click seeks ----
  const box2 = await s.eval(`(() => {
    const r = document.querySelector('.lane canvas').getBoundingClientRect();
    return {x: r.left + r.width * 0.7, y: r.top + r.height * 0.5};
  })()`);
  await s.send("Input.dispatchMouseEvent",
               { type: "mousePressed", x: box2.x, y: box2.y, button: "left", buttons: 1, clickCount: 1 });
  await s.send("Input.dispatchMouseEvent",
               { type: "mouseReleased", x: box2.x, y: box2.y, button: "left", buttons: 0, clickCount: 1 });
  await sleep(700);
  const sought = await s.eval(`(() => {
    const a = document.getElementById('audio');
    const dur = S.site[S.active].meta.duration_s;
    return {frac: a.currentTime / dur};
  })()`);
  check("click seeks the active site's audio",
        sought.frac > 0.5 && sought.frac < 0.9, `${(sought.frac * 100).toFixed(0)}%`);

  // ---- the displayed absolute hour must match the filename ----
  // Regression: the epoch was built with `Date.UTC(1970, 1, 1)`, and JS months
  // are 0-based, so 1 is February - every displayed hour was 31 days ahead.
  const clock = await s.eval(String.raw`(() => {
    const g = 496404;                       // 2026-08-18T12:00:00Z exactly
    const viaFunc = (typeof utcOf === 'function') ? utcOf(g) : '(utcOf gone)';
    // Cross-check straight from the slot metadata: filename vs displayed hour.
    let mismatches = 0, checked = 0, example = null;
    for (const site of S.sites) {
      for (const sl of S.site[site].slots) {
        if (!sl.audio || !sl.own_name) continue;
        // filename YYYYMMDD_HHMMSS -> expected ISO hour
        const m = /^(\d{4})(\d{2})(\d{2})_?(\d{2})/.exec(sl.own_name);
        if (!m) continue;
        checked++;
        const shown = (typeof utcOf === 'function') ? utcOf(sl.grid) : '';
        const want = m[1] + '-' + m[2] + '-' + m[3] + ' ' + m[4] + ':00:00 UTC';
        if (shown !== want) {
          mismatches++;
          if (!example) example = {file: sl.own_name, shown, want, grid: sl.grid};
        }
      }
    }
    return {viaFunc, checked, mismatches, example,
            sites: S.sites.length,
            perSite: S.sites.map(x => (S.site[x] && S.site[x].slots
                       ? S.site[x].slots.length : -1)),
            regexOk: /^(\d{4})(\d{2})(\d{2})_?(\d{2})/.test('20260818_120000.WAV')};
  })()`);
  check("displayed absolute hour matches the recording's filename",
        clock.checked > 1000 && clock.mismatches === 0 &&
        /^2026-08-18 12:00:00 UTC$/.test(clock.viaFunc),
        `${clock.checked} slots checked, ${clock.mismatches} mismatches` +
        (clock.example ? ` (e.g. ${clock.example.file} shown as ${clock.example.shown},` +
                         ` wanted ${clock.example.want})` : '') +
        `; epoch self-test -> ${clock.viaFunc}; sites ${clock.sites} ` +
        `slots/site ${JSON.stringify(clock.perSite)} regexOk ${clock.regexOk}`);

  // ---- the event must always sit near the START of its excerpt ----
  // This is the invariant that makes the excerpt length a free render-time
  // choice: if the event drifted into the tail (as it did when placement was a
  // fraction of the *analysed* window), shortening the cut would silently drop
  // it. Checked from the slot metadata the viewer actually uses.
  const placement = await s.eval(`(() => {
    const cut = S.meta.cut_seconds;
    let n = 0, outside = 0, worst = 0, sumOff = 0;
    for (const site of S.sites) {
      for (const sl of S.site[site].slots) {
        if (!sl.audio || sl.event_s === null || sl.cut_s === null) continue;
        const off = sl.event_s - sl.cut_s;   // event offset into the excerpt
        n++;
        sumOff += off;
        // "Outside" means at or past the end of the requested excerpt, which is
        // exactly the failure mode that shortening the cut used to trigger.
        if (off < 0 || off >= cut) outside++;
        if (off > worst) worst = off;
      }
    }
    return {cut, n, outside, worst, meanOff: n ? sumOff / n : -1};
  })()`);
  check("event always falls inside its excerpt",
        placement.n > 1000 && placement.outside === 0 && placement.worst < placement.cut,
        `${placement.n} slots of ${placement.cut}s: ${placement.outside} with the ` +
        `event outside, mean offset ${(placement.meanOff * 1000).toFixed(1)} ms, ` +
        `max ${(placement.worst * 1000).toFixed(1)} ms`);

  // ---- switching site keeps the same absolute hour ----
  // Measured from a *deliberately parked* playhead rather than mid-playback:
  // `currentTime` is quantised to the media element's seek precision (~1 frame
  // at 16 kHz), so reading a moving playhead would make this assertion flaky
  // rather than meaningful. One frame is 1/16000 s = 0.00011 slot at the
  // default 0.55 s slot, so half a slot is a very generous bound.
  const swap = await s.eval(`(async () => {
    const a = document.getElementById('audio');
    const per = S.meta.slot_seconds;
    a.pause();
    const park = 700;                        // a stable, well-inside offset
    a.currentTime = park;
    await new Promise(r => setTimeout(r, 600));
    const before = a.currentTime / per;
    const other = S.sites.find(s => s !== S.active);
    setActiveSite(other, true);
    await new Promise(res => {
      a.addEventListener('loadedmetadata', res, {once:true});
      setTimeout(res, 3000);
    });
    await new Promise(r => setTimeout(r, 400));
    const after = a.currentTime / per;
    return {other, before, after, drift: Math.abs(after - before),
            active: S.active, src: a.src.split('/').pop()};
  })()`);
  check("switching site keeps the same wall-clock hour",
        swap.drift < 0.05 && swap.active === swap.other,
        `${(swap.before * 0.55).toFixed(3)}s -> ${(swap.after * 0.55).toFixed(3)}s ` +
        `(drift ${(swap.drift * 0.55 * 1000).toFixed(0)} ms); now ${swap.active} (${swap.src})`);

  // ---- switching site while playing must keep playing ----
  // Assigning `src` pauses the media element, so the player has to resume on
  // the new stream; otherwise pressing play on another site goes silent.
  const swapPlaying = await s.eval(`(async () => {
    const a = document.getElementById('audio');
    a.muted = true;
    if (a.paused) { try { await a.play(); } catch (e) {} }
    await new Promise(r => setTimeout(r, 700));
    const wasPlaying = !a.paused;
    const other = S.sites.find(s => s !== S.active);
    setActiveSite(other, true);
    await new Promise(r => setTimeout(r, 3000));
    const stillPlaying = !a.paused && !a.ended;
    const t0 = a.currentTime;
    await new Promise(r => setTimeout(r, 1200));
    const advanced = a.currentTime - t0;
    return {wasPlaying, stillPlaying, advanced, active: S.active, flagged: S.playing};
  })()`);
  check("switching site while playing keeps playing",
        swapPlaying.stillPlaying && swapPlaying.advanced > 0.3,
        `was ${swapPlaying.wasPlaying}, now ${swapPlaying.stillPlaying} on ` +
        `${swapPlaying.active}, advanced ${swapPlaying.advanced.toFixed(2)}s`);

  // ---- playback ----
  const play = await s.eval(`(async () => {
    const a = document.getElementById('audio');
    a.muted = true;
    let err = null;
    a.addEventListener('error', () => { err = a.error ? a.error.message : 'error'; }, {once:true});
    try { await a.play(); } catch (e) { err = e.message; }
    const t0 = a.currentTime;
    await new Promise(r => setTimeout(r, 2500));
    const t1 = a.currentTime;
    a.pause();
    return {err, advanced: t1 - t0, duration: a.duration, site: S.active};
  })()`);
  check("the active site's stream decodes and plays",
        !play.err && play.advanced > 0.4,
        `${play.site}: advanced ${play.advanced.toFixed(2)}s of ${play.duration}s, err=${play.err}`);

  // ---- exports ----
  const exp = await s.eval(`(async () => {
    const out = {};
    for (const fmt of ['coll','cuepoints','csv','dict','maxpat']) {
      for (const scope of ['active','all']) {
        const p = new URLSearchParams({cut: '0.5', gap: '0.05'});
        if (scope === 'active') p.set('sites', S.active); else p.set('per_site','true');
        const r = await fetch('/api/export/' + fmt + '?' + p);
        out[fmt + ':' + scope] = {ok: r.ok, bytes: (await r.blob()).size};
      }
    }
    return out;
  })()`);
  const expBad = Object.entries(exp).filter(([, v]) => !v.ok || v.bytes < 200);
  check("all exports download (active site + all sites)",
        expBad.length === 0,
        expBad.length ? JSON.stringify(expBad)
                      : Object.entries(exp).map(([k, v]) => `${k}=${v.bytes}`).join(" "));

  const coll = await s.eval(`(async () => {
    const r = await fetch('/api/export/coll?cut=0.5&gap=0.05&sites=' +
                          encodeURIComponent(S.active));
    return (await r.text()).slice(0, 500);
  })()`);
  check("per-site coll is well formed with a grid column",
        /\n\d+ \d+ \d+ \d+ \d+/.test(coll) && /\.WAV"/.test(coll),
        JSON.stringify((coll.split("\n")[1] || "").slice(0, 130)));

  // ---- re-render with a different pulse and missing-hour policy ----
  const rendered = await s.eval(`(async () => {
    const r = await fetch('/api/render', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({cut_seconds: 0.4, gap_seconds: 0.2,
                            out_sample_rate: 16000, limit: 300,
                            out_name: 'verify_tmp_silence',
                            missing_policy: 'silence'})
    });
    return {status: r.status, body: await r.json()};
  })()`);
  check("re-render with new cut/gap works",
        rendered.status === 200 &&
        Object.values(rendered.body.sites || {}).every((v) => v.n_slots > 0),
        JSON.stringify(rendered.body.totals));

  const skipped = await s.eval(`(async () => {
    const r = await fetch('/api/render', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({cut_seconds: 0.4, gap_seconds: 0.2,
                            out_sample_rate: 16000, limit: 300,
                            out_name: 'verify_tmp_skip',
                            missing_policy: 'skip'})
    });
    const b = await r.json();
    return {status: r.status, totals: b.totals,
            sites: Object.fromEntries(Object.entries(b.sites).map(
              ([k, v]) => [k, v.n_silent_slots]))};
  })()`);
  check("skip policy drops the silent slots",
        skipped.status === 200 &&
        Object.values(skipped.sites).every((n) => n === 0),
        `silent slots per site: ${JSON.stringify(skipped.sites)}`);

  // ---- the limiter must actually be applied to the rendered audio ----
  const limiter = await s.eval(`(async () => {
    const mk = async (lim, ceiling) => {
      const r = await fetch('/api/render', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({cut_seconds: 0.5, gap_seconds: 0.05,
          out_sample_rate: 16000, limit: 400,
          sites: [S.active],
          out_name: lim ? 'verify_tmp_limon' : 'verify_tmp_limoff',
          limiter: lim, limiter_ceiling_dbfs: ceiling})
      });
      return r.ok;
    };
    const a = await mk(true, -12.0);
    const b = await mk(false, -12.0);
    // Read the rendered WAVs back through the browser and measure their peaks.
    const peakOf = async (name) => {
      const meta = await (await fetch('/api/previews/' + name)).json();
      const buf = await (await fetch('/media/' + meta.sites[S.active].wav)).arrayBuffer();
      const dv = new DataView(buf);
      // PCM_16 WAV: walk to the 'data' chunk, then find the max |sample|
      let off = 12, peak = 0;
      while (off + 8 <= dv.byteLength) {
        const id = String.fromCharCode(dv.getUint8(off), dv.getUint8(off+1),
                                       dv.getUint8(off+2), dv.getUint8(off+3));
        const sz = dv.getUint32(off+4, true);
        if (id === 'data') {
          for (let i = off+8; i + 1 < off+8+sz && i + 1 < dv.byteLength; i += 2) {
            const v = Math.abs(dv.getInt16(i, true)) / 32768;
            if (v > peak) peak = v;
          }
          break;
        }
        off += 8 + sz + (sz % 2);
      }
      return 20 * Math.log10(peak || 1e-12);
    };
    return {a, b, onPeak: await peakOf('verify_tmp_limon'),
            offPeak: await peakOf('verify_tmp_limoff')};
  })()`);
  // An all-silent pair is almost always the source volume being unmounted, not
  // a limiter fault - the renderer falls back to silence per unreadable excerpt.
  // Reporting that as "the limiter failed" sends you hunting a bug that does not
  // exist, so distinguish it explicitly.
  const bothSilent = limiter.onPeak < -100 && limiter.offPeak < -100;
  check("limiter caps the rendered peak",
        limiter.a && limiter.b && !bothSilent &&
        limiter.onPeak <= -11.5 && limiter.onPeak < limiter.offPeak - 0.5,
        bothSilent
          ? "both renders are silent - the source recordings are unreadable " +
            "(is the drive mounted?). Not a limiter fault."
          : `peak with limiter ${limiter.onPeak.toFixed(2)} dBFS vs without ` +
            `${limiter.offPeak.toFixed(2)} dBFS`);

  // ---- speed must mean "faster", and "pitch" must preserve pitch ----
  // Regression: the original implementation resampled, which *halved the sample
  // rate and played the buffer at the original rate* - so 2x produced double the
  // duration at an octave down. Varispeed had the same inversion. And `speed`
  // was never sent by the UI at all.
  const speed = await s.eval(`(async () => {
    const render = async (sp, mode) => {
      const r = await fetch('/api/render', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({cut_seconds: 0.5, gap_seconds: 0.05,
          out_sample_rate: 16000, limit: 200, sites: [S.active],
          speed: sp, speed_mode: mode,
          out_name: 'verify_tmp_sp' + sp + mode, limiter: true,
          limiter_ceiling_dbfs: -12})
      });
      return r.ok ? await r.json() : null;
    };
    const a = await render(1.0, 'pitch');
    const b = await render(2.0, 'pitch');
    const c = await render(2.0, 'varispeed');
    // The grid pitch (cut+gap) is fixed and must NOT change with speed - the
    // grid is the sync mechanism. What changes is how much of each slot is
    // occupied by sound, so that is what "faster" means here.
    const site = S.active;
    const duty = (res) => {
      const x = res.sites[site];
      return x.n_audio_slots * (x.occupied_s || 0) / res.totals.duration_s;
    };
    const meta1 = await (await fetch('/api/previews/' + a.name)).json();
    const meta2 = await (await fetch('/api/previews/' + b.name)).json();
    return {
      ok: !!(a && b && c),
      // Per-site slot_seconds is the *stream* pitch (compressed by speed); the
      // set-level one is the grid pitch in wall-clock seconds.
      baseSlot: meta1.sites[site].slot_seconds,
      pitchSlot: meta2.sites[site].slot_seconds,
      variSlot: meta1.sites[site].slot_seconds,
      occ1: meta1.sites[site].occupied_s,
      occ2: meta2.sites[site].occupied_s,
      dur1: meta1.sites[site].duration_s,
      dur2: meta2.sites[site].duration_s,
      scale1: meta1.sites[site].time_scale,
      scale2: meta2.sites[site].time_scale,
      mode1: meta1.speed_mode, mode2: meta2.speed_mode,
      nSlots: a.totals.n_slots,
    };
  })()`);
  // Regression: speed used to shrink only the audio inside a slot, leaving the
  // slot spacing at full length - so every slot was a short burst plus a long
  // silence, which sounds like a click train rather than a faster scan.
  check("speed compresses the whole stream (slot pitch scales too)",
        speed.ok &&
        Math.abs(speed.pitchSlot - speed.baseSlot / 2) < 1e-6 &&
        Math.abs(speed.occ1 - 0.5) < 0.01 &&
        Math.abs(speed.occ2 - 0.25) < 0.01 &&
        speed.dur2 < speed.dur1 * 0.55,
        `slot pitch ${speed.baseSlot}s -> ${speed.pitchSlot}s, audio per slot ` +
        `${speed.occ1}s -> ${speed.occ2}s, stream ${speed.dur1?.toFixed(0)}s -> ` +
        `${speed.dur2?.toFixed(0)}s at 2x (so the excerpt fills its slot)`);

  // Pitch preservation itself is asserted in the Python unit tests
  // (tests/test_preview.py measures f0 on tonal real material, where an
  // autocorrelation over a continuous excerpt is trustworthy). Measuring it
  // here would mean pitch-tracking a preview stream full of 50 ms gaps and
  // concatenated unrelated sounds, on which any estimator is unreliable - so
  // this checks the machinery instead: both modes are accepted and reported,
  // and they are genuinely different code paths.
  const modes = await s.eval(`(async () => {
    const out = {};
    for (const m of ['pitch', 'varispeed']) {
      const r = await fetch('/api/render', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({cut_seconds: 0.5, gap_seconds: 0.05,
          out_sample_rate: 16000, limit: 100, sites: [S.active],
          speed: 2.0, speed_mode: m, out_name: 'verify_tmp_mode' + m})
      });
      const res = r.ok ? await r.json() : null;
      out[m] = res ? {ok: true, speed: res.speed, mode: res.speed_mode} : {ok: false};
    }
    return out;
  })()`);
  check("both speed modes render and report themselves",
        modes.pitch.ok && modes.varispeed.ok &&
        modes.pitch.mode === 'pitch' && modes.varispeed.mode === 'varispeed' &&
        modes.pitch.speed === 2.0,
        `pitch -> speed ${modes.pitch.speed} mode ${modes.pitch.mode}; ` +
        `varispeed -> speed ${modes.varispeed.speed} mode ${modes.varispeed.mode}`);

  // ---- master stop must actually stop the audio ----
  const stopped = await s.eval(`(async () => {
    const a = document.getElementById('audio');
    a.muted = true;
    a.currentTime = 700;
    try { await a.play(); } catch (e) {}
    await new Promise(r => setTimeout(r, 800));
    const wasPlaying = !a.paused;
    const tBefore = a.currentTime;
    document.getElementById('stop').click();
    await new Promise(r => setTimeout(r, 400));
    const nowPlaying = !a.paused;
    const per = S.meta.slot_seconds;
    const slotStart = Math.floor(tBefore / per) * per;
    return {wasPlaying, nowPlaying, flagged: S.playing,
            tAfter: a.currentTime, slotStart,
            rewound: Math.abs(a.currentTime - slotStart) < 0.02};
  })()`);
  check("stop halts playback and rewinds to the excerpt start",
        stopped.wasPlaying && !stopped.nowPlaying && !stopped.flagged &&
        stopped.rewound,
        `was playing=${stopped.wasPlaying}, after stop playing=${stopped.nowPlaying}, ` +
        `position ${stopped.tBefore?.toFixed(2)}s -> ${stopped.tAfter?.toFixed(2)}s ` +
        `(excerpt start ${stopped.slotStart?.toFixed(2)}s)`);

  check("space bar toggles the transport",
        (await s.eval(`(() => {
          const before = !document.getElementById('audio').paused;
          window.dispatchEvent(new KeyboardEvent('keydown', {code: 'Space', bubbles: true}));
          return !document.getElementById('audio').paused !== before ? 'toggled' : 'no-change';
        })()`)) === 'toggled',
        "space starts/stops");

  // ---- the gap slider must not offer an impossible overlap ----
  const gapBounds = await s.eval(`(() => {
    const cut = document.getElementById('cut'), gap = document.getElementById('gap');
    cut.value = '0.5'; cut.dispatchEvent(new Event('input'));
    gap.value = gap.min; gap.dispatchEvent(new Event('input'));
    return {min: +gap.min, value: +gap.value, cut: +cut.value,
            pitch: +cut.value + +gap.value};
  })()`);
  check("gap slider cannot request a slot pitch of zero or less",
        gapBounds.pitch > 0 && gapBounds.min >= -gapBounds.cut,
        `cut ${gapBounds.cut}s, gap min ${gapBounds.min}s -> slot pitch ` +
        `${gapBounds.pitch?.toFixed(3)}s`);

  // ---- live playback speed must work and keep its pitch ----
  const playrate = await s.eval(`(async () => {
    const a = document.getElementById('audio');
    const r = document.getElementById('playrate');
    a.muted = true;
    r.value = '2';
    r.dispatchEvent(new Event('input'));
    await new Promise(res => setTimeout(res, 200));
    const applied = a.playbackRate;
    const preserves = ('preservesPitch' in a) ? a.preservesPitch
                    : ('mozPreservesPitch' in a) ? a.mozPreservesPitch : null;
    const label = document.getElementById('playrateval').textContent;
    // and does the playhead actually move faster?
    await loadStream('/media/' + S.site[S.active].meta.wav, 200, false);
    a.currentTime = 200;
    let err = null;
    try { await a.play(); } catch (e) { err = e.message; }
    const t0 = a.currentTime;
    await new Promise(res => setTimeout(res, 2000));
    const advanced = a.currentTime - t0;
    a.pause();
    r.value = '1'; r.dispatchEvent(new Event('input'));
    return {applied, preserves, label, advanced, err};
  })()`);
  check("live playback speed applies, runs faster, and keeps pitch",
        playrate.applied === 2 && playrate.preserves === true &&
        !playrate.err && playrate.advanced > 3.0,
        `${playrate.label}, playbackRate ${playrate.applied}, ` +
        `preservesPitch ${playrate.preserves}, advanced ` +
        `${playrate.advanced.toFixed(2)}s of source time in 2s ` +
        `(2x would be ~4s)`);

  // ---- overlapping excerpts (negative gap) ----
  // The slot pitch is the mechanism that keeps the three sites sample-aligned, so
  // a negative gap must shorten the *pitch* without disturbing slot identity.
  const overlap = await s.eval(`(async () => {
    const mk = async (gap, cut, fade) => {
      const r = await fetch('/api/render', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({cut_seconds: cut, gap_seconds: gap,
          fade_seconds: fade, out_sample_rate: 16000, limit: 400,
          out_name: 'verify_tmp_ov' + String(gap).replace('-','m').replace('.','_'),
          limiter: true, limiter_ceiling_dbfs: -12})
      });
      if (!r.ok) return {ok: false, status: r.status, text: await r.text()};
      return {ok: true, body: await r.json()};
    };
    const seq = await mk(0.05, 0.5, 0.004);
    const ovl = await mk(-0.25, 0.5, 0.05);
    // Pitch must be cut + gap, and the stream correspondingly shorter.
    const pit = (x) => x.body.slot_seconds;
    return {
      seqOk: seq.ok, ovlOk: ovl.ok,
      seqPitch: seq.ok ? pit(seq) : null,
      ovlPitch: ovl.ok ? pit(ovl) : null,
      seqDur: seq.ok ? seq.body.totals.duration_s : null,
      ovlDur: ovl.ok ? ovl.body.totals.duration_s : null,
    };
  })()`);
  check("negative gap overlaps and shortens the slot pitch",
        overlap.seqOk && overlap.ovlOk &&
        Math.abs(overlap.ovlPitch - 0.25) < 1e-6 &&
        Math.abs(overlap.seqPitch - 0.55) < 1e-6 &&
        overlap.ovlDur < overlap.seqDur * 0.6,
        `gap +0.05 -> pitch ${overlap.seqPitch}s (${overlap.seqDur?.toFixed(0)}s) vs ` +
        `gap -0.25 -> pitch ${overlap.ovlPitch}s (${overlap.ovlDur?.toFixed(0)}s)`);

  const overlapAlign = await s.eval(`(async () => {
    const r = await fetch('/api/render', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({cut_seconds: 0.5, gap_seconds: -0.3, fade_seconds: 0.05,
        out_sample_rate: 16000, limit: 600, out_name: 'verify_tmp_ovalign',
        limiter: true, limiter_ceiling_dbfs: -12})
    });
    if (!r.ok) return {ok: false};
    const res = await r.json();
    await loadSet(res.name, {});
    let checked = 0, bad = 0;
    const n = Math.min(...S.sites.map(x => S.site[x].slots.length));
    for (let i = 0; i < n; i++) {
      const g = S.site[S.sites[0]].slots[i].grid;
      const offs = new Set(), grids = new Set();
      for (const x of S.sites) {
        const sl = S.site[x].slots[i];
        if (!sl) continue;
        grids.add(sl.grid); offs.add(sl.stream_s);
      }
      checked++;
      if (grids.size !== 1 || offs.size !== 1) bad++;
    }
    return {ok: true, checked, bad,
            durations: S.sites.map(x => S.meta.sites[x].duration_s)};
  })()`);
  check("overlapping keeps the sites sample-aligned",
        overlapAlign.ok && overlapAlign.bad === 0 && overlapAlign.checked > 100 &&
        new Set(overlapAlign.durations).size === 1,
        `${overlapAlign.checked} slots: ${overlapAlign.bad} mismatches, ` +
        `durations ${JSON.stringify(overlapAlign.durations?.map(d => +d.toFixed(2)))}`);

  // Speed compresses the stream, so it must compress it *identically* for every
  // site - otherwise a sped-up render would lose the alignment the grid exists
  // to provide.
  const speedAlign = await s.eval(`(async () => {
    const r = await fetch('/api/render', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({cut_seconds: 0.5, gap_seconds: 0.05, speed: 2.0,
        out_sample_rate: 16000, limit: 600, out_name: 'verify_tmp_spalign',
        limiter: true, limiter_ceiling_dbfs: -12})
    });
    if (!r.ok) return {ok: false, status: r.status};
    const res = await r.json();
    await loadSet(res.name, {});
    let checked = 0, bad = 0;
    const n = Math.min(...S.sites.map(x => S.site[x].slots.length));
    for (let i = 0; i < n; i++) {
      const offs = new Set(), grids = new Set();
      for (const x of S.sites) {
        const sl = S.site[x].slots[i];
        if (!sl) continue;
        grids.add(sl.grid); offs.add(sl.stream_s);
      }
      checked++;
      if (grids.size !== 1 || offs.size !== 1) bad++;
    }
    const m = await (await fetch('/api/previews/' + res.name)).json();
    return {ok: true, checked, bad,
            streamPitch: m.sites[S.sites[0]].slot_seconds,
            gridPitch: m.grid_slot_seconds,
            durations: S.sites.map(x => m.sites[x].duration_s)};
  })()`);
  check("speed compresses every site identically (alignment survives)",
        speedAlign.ok && speedAlign.bad === 0 && speedAlign.checked > 100 &&
        new Set(speedAlign.durations).size === 1 &&
        Math.abs(speedAlign.streamPitch - speedAlign.gridPitch / 2) < 1e-6,
        `${speedAlign.checked} slots: ${speedAlign.bad} mismatches; grid pitch ` +
        `${speedAlign.gridPitch}s -> stream pitch ${speedAlign.streamPitch}s at 2x; ` +
        `durations ${JSON.stringify(speedAlign.durations?.map(d => +d.toFixed(2)))}`);

  const overlapReject = await s.eval(`(async () => {
    const r = await fetch('/api/render', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({cut_seconds: 0.5, gap_seconds: -0.5,
        out_sample_rate: 16000, limit: 50, out_name: 'verify_tmp_bad'})
    });
    return r.status;
  })()`);
  check("a slot pitch of zero is refused, not silently rendered",
        overlapReject >= 400, `HTTP ${overlapReject} for gap = -cut`);

  // ---- after Render preview, the new stream must actually be loaded ----
  const reload = await s.eval(`(async () => {
    const a = document.getElementById('audio');
    // Simulate the real button: render via the API, then reload the set the way
    // the handler does, and report whether audio is ready and playable.
    const res = await (await fetch('/api/render', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({cut_seconds: 0.25, gap_seconds: 0.05,
        out_sample_rate: 16000, limit: 400, sites: [S.active],
        out_name: 'verify_tmp_reload',
        limiter: true, limiter_ceiling_dbfs: -12})
    })).json();
    await loadSet(res.name, {gridIndex: 100, wasPlaying: false});
    const fresh = S.meta.name === res.name;
    const dur = a.duration;
    const t0 = a.currentTime;
    a.muted = true;
    let err = null;
    try { await a.play(); } catch (e) { err = e.message; }
    await new Promise(r => setTimeout(r, 1500));
    const advanced = a.currentTime - t0;
    a.pause();
    return {fresh, dur, t0, advanced, err, readyState: a.readyState,
            src: a.src.split('/').pop()};
  })()`);
  check("render preview loads the new stream, ready to play",
        reload.fresh && !reload.err && reload.readyState >= 2 && reload.t0 > 10,
        `${reload.src}: readyState ${reload.readyState}, started at ` +
        `${reload.t0.toFixed(1)}s (kept its hour), advanced ` +
        `${reload.advanced.toFixed(2)}s, err=${reload.err}`);

  const realFailures = failedRequests.filter((f) => !/ERR_ABORTED/.test(f));
  // The "slot pitch <= 0 is refused" check deliberately provokes a 400, which
  // the browser logs as a console error. That is the expected outcome of that
  // test, not a page defect, so it is filtered out here.
  const realConsoleErrors = consoleErrors.filter(
    (e) => !/status of 400/.test(e));
  check("no JS console errors (ignoring the deliberate 400)",
        realConsoleErrors.length === 0,
        realConsoleErrors.slice(0, 3).join(" | ").slice(0, 240) || "none");
  check("no failed network requests (ignoring expected stream aborts)",
        realFailures.length === 0, realFailures.slice(0, 3).join(" | ") || "none");

  ws.close();
} catch (err) {
  console.error("verification error:", err.message);
  exitCode = 1;
} finally {
  proc.kill("SIGKILL");
}

// Remove everything this test generated, so a later run cannot pick up a
// one-site preview set and so the deliverables directory stays canonical.
try {
  const { readdirSync, rmSync } = await import("node:fs");
  const base = new URL("../../data/", import.meta.url).pathname;
  const clean = (sub, test) => {
    try {
      for (const f of readdirSync(base + sub)) {
        if (test(f)) rmSync(base + sub + f, { force: true });
      }
    } catch { /* best effort */ }
  };
  clean("previews/", (f) => f.startsWith("verify_tmp"));
  clean("exports/", (f) => /^(bundle_|points_\d{8}-|cuepoints_\d{8}-|max_notes_\d{8}-)/.test(f));
} catch { /* best effort */ }

const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length) {
  exitCode = 1;
  for (const f of failed) console.log(`  FAILED: ${f.name} — ${f.detail}`);
}
process.exit(exitCode);
