"""mothscan command line.

    python3 -m mothscan.cli scan            # analyse every file -> index
    python3 -m mothscan.cli scan --limit 500 --sites "moth oculus"
    python3 -m mothscan.cli retune          # re-score from cache with new weights
    python3 -m mothscan.cli stats           # what is in the index
    python3 -m mothscan.cli preview         # render the concatenated pulse train
    python3 -m mothscan.cli export          # max/msp coll + csv
    python3 -m mothscan.cli inspect FILE    # explain one file's pick
    python3 -m mothscan.cli serve           # browser viewer
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import (
    DATA_DIR,
    EXPORT_DIR,
    PREVIEW_DIR,
    SITE_DIRS,
    AnalysisConfig,
    PreviewConfig,
    RunConfig,
)
from .index_store import IndexStore, write_jsonl
from .pipeline import analyse_one, discover, rescan_from_cache, run


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _overlap_scale(v):
    """Parse --overlap-scale: 'auto', 'unity', or a fixed gain number."""
    if isinstance(v, str):
        if v.lower() in ("auto", "unity"):
            return v.lower()
        try:
            return float(v)
        except ValueError:
            raise SystemExit(
                f"--overlap-scale: expected auto, unity or a number, got {v!r}"
            )
    return v


def _pct(a: float, b: float) -> str:
    return f"{(100.0 * a / b if b else 0):5.1f}%"


def _progress_printer():
    start = time.time()

    def show(s: dict) -> None:
        el = time.time() - start
        n = s["analysed"]
        rate = n / el if el > 0 else 0
        remaining = (s["discovered"] - s["skipped"] - n) / rate if rate > 0 else 0
        print(
            f"\r  {n:5d}/{s['discovered'] - s['skipped']:5d} analysed  "
            f"{rate:5.1f} files/s  elapsed {el:5.0f}s  eta {remaining:5.0f}s  "
            f"errors {s['errors']}",
            end="", flush=True,
        )

    return show


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_scan(args: argparse.Namespace) -> int:
    sites = tuple(args.sites) if args.sites else SITE_DIRS
    cfg = AnalysisConfig()
    if args.event_offset is not None:
        object.__setattr__(cfg, "event_offset", args.event_offset)
        object.__setattr__(cfg, "event_lead_mode", "fraction")
    if args.event_lead is not None:
        object.__setattr__(cfg, "event_lead_s", args.event_lead)
    if args.weights:
        for spec in args.weights:
            name, _, val = spec.partition("=")
            attr = f"w_{name}"
            if not hasattr(cfg, attr):
                print(f"unknown weight {name!r}", file=sys.stderr)
                return 2
            object.__setattr__(cfg, attr, float(val))

    from .noise import ContaminationConfig, SubstitutionConfig

    run_cfg = RunConfig(
        analysis=cfg,
        sites=sites,
        workers=args.workers,
        contamination=ContaminationConfig(),
        substitution=SubstitutionConfig(
            mode=args.substitute, max_distance_h=args.substitute_window
        ),
    )
    print(f"scanning {sites} from the drive")
    print(f"  n_fft={cfg.n_fft} hop={cfg.hop} n_mels={cfg.n_mels} "
          f"scan={cfg.scan_seconds}s "
          + (f"event_lead={cfg.event_lead_s * 1000:.1f}ms"
             if cfg.event_lead_mode == "lead"
             else f"event_offset={cfg.event_offset:.2f} of window"))
    print(f"  weights novelty={cfg.w_novelty} transient={cfg.w_transient} "
          f"contrast={cfg.w_contrast} flatness={cfg.w_flatness} "
          f"tonality={cfg.w_tonality}")
    with IndexStore() as store:
        if args.reset:
            store.conn.execute("DELETE FROM points")
            store.conn.commit()
            print("  index cleared")
        summary = run(run_cfg, store=store, force=args.force,
                      use_cache=not args.no_cache,
                      refresh_cache=args.force or args.refresh,
                      limit=args.limit, progress=_progress_printer())
        print()
        records = store.records(order="chronological")
        n = write_jsonl(records)
        counts = store.counts()

    s = summary
    stale = s.get("reanalysed", 0)
    print(f"analysed {s['analysed']} files in {s['wall_seconds']:.0f}s "
          f"({s['analysed'] / max(s['wall_seconds'], 1e-9):.1f} files/s), "
          f"{s['errors']} errors, {s['skipped']} already current"
          + (f" ({stale} re-analysed: stale feature cache)" if stale else ""))
    if s["short_files"]:
        print(f"  {len(s['short_files'])} partial files (<100 s):")
        for d in s["short_files"][:8]:
            print(f"    {Path(d['path']).name}  {d['duration_s']:.1f}s")
    print(f"\nindex -> {DATA_DIR / 'mothscan.sqlite3'}  ({n} points, jsonl written)")
    for site, c in counts.items():
        if site.startswith("_"):
            continue
        print(f"  {site:14s} {c['n']:5d} points, {c['passes_gate']:5d} passed the "
              f"activity gate ({_pct(c['passes_gate'], c['n'])}), "
              f"mean score {c['mean_score']:.2f}")
    sub = s.get("substitution") or {}
    tot = sub.get("totals") or {}
    if tot.get("substituted"):
        print(f"\nnearby-hour substitution (wind / sustained clipping):")
        print(f"  {tot['substituted']} of {tot['hours']} hours took a neighbour's "
              f"excerpt ({100 * tot['substituted'] / max(1, tot['hours']):.2f}%)")
        for site, v in sorted((sub.get("per_site") or {}).items()):
            if v["substituted"]:
                print(f"    {site:14s} {v['substituted']:5d} / {v['hours']:5d}")
        detail = sub.get("detail") or {}
        for site, d in sorted(detail.items()):
            print(f"    {site:14s} wind={d['wind']:4d}  rail-touch={d['clipped']:4d}  "
                  f"sustained-clip={d['sustained_clip']:4d}")
    if s["errors_detail"]:
        print("\nerrors:")
        for line in s["errors_detail"]:
            print("  " + line)
    return 0


def cmd_retune(args: argparse.Namespace) -> int:
    cfg = AnalysisConfig()
    if args.weights:
        for spec in args.weights:
            name, _, val = spec.partition("=")
            attr = f"w_{name}"
            if not hasattr(cfg, attr):
                print(f"unknown weight {name!r}", file=sys.stderr)
                return 2
            object.__setattr__(cfg, attr, float(val))
    print("re-scoring every cached file with new weights (no WAV reads)")
    t0 = time.time()
    with IndexStore() as store:
        n = rescan_from_cache(cfg, sites=tuple(args.sites) if args.sites else SITE_DIRS,
                              store=store)
        write_jsonl(store.records(order="chronological"))
    print(f"  re-scored {n} files in {time.time() - t0:.1f}s")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    with IndexStore() as store:
        counts = store.counts()
        total = counts.pop("_total", 0)
        print(f"indexed points: {total}")
        for site, c in sorted(counts.items()):
            print(f"  {site:14s} {c['n']:5d} points  gate {_pct(c['passes_gate'], c['n'])}"
                  f"  mean score {c['mean_score']:.2f}")
        rows = store.conn.execute(
            """SELECT site, COUNT(DISTINCT substr(timestamp,1,10)) days,
                      MIN(timestamp) first, MAX(timestamp) last FROM points
               GROUP BY site ORDER BY site"""
        ).fetchall()
        print("\ncoverage:")
        for r in rows:
            print(f"  {r['site']:14s} {r['days']:4d} days   {r['first']} .. {r['last']}")
        print("\nmost interesting points:")
        for r in store.records(order="score", limit=10):
            print(f"  {r.site:14s} {r.timestamp}  cut {r.cut_s:7.3f}s  "
                  f"score {r.score:5.2f}  peak {r.peak_db:6.1f} dBFS")
        runs = store.conn.execute(
            "SELECT * FROM runs ORDER BY id DESC LIMIT 3"
        ).fetchall()
        if runs:
            print("\nrecent runs:")
            for r in runs:
                print(f"  #{r['id']} {r['started_at']} -> {r['finished_at']}  "
                      f"{r['n_files']} files  {r['wall_seconds'] or 0:.0f}s  {r['status']}")
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    from .preview import build_preview
    from .timeline import MissingPolicy

    cfg = PreviewConfig(
        cut_seconds=args.cut,
        gap_seconds=args.gap,
        fade_seconds=args.fade,
        overlap_scale=_overlap_scale(args.overlap_scale),
        speed=args.speed,
        speed_mode=args.speed_mode,
        normalise=args.normalise,
        out_sample_rate=args.rate,
        limiter=args.limiter,
        limiter_ceiling_dbfs=args.ceiling,
    )
    with IndexStore() as store:
        recs = store.records(
            sites=args.sites, order=args.order, limit=args.limit,
            only_gate=args.only_gate, min_score=args.min_score,
        )
    if not recs:
        print("no points in the index - run `scan` first", file=sys.stderr)
        return 1

    policy = MissingPolicy(args.missing)
    sites = sorted({r.site for r in recs})
    print(f"rendering {len(recs)} excerpts across {len(sites)} site(s), one WAV each:")
    print(f"  cut={cfg.cut_seconds}s gap={cfg.gap_seconds}s "
          f"slot={cfg.cut_seconds + cfg.gap_seconds}s speed={cfg.speed}x "
          f"rate={cfg.out_sample_rate}Hz")
    if args.event_lead is not None:
        object.__setattr__(cfg, "event_lead_s", args.event_lead)
    lead = max(cfg.event_lead_s, cfg.cut_seconds * 0.10)
    pitch = cfg.cut_seconds + cfg.gap_seconds
    if cfg.gap_seconds < 0:
        inst = cfg.cut_seconds / max(pitch, 1e-9)
        print(f"  OVERLAP: {abs(cfg.gap_seconds):.3f}s of each excerpt "
              f"plays under the next; {inst:.2f} sounding at once; "
              f"slot pitch {pitch:.3f}s ({100 * pitch / cfg.cut_seconds:.0f}% "
              f"of sequential length)")
        print(f"  crossfade: {cfg.fade_seconds}s (capped at cut/2), "
              f"overlap level: {cfg.overlap_scale}")
    print(f"  limiter: {'on, ceiling ' + str(cfg.limiter_ceiling_dbfs) + ' dBFS' if cfg.limiter else 'off'}")
    print(f"  event placement: {1000 * lead:.1f} ms into each excerpt "
          f"({100 * lead / cfg.cut_seconds:.0f}% in), so every excerpt contains "
          f"its event")
    print(f"  missing hours: {policy.value}"
          f"{' (site streams share one grid origin, so they stay aligned)' if args.align else ' (per-site origin: shorter, not aligned)'}")
    t0 = time.time()

    def prog(site: str, i: int, n: int) -> None:
        print(f"\r  {site}: {i}/{n} excerpts", end="", flush=True)

    res = build_preview(
        recs, cfg, name=args.name, policy=policy, align=args.align, progress=prog
    )
    print()
    for site in sorted(res.sites):
        r = res.sites[site]
        note = f", {r.n_silent_slots} silent" if r.n_silent_slots else ""
        print(f"  {site:14s} {r.duration_s / 60:6.2f} min  {r.n_audio_slots:5d} audio"
              f"{note}  -> {r.wav_path.name}")
    print(f"  slot pitch {res.slot_seconds}s; every site's stream starts at grid "
          f"origin {next(iter(res.sites.values())).grid_origin if res.sites else '-'}")
    print(f"  index {res.sidecar_path}")
    print(f"  rendered in {time.time() - t0:.1f}s")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .export import export_all

    with IndexStore() as store:
        recs = store.records(
            sites=args.sites, order=args.order, limit=args.limit,
            only_gate=args.only_gate, min_score=args.min_score,
        )
    if not recs:
        print("no points to export", file=sys.stderr)
        return 1
    from .export import (
        export_coll, export_csv, export_cuepoints_coll, export_dict_json,
        export_maxpat_text,
    )

    if getattr(args, "per_site_only", False):
        paths = {}
        sp = getattr(args, "speed", 1.0)
        paths.update(export_coll(recs, args.cut, args.gap, EXPORT_DIR,
                                 combined=False, speed=sp))
        paths.update(export_cuepoints_coll(recs, args.cut, args.gap, EXPORT_DIR,
                                          speed=sp))
        paths.update(export_csv(recs, args.cut, args.gap, EXPORT_DIR,
                                combined=False, speed=sp))
        paths.update(export_dict_json(recs, args.cut, args.gap, EXPORT_DIR, speed=sp))
        paths["notes"] = export_maxpat_text(recs, args.cut, args.gap, EXPORT_DIR,
                                            speed=sp)
    else:
        paths = export_all(
            recs, cut_seconds=args.cut, gap_seconds=args.gap, out_dir=EXPORT_DIR,
            speed=args.speed,
        )
    sites = sorted({r.site for r in recs})
    print(f"exported {len(recs)} points across {len(sites)} site(s): "
          f"{', '.join(sites)}")
    for name, p in sorted(paths.items()):
        print(f"  {name:26s} {p.name}")
    print(f"\nall files in {EXPORT_DIR}")
    print(f"grid: one coll per site (points_<site>.coll), plus points.coll combined;")
    print(f"      every site shares one grid origin, so the same hour is at the")
    print(f"      same stream_ms offset in each site's preview WAV.")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from .scoring import explain
    from .features import load_mono
    from .index_store import load_feature_cache
    from .scoring import EventScorer
    from .features import extract_features

    path = Path(args.file)
    if not path.is_absolute():
        hits = [p for _, p in discover() if p.name == args.file]
        if not hits:
            print(f"not found and not in any site: {args.file}", file=sys.stderr)
            return 1
        path = hits[0]
    site = path.parent.name
    cfg = AnalysisConfig()
    feats = load_feature_cache(path, site) or extract_features(path, cfg)
    scorer = EventScorer(feats, cfg)
    best, alts = scorer.find_best()
    x, sr = load_mono(path, cfg)
    peak = float(abs(x).max()) if x.size else 0.0
    print(f"{path}")
    print(f"  duration        {feats.duration_s:.3f} s")
    print(f"  noise floor     {feats.noise_floor_db:6.1f} dBFS")
    print(f"  file peak       {20 * __import__('math').log10(peak) if peak else -120:6.1f} dBFS")
    print(f"  gate            {scorer.floor_gate_db:6.1f} dBFS "
          f"(noise floor + {cfg.gate_floor_db:g})")
    print(f"  normalisers     " + "  ".join(
        f"{k}={v:.4g}" for k, v in scorer.norm_scale.items()))
    print(f"\n  chosen  {explain(scorer, best)}")
    print(f"          cut {best.start_s:.3f}s .. {best.start_s + cfg.scan_seconds:.3f}s"
          f"   (event at {cfg.event_offset:.0%} into the excerpt)")
    for a in alts:
        print(f"  alt     {explain(scorer, a)}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    host = args.host
    port = args.port
    print(f"mothscan viewer -> http://{host}:{port}")
    print("  (index and previews are read from data/; press ctrl-c to stop)")
    uvicorn.run("mothscan.webapp:app", host=host, port=port, log_level="warning",
                reload=False)
    return 0


def cmd_pick(args: argparse.Namespace) -> int:
    """Audition one moment immediately, without the index or the viewer."""
    import numpy as np
    import soundfile as sf

    from .features import extract_features, load_mono
    from .preview import apply_fade
    from .scoring import EventScorer

    path = Path(args.file)
    if not path.is_absolute():
        hits = [p for _, p in discover() if p.name == args.file]
        if not hits:
            print(f"not found: {args.file}", file=sys.stderr)
            return 1
        path = hits[0]
    cfg = AnalysisConfig()
    feats = extract_features(path, cfg)
    scorer = EventScorer(feats, cfg)
    best, _ = scorer.find_best()
    x, sr = load_mono(path, cfg)
    n = int(round(args.cut * sr))
    s = int(round(best.start_s * sr))
    seg = apply_fade(x[s : s + n], sr, 0.004)
    out = Path(args.out) if args.out else (PREVIEW_DIR / f"pick_{path.stem}.wav")
    sf.write(str(out), seg, sr, subtype="PCM_16")
    print(f"cut {best.start_s:.3f}s (event {best.event_s:.3f}s, "
          f"score {best.score:.2f}, peak {best.peak_db:.1f} dBFS) -> {out}")
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mothscan", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scan", help="analyse every file and build the index")
    p.add_argument("--sites", nargs="+", default=None)
    p.add_argument("--workers", type=int, default=max(1, (__import__("os").cpu_count() or 4) - 1))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true", help="re-analyse files already indexed")
    p.add_argument("--reset", action="store_true", help="clear the index first")
    p.add_argument("--no-cache", action="store_true", help="ignore the feature cache")
    p.add_argument("--refresh", action="store_true",
                   help="recompute cached features even if the cache looks current "
                        "(use after adding or changing a detector)")
    p.add_argument("--weights", nargs="+", default=None,
                   help="override weights, e.g. novelty=1 transient=1.5 tonality=0.4")
    p.add_argument("--event-lead", type=float, default=None,
                   help="pre-roll before the detected event, in seconds "
                        "(default 0.010667, ~2 analysis frames). The event then "
                        "sits the SAME short distance into the excerpt whatever "
                        "length you later cut.")
    p.add_argument("--event-offset", type=float, default=None,
                   help="legacy fractional placement: event position inside the "
                        "analysed window, 0..1 (implies --event-lead-mode fraction)")
    p.add_argument("--substitute", default="both",
                   choices=["both", "back", "forward", "silence"],
                   help="when an hour is wind-dominated or clipped, take the "
                        "nearest acceptable hour instead (default both)")
    p.add_argument("--substitute-window", type=int, default=6,
                   help="how many hours either side to search (default 6)")
    p.add_argument("--no-substitute", dest="substitute", action="store_const",
                   const="silence", help="flag bad hours instead of replacing them")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("retune", help="re-score cached features with new weights")
    p.add_argument("--sites", nargs="+", default=None)
    p.add_argument("--weights", nargs="+", default=None)
    p.set_defaults(func=cmd_retune)

    p = sub.add_parser("stats", help="summarise the index")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("preview", help="render the concatenated pulse train")
    p.add_argument("--cut", type=float, default=0.5)
    p.add_argument("--gap", type=float, default=0.05,
                   help="silence between excerpts; NEGATIVE overlaps them, so the "
                        "next excerpt starts while this one is still sounding "
                        "(e.g. -0.25 with a 0.5s cut = 2 sounding at once)")
    p.add_argument("--overlap-scale", default="auto",
                   help="level handling when overlapping: 'auto' (1/sqrt(overlap)), "
                        "'unity' (let overlaps add), or a fixed gain number")
    p.add_argument("--fade", type=float, default=0.004)
    p.add_argument("--speed", type=float, default=1.0,
                   help="playback speed factor; 2 = twice as fast")
    p.add_argument("--speed-mode", default="pitch", choices=["pitch", "varispeed"],
                   help="'pitch' keeps the pitch while changing duration "
                        "(phase vocoder); 'varispeed' resamples so pitch follows "
                        "speed, like a tape machine")
    p.add_argument("--rate", type=int, default=16000)
    p.add_argument("--normalise", action="store_true")
    p.add_argument("--limiter", dest="limiter", action="store_true", default=True,
                   help="push peaks down so loud events do not drown quiet ones "
                        "(default on)")
    p.add_argument("--no-limiter", dest="limiter", action="store_false",
                   help="render the raw excerpts, unlevelled")
    p.add_argument("--ceiling", type=float, default=-12.0,
                   help="limiter ceiling in dBFS (default -12; near 0 = off)")
    p.add_argument("--order", default="chronological",
                   choices=["chronological", "score", "score_norm"])
    p.add_argument("--sites", nargs="+", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--only-gate", action="store_true")
    p.add_argument("--min-score", type=float, default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--missing", default="silence", choices=["silence", "skip"],
                   help="hours a site has no recording for: pad with silence "
                        "(keeps every site aligned) or skip them (shorter stream)")
    p.add_argument("--no-align", dest="align", action="store_false",
                   help="let each site's stream start at its own first hour")
    p.add_argument("--event-lead", type=float, default=None,
                   help="pre-roll before the event when re-placing an excerpt "
                        "(default 0.010667 s). Placement is max(lead, 10%% of "
                        "the excerpt), so the event always sits near the start.")
    p.set_defaults(func=cmd_preview, align=True)

    p = sub.add_parser("export", help="write max/msp coll + csv exports")
    p.add_argument("--cut", type=float, default=0.5)
    p.add_argument("--gap", type=float, default=0.05,
                   help="silence between excerpts; NEGATIVE overlaps them, so the "
                        "next excerpt starts while this one is still sounding "
                        "(e.g. -0.25 with a 0.5s cut = 2 sounding at once)")
    p.add_argument("--overlap-scale", default="auto",
                   help="level handling when overlapping: 'auto' (1/sqrt(overlap)), "
                        "'unity' (let overlaps add), or a fixed gain number")
    p.add_argument("--order", default="chronological",
                   choices=["chronological", "score", "score_norm"])
    p.add_argument("--sites", nargs="+", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--only-gate", action="store_true")
    p.add_argument("--min-score", type=float, default=None)
    p.add_argument("--speed", type=float, default=1.0,
                   help="must match the speed the preview was rendered with, or "
                        "the exported stream_ms offsets will not line up")
    p.add_argument("--per-site-only", action="store_true",
                   help="write only the per-site coll files, not the combined one")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("inspect", help="explain one file's chosen moment")
    p.add_argument("file")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("pick", help="write one file's chosen excerpt as a wav")
    p.add_argument("file")
    p.add_argument("--cut", type=float, default=0.5)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_pick)

    p = sub.add_parser("serve", help="browser viewer")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8770)
    p.set_defaults(func=cmd_serve)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
