"""Exports for Max/MSP and spreadsheets, per site and combined.

Every selected point lives on a shared hourly grid (see :mod:`mothscan.timeline`),
so an export always carries **both** coordinate systems:

* ``grid``   - the absolute hour index, identical across sites.  Two events from
  different sites with the same ``grid`` are the same wall-clock hour.  Recover
  the instant with ``epoch + grid * 3600 s``.
* ``stream_ms`` - the position inside that site's rendered WAV, which is what
  you feed to ``[sfplay~]`` when playing the concatenated preview.  With the
  default ``silence`` policy the two are related by
  ``stream_ms == (grid - grid_origin) * slot_seconds * 1000``.

Artefacts
---------
``points_<site>.coll``      one file per site, index-addressable
``points.coll``             all sites concatenated, with a ``site`` column
``cuepoints_<site>.coll``   two columns: ``stream_ms duration_ms``
``points_<site>.csv``       per site, for Numbers/Excel/pandas
``points.csv``              all sites
``points_<site>.dict.json`` per site, for Max ``[dict]``
``max_notes.txt``           column layout + working patch recipes
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .config import EXPORT_DIR
from .timeline import Timeline

#: Column layout advertised in the notes file.
COLL_COLUMNS = (
    ("index", "event index within this site's coll"),
    ("grid", "absolute hour index on the shared grid"),
    ("grid_origin", "grid index of stream slot 0 (for absolute maths)"),
    ("stream_ms", "position in this site's preview WAV, ms"),
    ("cut_ms", "cut position inside the source WAV, ms"),
    ("event_ms", "detected event position inside the source WAV, ms"),
    ("dur_ms", "excerpt duration, ms"),
    ("score", "hybrid score x 1000"),
    ("peak_db", "peak level x 10"),
    ("gate", "1 if the point passed the activity gate"),
    ("epoch_hour", "hours since 1970-01-01T00:00Z for absolute scheduling"),
    ("noise", "contamination score x 1000 (wind / clipping)"),
    ("flags", "wind*1 + clipped*2 + sustained*4"),
    ("substituted", "1 if the audio came from a nearby hour"),
    ("origin_grid", "grid hour the audio came from (-1 when not substituted)"),
    ("path", "quoted absolute source path"),
)


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _ms(seconds: float) -> int:
    return int(round(seconds * 1000.0))


def _slug(site: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in site)


# --------------------------------------------------------------------------
# Core: turn records into grid-aware rows
# --------------------------------------------------------------------------


def build_rows(
    records: Sequence,
    cut_seconds: float = 0.5,
    gap_seconds: float = 0.05,
    epoch: datetime | None = None,
    align: bool = True,
    speed: float = 1.0,
) -> dict[str, list[dict]]:
    """Group records by site and give each one its grid and stream coordinates.

    ``stream_ms`` assumes the aligned ``silence`` policy: **every** site's stream
    begins at the grid's first slot, so a given hour lands at the same
    millisecond offset in all of them.  This mirrors the renderer's default
    exactly - if the two disagreed, a Max patch would seek to the wrong place.

    ``align=False`` falls back to a per-site origin, giving the shortest stream
    that contains that site's data.
    """
    epoch = epoch or datetime(1970, 1, 1, tzinfo=timezone.utc)
    # A render-time speed compresses the stream, so the stream offsets are
    # `slot_pitch / speed`. Without this the coll's stream_ms would point into
    # the middle of a later excerpt - the grid index is unaffected, so the
    # alignment key stays valid either way.
    slot_pitch = (cut_seconds + gap_seconds) / max(speed, 1e-9)
    by_site: dict[str, list] = defaultdict(list)
    for r in records:
        by_site[r.site].append(r)

    parsed_by_site: dict[str, list[tuple[int, object]]] = {}
    for site, recs in by_site.items():
        parsed: list[tuple[int, object]] = []
        for r in sorted(recs, key=lambda x: x.timestamp):
            dt = Timeline.parse_timestamp(r.timestamp)
            if dt is None:
                continue
            parsed.append((Timeline.slot_of(dt, epoch), r))
        if parsed:
            parsed_by_site[site] = parsed

    global_origin = min(
        (g for parsed in parsed_by_site.values() for g, _ in parsed), default=0
    )

    out: dict[str, list[dict]] = {}
    for site, parsed in parsed_by_site.items():
        origin = global_origin if align else min(g for g, _ in parsed)
        rows = []
        for i, (grid, r) in enumerate(parsed):
            rows.append(
                {
                    "index": i,
                    "site": site,
                    "grid": grid,
                    "grid_origin": origin,
                    "stream_ms": _ms((grid - origin) * slot_pitch),
                    "cut_ms": _ms(r.cut_s),
                    "event_ms": _ms(r.event_s),
                    "dur_ms": _ms(cut_seconds),
                    "score": int(round(r.score * 1000)),
                    "peak_db": int(round(r.peak_db * 10)),
                    "gate": 1 if r.passes_gate else 0,
                    "epoch_hour": grid,
                    "speed": speed,
                    "noise": int(round(float(getattr(r, "noise_score", 0.0)) * 1000)),
                    "flags": (int(bool(getattr(r, "noise_wind", False)))
                              + 2 * int(bool(getattr(r, "noise_clipped", False)))
                              + 4 * int(bool(getattr(r, "noise_sustained", False)))),
                    "substituted": int(bool(getattr(r, "substituted", False))),
                    "origin_grid": (getattr(r, "origin_grid", None)
                                    if getattr(r, "substituted", False) else -1),
                    # The path is the file the audio was READ from, so for a
                    # substituted point it is the neighbour's file.
                    "path": (getattr(r, "origin_path", None)
                             or r.source_path).replace("\\", "/"),
                    "timestamp": r.timestamp,
                    "source_name": r.source_name,
                    "cut_s": r.cut_s,
                    "event_s": r.event_s,
                    "score_f": r.score,
                    "score_norm": r.score_norm,
                    "peak_db_f": r.peak_db,
                    "novelty": r.novelty,
                    "transient": r.transient,
                    "contrast": r.contrast,
                    "flatness": r.flatness,
                    "tonality": r.tonality,
                    "rms_db": r.rms_db,
                    "file_peak_db": r.file_peak_db,
                    "noise_floor_db": r.noise_floor_db,
                    "duration_s": r.duration_s,
                }
            )
        out[site] = rows
    return out


# --------------------------------------------------------------------------
# coll
# --------------------------------------------------------------------------


def _coll_line(row: dict, include_site: bool = False) -> str:
    parts = [
        str(row["index"]),
        str(row["grid"]),
        str(row["grid_origin"]),
        str(row["stream_ms"]),
        str(row["cut_ms"]),
        str(row["event_ms"]),
        str(row["dur_ms"]),
        str(row["score"]),
        str(row["peak_db"]),
        str(row["gate"]),
        str(row["epoch_hour"]),
        str(row["noise"]),
        str(row["flags"]),
        str(row["substituted"]),
        str(row["origin_grid"]),
    ]
    if include_site:
        parts.append(f'"{row["site"]}"')
    parts.append(f'"{row["path"]}"')
    return " ".join(parts)


def export_coll(
    records: Sequence,
    cut_seconds: float = 0.5,
    gap_seconds: float = 0.05,
    out_dir: Path | None = None,
    combined: bool = True,
    align: bool = True,
    speed: float = 1.0,
) -> dict[str, Path]:
    """Write Max ``coll`` files: one per site, plus a combined one.

    Event *i* of a site's file is coll line *i*+1 (line 1 is the header, which is
    why each file starts with a ``;`` comment).  Column order is
    :data:`COLL_COLUMNS`.
    """
    rows_by_site = build_rows(records, cut_seconds, gap_seconds, align=align,
                              speed=speed)
    out_dir = Path(out_dir or EXPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for s, rows in sorted(rows_by_site.items()):
        p = out_dir / f"points_{_slug(s)}.coll"
        lines = [
            f"; mothscan cue points - site {s} - {len(rows)} events "
            f"cut={cut_seconds}s gap={gap_seconds}s"
        ]
        lines += [_coll_line(r) for r in rows]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written[f"coll:{s}"] = p

    if combined and len(rows_by_site) > 1:
        p = out_dir / "points.coll"
        total = sum(len(v) for v in rows_by_site.values())
        lines = [f"; mothscan cue points - ALL SITES - {total} events"]
        for s in sorted(rows_by_site):
            lines += [_coll_line(r, include_site=True) for r in rows_by_site[s]]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written["coll:all"] = p
    return written


def export_cuepoints_coll(
    records: Sequence,
    cut_seconds: float = 0.5,
    gap_seconds: float = 0.05,
    out_dir: Path | None = None,
    align: bool = True,
    speed: float = 1.0,
) -> dict[str, Path]:
    """Two-column coll (``stream_ms duration_ms``) - the simplest Max mapping."""
    rows_by_site = build_rows(records, cut_seconds, gap_seconds, align=align,
                              speed=speed)
    out_dir = Path(out_dir or EXPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for s, rows in sorted(rows_by_site.items()):
        p = out_dir / f"cuepoints_{_slug(s)}.coll"
        lines = [f"; stream_ms duration_ms - site {s} ({len(rows)} cues)"]
        lines += [f'{r["stream_ms"]} {r["dur_ms"]}' for r in rows]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written[f"cuepoints:{s}"] = p
    return written


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------


CSV_FIELDS = [
    "index", "site", "timestamp", "source_name", "source_path",
    "grid", "grid_origin", "stream_ms", "cut_ms", "event_ms", "dur_ms",
    "score", "score_norm", "novelty", "transient", "contrast", "flatness",
    "tonality", "rms_db", "peak_db", "file_peak_db", "noise_floor_db",
    "duration_s", "passes_gate",
    "noise_score", "noise_wind", "noise_clipped", "noise_sustained",
    "substituted", "origin_grid", "origin_name",
]


def export_csv(
    records: Sequence,
    cut_seconds: float = 0.5,
    gap_seconds: float = 0.05,
    out_dir: Path | None = None,
    combined: bool = True,
    align: bool = True,
    speed: float = 1.0,
) -> dict[str, Path]:
    """Flat table per site plus a combined one, for Numbers/Excel/pandas."""
    rows_by_site = build_rows(records, cut_seconds, gap_seconds, align=align,
                              speed=speed)
    out_dir = Path(out_dir or EXPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def write(rows: list[dict], p: Path) -> None:
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                out = dict(r)
                out["passes_gate"] = bool(r["gate"])
                out["source_path"] = r["path"]
                w.writerow(out)

    for s, rows in sorted(rows_by_site.items()):
        p = out_dir / f"points_{_slug(s)}.csv"
        write(rows, p)
        written[f"csv:{s}"] = p

    if combined and len(rows_by_site) > 1:
        p = out_dir / "points.csv"
        write([r for s in sorted(rows_by_site) for r in rows_by_site[s]], p)
        written["csv:all"] = p
    return written


# --------------------------------------------------------------------------
# JSON for Max [dict] / [jit.coll]
# --------------------------------------------------------------------------


def export_dict_json(
    records: Sequence,
    cut_seconds: float = 0.5,
    gap_seconds: float = 0.05,
    out_dir: Path | None = None,
    align: bool = True,
    speed: float = 1.0,
) -> dict[str, Path]:
    rows_by_site = build_rows(records, cut_seconds, gap_seconds, align=align,
                              speed=speed)
    out_dir = Path(out_dir or EXPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for s, rows in sorted(rows_by_site.items()):
        p = out_dir / f"points_{_slug(s)}.dict.json"
        payload = {
            "site": s,
            "n": len(rows),
            "slot_seconds": (cut_seconds + gap_seconds) / max(speed, 1e-9),
            "grid_origin": rows[0]["grid_origin"] if rows else 0,
            "epoch": "1970-01-01T00:00:00Z",
            "points": [
                {
                    "i": r["index"],
                    "grid": r["grid"],
                    "stream_ms": r["stream_ms"],
                    "cut_ms": r["cut_ms"],
                    "event_ms": r["event_ms"],
                    "ts": r["timestamp"],
                    "file": r["source_name"],
                    "path": r["path"],
                    "score": round(r["score_f"], 4),
                    "peak_db": round(r["peak_db_f"], 2),
                    "gate": r["gate"],
                }
                for r in rows
            ],
        }
        p.write_text(json.dumps(payload), encoding="utf-8")
        written[f"dict:{s}"] = p
    return written


# --------------------------------------------------------------------------
# Wiring notes for the Max patch
# --------------------------------------------------------------------------


def export_maxpat_text(
    records: Sequence,
    cut_seconds: float = 0.5,
    gap_seconds: float = 0.05,
    out_dir: Path | None = None,
    align: bool = True,
    speed: float = 1.0,
) -> Path:
    rows_by_site = build_rows(records, cut_seconds, gap_seconds, align=align,
                              speed=speed)
    out_dir = Path(out_dir or EXPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "max_notes.txt"

    total = sum(len(v) for v in rows_by_site.values())
    site_lines = "\n".join(
        f"  {s:14s} {len(rows):5d} events   points_{_slug(s)}.coll   "
        f"(grid_origin {rows[0]['grid_origin']})"
        for s, rows in sorted(rows_by_site.items())
    )
    first_path = ""
    for s in sorted(rows_by_site):
        if rows_by_site[s]:
            first_path = rows_by_site[s][0]["path"]
            break
    first_site = next(iter(sorted(rows_by_site)), "moth-oculus")

    text = f"""mothscan -> Max/MSP wiring notes
================================================================
sites: {len(rows_by_site)}   events: {total}
cut: {cut_seconds}s   gap: {gap_seconds}s   slot: {cut_seconds + gap_seconds}s
generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}

FILES (one coll per site, plus a combined one)
----------------------------------------------
{site_lines}
  points.coll             all sites concatenated, with a site-name column
  cuepoints_<site>.coll   2 columns: stream_ms duration_ms
  points_<site>.csv       per site, for Numbers/Excel
  points_<site>.dict.json per site, for Max [dict]

Each site also has its own preview WAV, rendered alongside the index:
  data/previews/<name>_<site>.wav

points_<site>.coll COLUMNS
--------------------------
0  event index within this site
1  grid          absolute hour index on the shared timeline
2  grid_origin   grid index of this site's stream slot 0
3  stream_ms     position in THIS SITE's preview WAV, ms   <- sfplay~ on preview
4  cut_ms        cut position inside the source WAV, ms    <- sfplay~ on source
5  event_ms      detected event position inside the source WAV, ms
6  dur_ms        excerpt duration, ms
7  score x 1000
8  peak dBFS x 10
9  gate          1 = passed the activity gate
10 epoch_hour    hours since 1970-01-01T00:00Z (absolute scheduling)
11 noise         contamination score x 1000 (wind / clipping)
12 flags         wind*1 + clipped*2 + sustained*4
13 substituted   1 = the audio came from a DIFFERENT hour (see 14)
14 origin_grid   grid hour the audio came from, or -1
15 path          quoted absolute source path (a Max symbol)

WIND AND CLIPPING
-----------------
Wind-free and clip-free material is preferred automatically: a wind-dominated
hour is replaced by the nearest acceptable hour, and 13/14 record where the
audio really came from.  A handful of hours inside a long storm have no clean
neighbour within range; they keep 13=0 and a high 11 (noise), which is the
filter to use if you want to exclude them:

    column 11 <= 400   and   column 12 == 0     -> clean hours only

Line 1 of a coll file is a header; mothscan writes a leading ';' comment, so
event i is coll line i+1.

ALIGNMENT - THE POINT OF THE GRID
---------------------------------
Every site's stream starts at the same grid origin and uses the same slot pitch,
so for two sites
    grid_A == grid_B
means those two events are the SAME WALL-CLOCK HOUR and sit at the SAME offset
in their respective preview WAVs.  That is what makes cross-site comparison
valid: drive both [sfplay~] objects from one clock and the sites stay locked
together for the whole deployment.

Absolute instant of any event:
    epoch(1970-01-01T00:00Z) + grid * 3600 seconds

RECIPE A - one site, played from its own preview
------------------------------------------------
    [coll points_{first_site}.coll]
        |
    [line 3]            <- stream_ms
        |
    [sfplay~ 1] <- [open data/previews/<name>_{first_site}.wav]
    step with [uzi] or a [metro] + counter to walk event by event.

RECIPE B - original 2-minute recordings
---------------------------------------
    [coll points_<site>.coll]
        |
    +-- [line 4] -> cut_ms ----------------> start position
    +-- [line 11] -> symbol -> [open] -----> [sfplay~ 2]

    Example first path (column 11, event 0):
      {first_path}

RECIPE C - aligned multi-site comparison (the synced playback)
--------------------------------------------------------------
    Load each site's preview into its own [buffer~] / [sfplay~], then drive all
    of them from ONE time source:

        [metro 20] -> [accum] (ms) -> route to every [sfplay~] start/seek

    Because the streams share a grid origin, position P in site A and position P
    in site B are the same hour.  Route each [sfplay~] through its own gain~ and
    crossfade between sites to compare the same hour at three locations.

RECIPE D - granular / randomised
---------------------------------
    Read a site's coll into a [dict], pick keys with [random N] + [dict.get], or
    [zl sort] on column 7 to play only the strongest events of that site.

NOTES
-----
* All times are integer milliseconds; [line] and [sfplay~] take those directly.
* Column 9 (gate) lets you skip hours where nothing crossed the activity
  threshold - filter with [select 0] before counting.
* A site whose deployment ended early has silent slots at the end of its
  preview; its coll simply has no entries there.
* A site with missing hours can be rendered either with silent slots (sync
  preserved) or compacted; the coll carries the grid index either way, so
  scheduling stays correct.
* Absolute paths are exported, so a patch does not care where it lives.
* Re-export after re-tuning weights; the column order does not change.
"""
    p.write_text(text, encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# One call for everything
# --------------------------------------------------------------------------


def export_all(
    records: Sequence,
    cut_seconds: float = 0.5,
    gap_seconds: float = 0.05,
    out_dir: Path | None = None,
    speed: float = 1.0,
) -> dict[str, Path]:
    """Produce every export artefact; returns a name -> path map."""
    out_dir = Path(out_dir or EXPORT_DIR)
    out: dict[str, Path] = {}
    out.update(export_coll(records, cut_seconds, gap_seconds, out_dir, speed=speed))
    out.update(export_cuepoints_coll(records, cut_seconds, gap_seconds, out_dir,
                                     speed=speed))
    out.update(export_csv(records, cut_seconds, gap_seconds, out_dir, speed=speed))
    out.update(export_dict_json(records, cut_seconds, gap_seconds, out_dir,
                                speed=speed))
    out["notes"] = export_maxpat_text(records, cut_seconds, gap_seconds, out_dir,
                                      speed=speed)
    return out
