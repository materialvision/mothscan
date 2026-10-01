"""FastAPI app: browse the per-site streams, compare them, re-render, export.

Endpoints
---------
``GET  /``                                   the viewer
``GET  /api/state``                          index summary + rendered sets
``GET  /api/previews``                       list rendered sets
``GET  /api/previews/{name}``                grid + per-site summaries + gaps
``GET  /api/previews/{name}/{site}/slots``    that site's slot table (JSONL)
``GET  /media/{file}``                       wav / spec / env / slots assets
``GET  /api/waveform``                       envelope slice for deep zoom
``POST /api/render``                         rebuild with new cut/gap/rate/policy
``GET  /api/export/{fmt}``                   Max/CSV downloads, per site
``POST /api/export``                         same, as JSON
``GET  /api/timeline``                       the hour grid, coverage and gaps

The server holds no analysis state: everything comes from the SQLite index and
the rendered assets, so it can be restarted at any time.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field

from .config import EXPORT_DIR, PREVIEW_DIR, PreviewConfig
from .index_store import IndexStore
from .preview import build_preview
from .timeline import Timeline

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
STATIC_DIR = WEB_DIR / "static"

app = FastAPI(title="mothscan viewer", version="0.2.0")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _store() -> IndexStore:
    return IndexStore()


def _sidecar(name: str) -> Path:
    safe = Path(name).name
    p = PREVIEW_DIR / f"{safe}.index.json"
    if not p.exists():
        raise HTTPException(404, f"no preview set named {safe!r}")
    return p


def _preview_meta(name: str) -> dict:
    return json.loads(_sidecar(name).read_text(encoding="utf-8"))


def _list_previews() -> list[dict]:
    out = []
    for p in sorted(PREVIEW_DIR.glob("*.index.json"),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sites = meta.get("sites", {})
        out.append({
            "name": meta.get("name", p.stem),
            "rendered_at": meta.get("rendered_at"),
            "sample_rate": meta.get("sample_rate"),
            "slot_seconds": meta.get("slot_seconds"),
            "cut_seconds": meta.get("cut_seconds"),
            "gap_seconds": meta.get("gap_seconds"),
            "n_sites": len(sites),
            "sites": {
                s: {
                    "duration_s": v.get("duration_s"),
                    "n_slots": v.get("n_slots"),
                    "n_audio_slots": v.get("n_audio_slots"),
                    "n_silent_slots": v.get("n_silent_slots"),
                    "missing_policy": v.get("missing_policy"),
                }
                for s, v in sites.items()
            },
            "totals": meta.get("totals"),
        })
    return out


def _slot_path(name: str, site: str) -> Path:
    entry = _preview_meta(name).get("sites", {}).get(site)
    if not entry:
        raise HTTPException(404, f"preview {name!r} has no site {site!r}")
    fname = entry.get("slots_file")
    if not fname:
        raise HTTPException(404, f"site {site!r} has no slot table")
    p = PREVIEW_DIR / Path(fname).name
    if not p.exists():
        raise HTTPException(404, "slot table missing on disk")
    return p


# --------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    page = STATIC_DIR / "index.html"
    if not page.exists():
        return HTMLResponse("<h1>mothscan</h1><p>web/static/index.html missing</p>",
                            status_code=500)
    return HTMLResponse(page.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


@app.get("/api/state")
def api_state() -> dict:
    with _store() as store:
        counts = store.counts()
        total = counts.pop("_total", 0)
        runs = [dict(r) for r in store.conn.execute(
            "SELECT * FROM runs ORDER BY id DESC LIMIT 5")]
    return {
        "index": {"total_points": total, "by_site": counts},
        "previews": _list_previews(),
        "runs": runs,
        "defaults": PreviewConfig().__dict__,
    }


@app.get("/api/previews")
def api_previews() -> dict:
    return {"previews": _list_previews()}


@app.get("/api/previews/{name}")
def api_preview(name: str) -> dict:
    """Grid, per-site summaries and gaps - but not the slot tables."""
    return _preview_meta(name)


@app.get("/api/previews/{name}/{site}/slots")
def api_slots(name: str, site: str) -> Response:
    p = _slot_path(name, site)
    return FileResponse(str(p), media_type="application/x-ndjson")


@app.get("/api/timeline")
def api_timeline(slot_seconds: float = 0.55, name: str | None = None) -> dict:
    """The hour grid plus every site's coverage and gaps.

    Prefers a rendered set's grid so the viewer and the audio agree exactly;
    otherwise derives the grid straight from the index.
    """
    if name:
        return _preview_meta(name).get("timeline", {})
    with _store() as store:
        recs = store.records()
    tl = Timeline(slot_seconds=slot_seconds)
    tl.add_records(recs)
    return {"summary": tl.summary(), "gaps": {s: tl.gaps(s) for s in tl.sites}}


@app.get("/api/waveform")
def api_waveform(name: str, site: str, t0: float, t1: float) -> dict:
    """Envelope slice for one site's stream, used when zoomed in."""
    entry = _preview_meta(name).get("sites", {}).get(site)
    if not entry:
        raise HTTPException(404, "unknown site")
    env_path = PREVIEW_DIR / Path(entry["assets"]["env"]).name
    if not env_path.exists():
        raise HTTPException(404, "envelope asset missing")

    raw = np.fromfile(env_path, dtype=np.float16, offset=4 + 3 * 8)
    pts = raw.reshape(-1, 3)
    rate = float(entry["assets"]["envelope_rate"])
    i0 = max(0, int(t0 * rate))
    i1 = min(pts.shape[0], int(math.ceil(t1 * rate)))
    if i1 <= i0:
        return {"t0": t0, "t1": t1, "rate": rate, "min": [], "max": [], "rms": []}
    sl = pts[i0:i1].astype(np.float32)
    return {
        "t0": t0, "t1": t1, "rate": rate,
        "min": np.round(sl[:, 0], 5).tolist(),
        "max": np.round(sl[:, 1], 5).tolist(),
        "rms": np.round(sl[:, 2], 6).tolist(),
    }


class RenderRequest(BaseModel):
    sites: list[str] | None = None
    cut_seconds: float = Field(0.5, gt=0.02, le=5.0)
    # Negative allowed: the excerpts overlap. Bound is -2 s, and the server also
    # rejects a slot pitch <= 0 because that would break site alignment.
    gap_seconds: float = Field(0.05, ge=-2.0, le=2.0)
    fade_seconds: float = Field(0.004, ge=0.0, le=1.0)
    overlap_scale: float | None = None
    speed: float = Field(1.0, gt=0.1, le=4.0)
    speed_mode: str = Field("pitch", pattern="^(pitch|varispeed)$")
    normalise: bool = False
    out_sample_rate: int = Field(16000, ge=8000, le=48000)
    out_name: str | None = None
    order: str = "chronological"
    limit: int | None = None
    missing_policy: str = Field("silence", pattern="^(silence|skip)$")
    align: bool = True
    limiter: bool = True
    limiter_ceiling_dbfs: float = Field(-12.0, ge=-60.0, le=0.0)


@app.post("/api/render")
def api_render(req: RenderRequest) -> dict:
    with _store() as store:
        recs = store.records(sites=req.sites, order=req.order, limit=req.limit)
    if not recs:
        raise HTTPException(400, "no points in the index for that selection")

    cfg = PreviewConfig(
        cut_seconds=req.cut_seconds,
        gap_seconds=req.gap_seconds,
        fade_seconds=req.fade_seconds,
        overlap_scale=req.overlap_scale if req.overlap_scale is not None else "auto",
        speed=req.speed,
        speed_mode=req.speed_mode,
        normalise=req.normalise,
        out_sample_rate=req.out_sample_rate,
        limiter=req.limiter,
        limiter_ceiling_dbfs=req.limiter_ceiling_dbfs,
    )
    try:
        pset = build_preview(
            recs, cfg, name=req.out_name, policy=req.missing_policy, align=req.align
        )
    except ValueError as exc:
        # A bad combination of cut/gap (a slot pitch <= 0, which would break the
        # grid alignment) is the caller's mistake, not a server fault.
        raise HTTPException(400, str(exc)) from exc
    return {
        "name": pset.name,
        "slot_seconds": pset.slot_seconds,
        "cut_seconds": pset.cut_seconds,
        "speed": pset.config.get("speed"),
        "speed_mode": pset.config.get("speed_mode"),
        "sample_rate": pset.sample_rate,
        "sites": {
            s: {
                "duration_s": r.duration_s,
                "n_slots": r.n_slots,
                "n_audio_slots": r.n_audio_slots,
                "n_silent_slots": r.n_silent_slots,
            }
            for s, r in pset.sites.items()
        },
        "totals": {
            "n_sites": len(pset.sites),
            "n_slots": pset.n_slots_total,
            "duration_s": pset.duration_total_s,
        },
    }


@app.get("/media/{filename}")
def media(filename: str) -> Response:
    """Serve preview assets (WAV / spec / env / slots) from the preview dir."""
    safe = Path(filename).name
    path = PREVIEW_DIR / safe
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "not found")
    media_type = {
        ".wav": "audio/wav",
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".spec": "application/octet-stream",
        ".env": "application/octet-stream",
    }.get(path.suffix, "application/octet-stream")
    return FileResponse(str(path), media_type=media_type)


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------


class ExportRequest(BaseModel):
    sites: list[str] | None = None
    order: str = "chronological"
    limit: int | None = None
    cut_seconds: float = 0.5
    gap_seconds: float = 0.05
    #: Must match the speed the preview was rendered with.
    speed: float = 1.0


#: fmt -> declared media type for the download
_MEDIA = {
    "coll": "text/plain",
    "cuepoints": "text/plain",
    "csv": "text/csv",
    "dict": "application/json",
    "maxpat": "text/plain",
}


def _stamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def run_export(fmt: str, req: ExportRequest, per_site: bool = False) -> Path:
    """Produce an export file and return its path.

    With ``per_site`` (or more than one site and the combined file not wanted)
    the per-site files are bundled into a single JSON document, because a
    browser download is one file; the bundle holds one entry per site with the
    exact coll/CSV text, so nothing is lost and each site stays separate.
    """
    from . import export as ex

    if fmt not in _MEDIA:
        raise HTTPException(400, f"unknown format {fmt!r}")

    with _store() as store:
        recs = store.records(sites=req.sites, order=req.order, limit=req.limit)
    if not recs:
        raise HTTPException(400, "no points to export")

    sites = sorted({r.site for r in recs})
    kw = {"cut_seconds": req.cut_seconds, "gap_seconds": req.gap_seconds,
          "speed": req.speed}
    stamp = _stamp()

    if fmt == "maxpat":
        return ex.export_maxpat_text(recs, out_dir=EXPORT_DIR, **kw)

    # For a single site, hand back the plain file: that is the common case and
    # what a Max patch wants to point at.
    if len(sites) == 1 and not per_site:
        written = {
            "coll": lambda: ex.export_coll(recs, out_dir=EXPORT_DIR, **kw),
            "cuepoints": lambda: ex.export_cuepoints_coll(recs, out_dir=EXPORT_DIR, **kw),
            "csv": lambda: ex.export_csv(recs, out_dir=EXPORT_DIR, **kw),
            "dict": lambda: ex.export_dict_json(recs, out_dir=EXPORT_DIR, **kw),
        }[fmt]()
        return next(iter(written.values()))

    written = {
        "coll": lambda: ex.export_coll(
            recs, out_dir=EXPORT_DIR, combined=not per_site, **kw),
        "cuepoints": lambda: ex.export_cuepoints_coll(recs, out_dir=EXPORT_DIR, **kw),
        "csv": lambda: ex.export_csv(
            recs, out_dir=EXPORT_DIR, combined=not per_site, **kw),
        "dict": lambda: ex.export_dict_json(recs, out_dir=EXPORT_DIR, **kw),
    }[fmt]()

    payload = {
        "format": fmt,
        "generated": stamp,
        "cut_seconds": req.cut_seconds,
        "gap_seconds": req.gap_seconds,
        "note": ("one entry per file; coll is index-addressable, event i is "
                 "line i+1"),
        "files": {k: v.read_text(encoding="utf-8") for k, v in written.items()},
    }
    if fmt == "dict":
        # Already JSON: merge rather than nest a second time.
        payload["files"] = {k: json.loads(v) for k, v in payload["files"].items()}
    p = EXPORT_DIR / f"bundle_{fmt}_{stamp}.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


@app.get("/api/export/{fmt}")
def api_export_get(
    fmt: str,
    sites: str | None = None,
    order: str = "chronological",
    limit: int | None = None,
    cut: float = 0.5,
    gap: float = 0.05,
    speed: float = 1.0,
    per_site: bool = False,
) -> Response:
    site_list = [s for s in sites.split(",") if s] if sites else None
    req = ExportRequest(sites=site_list, order=order, limit=limit,
                        cut_seconds=cut, gap_seconds=gap, speed=speed)
    path = run_export(fmt, req, per_site=per_site)
    return FileResponse(str(path), media_type=_MEDIA[fmt], filename=path.name)


@app.post("/api/export")
def api_export(req: ExportRequest, fmt: str = "coll", per_site: bool = False) -> dict:
    path = run_export(fmt, req, per_site=per_site)
    return {"file": path.name, "path": str(path)}
