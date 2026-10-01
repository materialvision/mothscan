"""The analysis pipeline: walk every site, pick one moment per file, index it.

Structure
---------
``discover``   -> list of (site, path) for real audio only (REAPER peaks excluded)
``analyse_one``-> features + scoring -> one :class:`EventRecord`
``run``        -> parallel map over all files, streaming records into the index

The pipeline is resumable: files already present in the index for a site are
skipped unless ``force`` is set, and raw feature curves are cached so that
re-tuning weights later never needs to touch the 64 GB of WAV again.

Short files
-----------
Six of the 5,558 files are partial (deployment / retrieval truncation).  They
are analysed normally and flagged in ``EventRecord.duration_s``; the runner
reports them rather than silently dropping them.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator

from datetime import datetime, timezone

from .config import (
    FILE_SECONDS,
    SITE_ROOT,
    SITE_DIRS,
    AnalysisConfig,
    RunConfig,
    is_audio_file,
)
from .timeline import Timeline

#: Grid epoch used for slot indices; must match the timeline's default.
GRID_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
from .features import extract_features
from .index_store import (
    EventRecord,
    IndexStore,
    load_feature_cache,
    save_feature_cache,
)
from .noise import ContaminationConfig, SubstitutionConfig, classify
from .scoring import EventScorer
from .select import contamination_from_record, select_points, summarise

# A file shorter than this is treated as a partial / truncated recording.
FULL_FILE_SECONDS = 100.0


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def discover(
    sites: Iterable[str] = SITE_DIRS, root: Path | None = None
) -> list[tuple[str, Path]]:
    """All real audio files, site by site, chronologically ordered.

    REAPER ``.reapeaks`` peak caches live next to their WAVs and are excluded by
    :func:`mothscan.config.is_audio_file`; anything else non-audio is ignored.
    Nested date folders are walked too, so a site split into ``YYYYMMDD/``
    sub-directories works without changes.
    """
    root = Path(root or SITE_ROOT)
    out: list[tuple[str, Path]] = []
    for site in sites:
        d = root / site
        if not d.is_dir():
            continue
        files = [p for p in d.rglob("*") if p.is_file() and is_audio_file(p)]
        files.sort(key=lambda p: (p.parent.name, p.name))
        out.extend((site, p) for p in files)
    return out


def timestamp_from_name(name: str) -> str:
    """Parse AudioMoth's ``YYYYMMDD_HHMMSS`` prefix into an ISO-8601 UTC string."""
    stem = Path(name).stem
    core = stem.split("_")[0]
    time_part = stem.split("_")[1] if "_" in stem else ""
    if len(core) == 8 and core.isdigit():
        iso = f"{core[:4]}-{core[4:6]}-{core[6:8]}"
        if len(time_part) >= 6 and time_part[:6].isdigit():
            iso += f"T{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}Z"
        return iso
    return ""


# --------------------------------------------------------------------------
# One file
# --------------------------------------------------------------------------


@dataclass
class AnalyseOutcome:
    record: EventRecord | None
    error: str | None = None
    seconds: float = 0.0


def analyse_one(
    site: str,
    path: Path,
    cfg: AnalysisConfig,
    use_cache: bool = True,
    write_cache: bool = True,
    ordinal: int = 0,
) -> AnalyseOutcome:
    """Features + scoring for a single file, producing its chosen cut point."""
    t0 = time.perf_counter()
    try:
        feats = load_feature_cache(path, site) if use_cache else None
        if feats is None:
            feats = extract_features(path, cfg)
            if write_cache:
                try:
                    save_feature_cache(path, site, feats, cfg)
                except OSError:
                    pass  # a cache miss must never fail the analysis
        scorer = EventScorer(feats, cfg)
        best, alts = scorer.find_best()

        rec = EventRecord(
            site=site,
            source_path=str(path),
            source_name=path.name,
            timestamp=timestamp_from_name(path.name),
            index=ordinal,
            cut_s=round(best.start_s, 6),
            event_s=round(best.event_s, 6),
            duration_s=round(feats.duration_s, 3),
            score=best.score,
            score_norm=best.score_norm,
            novelty=best.novelty,
            transient=best.transient,
            contrast=best.contrast,
            flatness=best.flatness,
            tonality=best.terms.get("tonality", 0.0),
            noise_floor_db=feats.noise_floor_db,
            rms_db=best.rms_db,
            peak_db=best.peak_db,
            file_peak_db=feats.peak_db,
            passes_gate=best.passes_gate,
            alternatives=[a.as_row() for a in alts],
            low100=feats.low100,
            low200=feats.low200,
            tilt_db_oct=feats.tilt_db_oct,
            clip_run=feats.clip_run,
            clip_runs=feats.clip_runs,
            clip_frac=feats.clip_frac,
        )
        contam = classify(feats)
        rec.noise_score = contam.score
        rec.noise_wind = contam.wind
        rec.noise_clipped = contam.clipped
        rec.noise_sustained = contam.sustained
        return AnalyseOutcome(record=rec, seconds=time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
        return AnalyseOutcome(record=None, error=f"{type(exc).__name__}: {exc}",
                              seconds=time.perf_counter() - t0)


def _worker(args: tuple) -> tuple[str, str, AnalyseOutcome]:
    site, path_str, cfg, use_cache, write_cache, ordinal = args
    out = analyse_one(site, Path(path_str), cfg, use_cache, write_cache, ordinal)
    return site, path_str, out


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def run(
    run_cfg: RunConfig | None = None,
    store: IndexStore | None = None,
    force: bool = False,
    use_cache: bool = True,
    refresh_cache: bool = False,
    limit: int | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Analyse every discovered file and write the index.

    Returns a summary dict.  ``progress`` is called with a small status dict
    after each completion, which the CLI uses to print a live line.
    """
    run_cfg = run_cfg or RunConfig()
    cfg = run_cfg.analysis
    own_store = store is None
    store = store or IndexStore()

    files = discover(run_cfg.sites)
    if limit:
        files = files[:limit]

    run_id = store.start_run(run_cfg.sites, cfg.as_dict())
    t0 = time.perf_counter()

    # A file is skippable only if it is already indexed *and* its cached feature
    # set is current.  Analysis that changes (a new detector, a changed config)
    # bumps the cache version, which invalidates the cache and therefore forces
    # re-analysis even though an index row exists - otherwise `scan` would
    # happily report "0 analysed" while the index held stale numbers.
    done: dict[str, set[str]] = {}
    need_reanalysis: dict[str, set[str]] = {}
    if not force:
        for site in run_cfg.sites:
            done[site] = store.existing_names(site)
        if use_cache:
            for site in run_cfg.sites:
                need_reanalysis[site] = {
                    p.name
                    for s_, p in files
                    if s_ == site
                    and p.name in done.get(site, set())
                    and load_feature_cache(p, site) is None
                }
    todo = [
        (s, p) for s, p in files
        if force or p.name not in done.get(s, set())
        or p.name in need_reanalysis.get(s, set())
    ]

    summary = {
        "run_id": run_id,
        "discovered": len(files),
        "skipped": len(files) - len(todo),
        "reanalysed": sum(len(v) for v in need_reanalysis.values()),
        "analysed": 0,
        "errors": 0,
        "short_files": [],
        "wall_seconds": 0.0,
    }

    # `--force` re-analyses everything; whether it also *recomputes* is separate.
    # Without this, forcing a scan after adding a detector would rewrite the
    # index from stale caches and the new indicator would stay at its default.
    cache_read = use_cache and not refresh_cache
    tasks = [
        (site, str(path), cfg, cache_read, True, i)
        for i, (site, path) in enumerate(todo)
    ]

    buffer: list[EventRecord] = []
    error_lines: list[str] = []

    def flush() -> None:
        if buffer:
            store.upsert_many(buffer)
            buffer.clear()

    if run_cfg.workers <= 1:
        for task in tasks:
            site, path_str, out = _worker(task)
            _collect(out, site, path_str, buffer, summary, error_lines)
            if len(buffer) >= 50:
                flush()
                if progress:
                    progress(summary)
    else:
        with ProcessPoolExecutor(max_workers=run_cfg.workers) as pool:
            futures = [pool.submit(_worker, t) for t in tasks]
            for fut in as_completed(futures):
                site, path_str, out = fut.result()
                _collect(out, site, path_str, buffer, summary, error_lines)
                if len(buffer) >= 200:
                    flush()
                if progress and summary["analysed"] % 25 == 0:
                    progress(summary)
    flush()

    # ---- stage 2: nearby-hour substitution ---------------------------
    # Runs after every hour has been scored, because it needs the whole
    # deployment's contamination picture to decide what is worth replacing.
    summary["substitution"] = substitute_contaminated(
        store, run_cfg.sites, run_cfg.substitution, run_cfg.contamination
    )

    summary["wall_seconds"] = time.perf_counter() - t0
    summary["errors_detail"] = error_lines[:50]
    store.finish_run(
        run_id, summary["analysed"], len(summary["short_files"]),
        summary["wall_seconds"],
        status="ok" if not summary["errors"] else "partial",
    )
    if own_store:
        store.close()
    return summary


def substitute_contaminated(
    store: IndexStore,
    sites: Iterable[str],
    sub_cfg: SubstitutionConfig | None = None,
    contam_cfg: ContaminationConfig | None = None,
) -> dict:
    """Replace unusable hours with the nearest acceptable hour's excerpt.

    Reads the freshly written index back (cheap - it is 5.5 k small rows), groups
    by site and grid hour, and rewrites only the rows whose source changed.  The
    chosen point keeps its own hour for the timeline and the export's grid
    column, and gains ``origin_*`` naming where the audio actually came from.
    """
    sub_cfg = sub_cfg or SubstitutionConfig()
    contam_cfg = contam_cfg or ContaminationConfig()
    per_site: dict[str, dict] = {}
    stats: dict[str, dict] = {}

    for site in sites:
        recs = store.records(sites=[site])
        by_grid: dict[int, EventRecord] = {}
        for r in recs:
            dt = Timeline.parse_timestamp(r.timestamp)
            if dt is None:
                continue
            by_grid[Timeline.slot_of(dt, GRID_EPOCH)] = r
        if not by_grid:
            continue

        # The index already stores each recording's contamination verdict, so
        # reuse it rather than re-deriving it from indicators the record does
        # not keep.  One decision point, one source of truth.
        verdicts = {g: contamination_from_record(r) for g, r in by_grid.items()}
        points = select_points(site, by_grid, by_grid, sub_cfg, contam_cfg,
                               classes=verdicts)
        stats[site] = summarise(points)

        # Re-point the excerpt at the substitute's audio while keeping the row's
        # identity as *this hour*.  `source_name` deliberately stays put: it is
        # the row's unique key and what the index uses to name the hour, so it
        # must not be swapped for a neighbour's (which would collide).  The
        # renderer reads `origin_path` instead - see EventRecord.audio_path.
        updates = []
        for p in points:
            if not p.substituted:
                continue
            src = p.record
            own = p.own_record
            if src is own:
                continue
            updates.append(
                (
                    # `cut_s` keeps describing the hour's OWN recording; the
                    # borrowed excerpt's cut goes in `origin_cut_s`.  Keeping them
                    # separate is what stops the renderer reading the neighbour's
                    # file at this hour's offset and extracting the wrong audio.
                    own.cut_s, own.event_s, src.score, src.score_norm,
                    src.novelty, src.transient, src.contrast, src.flatness,
                    src.tonality, src.rms_db, src.peak_db,
                    int(bool(src.passes_gate)),
                    p.origin_grid, p.origin_name, src.source_path,
                    src.cut_s, src.event_s, p.reason,
                    p.own_score, int(bool(p.contamination.wind)),
                    int(bool(p.contamination.clipped)),
                    int(bool(p.contamination.sustained)),
                    site, own.source_name,
                )
            )
        if updates:
            with store.conn:  # one transaction: all of it, or none of it
                store.conn.executemany(
                    """UPDATE points SET
                         cut_s=?, event_s=?, score=?, score_norm=?, novelty=?,
                         transient=?, contrast=?, flatness=?, tonality=?,
                         rms_db=?, peak_db=?, passes_gate=?,
                         substituted=1, origin_grid=?, origin_name=?,
                         origin_path=?, origin_cut_s=?, origin_event_s=?,
                         substitution_reason=?,
                         noise_score=?, noise_wind=?, noise_clipped=?,
                         noise_sustained=?
                       WHERE site=? AND source_name=?""",
                    updates,
                )
        per_site[site] = {
            "hours": len(points),
            "substituted": len(updates),
        }

    totals = {
        "substituted": sum(v["substituted"] for v in per_site.values()),
        "hours": sum(v["hours"] for v in per_site.values()),
    }
    return {"per_site": per_site, "totals": totals, "detail": stats}


def _collect(
    out: AnalyseOutcome,
    site: str,
    path_str: str,
    buffer: list[EventRecord],
    summary: dict,
    error_lines: list[str],
) -> None:
    if out.record is None:
        summary["errors"] += 1
        error_lines.append(f"{path_str}: {out.error}")
        return
    summary["analysed"] += 1
    if out.record.duration_s < FULL_FILE_SECONDS:
        summary["short_files"].append(
            {"path": path_str, "duration_s": out.record.duration_s}
        )
    buffer.append(out.record)


# --------------------------------------------------------------------------
# Convenience
# --------------------------------------------------------------------------


def rescan_from_cache(
    cfg: AnalysisConfig | None = None,
    sites: Iterable[str] = SITE_DIRS,
    store: IndexStore | None = None,
) -> int:
    """Re-score every cached file with a new AnalysisConfig, no WAV reads.

    This is the tuning loop: adjust weights, rescan, and the index and preview
    update without touching the external drive.
    """
    cfg = cfg or AnalysisConfig()
    own = store is None
    store = store or IndexStore()
    files = discover(sites)
    n = 0
    batch: list[EventRecord] = []
    for ordinal, (site, path) in enumerate(files):
        feats = load_feature_cache(path, site)
        if feats is None:
            continue
        out = analyse_one(site, path, cfg, use_cache=True, write_cache=False,
                          ordinal=ordinal)
        if out.record is not None:
            batch.append(out.record)
            n += 1
        if len(batch) >= 500:
            store.upsert_many(batch)
            batch.clear()
    if batch:
        store.upsert_many(batch)
    if own:
        store.close()
    return n
