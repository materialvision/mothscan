"""Index storage: the persistent record of every chosen cut point.

Two representations are kept deliberately:

* **SQLite** (``data/mothscan.sqlite3``) is the queryable working set - the
  browser viewer, filtering and re-export all read from it.
* **JSONL** (``data/points.jsonl``) is the portable, human-diffable dump, one
  selected point per line, written in chronological order.

A small binary cache of the raw per-file feature curves also lives here so that
re-scoring with different weights costs milliseconds instead of re-reading
64 GB of WAV.
"""

from __future__ import annotations

import json
import sqlite3
import zipfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .config import (
    CACHE_DIR,
    INDEX_DB,
    INDEX_JSONL,
    SAMPLE_RATE,
    AnalysisConfig,
)

SCHEMA_VERSION = 6


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------


@dataclass
class EventRecord:
    """One selected 0.5 s excerpt from one 2-minute recording."""

    # identity
    site: str
    source_path: str
    source_name: str
    timestamp: str = ""          # ISO-8601 UTC, derived from the filename
    index: int = 0               # chronological position within the run

    # where to cut
    cut_s: float = 0.0           # excerpt start, seconds into the source
    event_s: float = 0.0         # the detected event itself
    duration_s: float = 0.0      # source file duration

    # why
    score: float = 0.0
    score_norm: float = 0.0
    novelty: float = 0.0
    transient: float = 0.0
    contrast: float = 0.0
    flatness: float = 0.0
    tonality: float = 0.0

    # level / confidence
    noise_floor_db: float = -120.0
    rms_db: float = -120.0
    peak_db: float = -120.0
    file_peak_db: float = -120.0
    passes_gate: bool = True

    # retained alternatives, best first
    alternatives: list[dict] = field(default_factory=list)

    # --- contamination indicators (whole-recording) ---
    #: Sub-100 Hz share of energy; the primary wind discriminant.
    low100: float = 0.0
    #: Sub-200 Hz share, for context.
    low200: float = 0.0
    #: Long-term spectral slope, dB/octave.
    tilt_db_oct: float = 0.0
    #: Longest run of identical PCM samples.
    clip_run: int = 0
    #: Number of such runs.
    clip_runs: int = 0
    #: Fraction of samples inside those runs.
    clip_frac: float = 0.0
    #: 0..~2 contamination score; >= exclude_score means "do not cut here".
    noise_score: float = 0.0
    noise_wind: bool = False
    noise_clipped: bool = False
    noise_sustained: bool = False

    # --- nearby-hour substitution provenance ---
    #: True when the audio comes from a different hour than ``timestamp``.
    substituted: bool = False
    #: Grid index of the hour the audio actually came from.
    origin_grid: int | None = None
    #: Source filename of the hour the audio actually came from.
    origin_name: str | None = None
    #: Absolute path of the recording the audio actually came from.  When a
    #: substitution happened this is *not* ``source_path``, which keeps naming
    #: the hour's own file so the record stays identifiable in the index.
    origin_path: str | None = None
    #: Cut position inside the recording the audio came from.  When substituted
    #: this is the neighbour's cut, not this hour's.
    origin_cut_s: float | None = None
    #: Detected event inside the recording the audio came from.  Kept alongside
    #: ``origin_cut_s`` so the pair always describes ONE file: mixing this hour's
    #: event with a neighbour's cut made the renderer place the excerpt around an
    #: event that was not in the audio it was reading.
    origin_event_s: float | None = None
    #: Why a substitution was made.
    substitution_reason: str = ""

    # provenance
    schema_version: int = SCHEMA_VERSION
    analyzer: str = "mothscan"

    # -- helpers ---------------------------------------------------------

    @property
    def audio_path(self) -> str:
        """The file the excerpt must be read from.

        Substitution is the only case where this differs from ``source_path``.
        Keeping the two separate means a record is still identifiable by its own
        hour, while the renderer reads the audio it was told to read.
        """
        return self.origin_path or self.source_path

    @property
    def audio_name(self) -> str:
        return self.origin_name or self.source_name

    @property
    def audio_cut_s(self) -> float:
        """Cut position to use with :attr:`audio_path`.

        Falls back to ``cut_s`` for records that predate the separation.
        """
        if self.origin_cut_s is not None:
            return self.origin_cut_s
        return self.cut_s

    @property
    def audio_event_s(self) -> float:
        """Detected event that belongs to :attr:`audio_path`.

        ``event_s`` always describes *this hour's own* recording, which is what
        the viewer should report.  Placement, however, must use the event that is
        actually in the file being read - otherwise a substituted excerpt is
        positioned around an event from a different recording.
        """
        if self.origin_cut_s is not None:
            if self.origin_event_s is not None:
                return self.origin_event_s
            # Records written before origin_event_s existed: fall back to the
            # neighbour's cut, which is at least in the right file.
            return self.origin_cut_s
        return self.event_s

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_row(cls, row: sqlite3.Row | dict) -> "EventRecord":
        d = dict(row)
        alts = d.pop("alternatives_json", None)
        if alts:
            try:
                d["alternatives"] = json.loads(alts)
            except json.JSONDecodeError:
                d["alternatives"] = []
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in d.items() if k in known}
        for flag in ("noise_wind", "noise_clipped", "noise_sustained",
                     "substituted", "passes_gate"):
            if flag in clean:
                clean[flag] = bool(clean[flag])
        # `origin_grid` is NULL-able; keep it as None rather than 0.
        return cls(**clean)

    def db_dict(self) -> dict:
        d = asdict(self)
        d["alternatives_json"] = json.dumps(d.pop("alternatives"))
        return d


_DB_COLUMNS = {
    "site": "TEXT NOT NULL",
    "source_path": "TEXT NOT NULL",
    "source_name": "TEXT NOT NULL",
    "timestamp": "TEXT",
    "idx": "INTEGER",
    "cut_s": "REAL",
    "event_s": "REAL",
    "duration_s": "REAL",
    "score": "REAL",
    "score_norm": "REAL",
    "novelty": "REAL",
    "transient": "REAL",
    "contrast": "REAL",
    "flatness": "REAL",
    "tonality": "REAL",
    "noise_floor_db": "REAL",
    "rms_db": "REAL",
    "peak_db": "REAL",
    "file_peak_db": "REAL",
    "passes_gate": "INTEGER",
    "alternatives_json": "TEXT",
    "low100": "REAL",
    "low200": "REAL",
    "tilt_db_oct": "REAL",
    "clip_run": "INTEGER",
    "clip_runs": "INTEGER",
    "clip_frac": "REAL",
    "noise_score": "REAL",
    "noise_wind": "INTEGER",
    "noise_clipped": "INTEGER",
    "noise_sustained": "INTEGER",
    "substituted": "INTEGER",
    "origin_grid": "INTEGER",
    "origin_name": "TEXT",
    "origin_path": "TEXT",
    "origin_cut_s": "REAL",
    "origin_event_s": "REAL",
    "substitution_reason": "TEXT",
    "schema_version": "INTEGER",
    "analyzer": "TEXT",
}


class IndexStore:
    """SQLite-backed store of selected points."""

    def __init__(self, db_path: Path | str = INDEX_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create()

    def _create(self) -> None:
        cols = ",\n    ".join(f"{k} {v}" for k, v in _DB_COLUMNS.items())
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS points (
                {cols},
                PRIMARY KEY (site, source_name)
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_points_order ON points (site, timestamp)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_points_score ON points (score DESC)"
        )
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT, finished_at TEXT, sites TEXT,
                config_json TEXT, n_files INTEGER, n_short INTEGER,
                wall_seconds REAL, status TEXT
            )"""
        )
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after a table was first created."""
        have = {r[1] for r in self.conn.execute("PRAGMA table_info(points)")}
        for col, decl in _DB_COLUMNS.items():
            if col not in have:
                self.conn.execute(f"ALTER TABLE points ADD COLUMN {col} {decl}")

    # -- writes ----------------------------------------------------------

    def upsert_many(self, records: Iterable[EventRecord]) -> int:
        rows = []
        for r in records:
            d = r.db_dict()
            d["idx"] = d.pop("index")
            d["passes_gate"] = int(bool(d["passes_gate"]))
            for flag in ("noise_wind", "noise_clipped", "noise_sustained",
                         "substituted"):
                d[flag] = int(bool(d[flag]))
            d["schema_version"] = SCHEMA_VERSION
            rows.append(d)
        if not rows:
            return 0
        cols = list(_DB_COLUMNS.keys())
        placeholders = ",".join("?" for _ in cols)
        sql = (
            f"INSERT OR REPLACE INTO points ({','.join(cols)}) VALUES ({placeholders})"
        )
        self.conn.executemany(sql, [[r[c] for c in cols] for r in rows])
        self.conn.commit()
        return len(rows)

    def start_run(self, sites: Iterable[str], cfg: dict) -> int:
        from datetime import datetime, timezone

        cur = self.conn.execute(
            "INSERT INTO runs (started_at, sites, config_json, status) VALUES (?,?,?,?)",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                json.dumps(list(sites)),
                json.dumps(cfg),
                "running",
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, n_files: int, n_short: int, wall: float,
                   status: str = "ok") -> None:
        from datetime import datetime, timezone

        self.conn.execute(
            """UPDATE runs SET finished_at=?, n_files=?, n_short=?, wall_seconds=?,
               status=? WHERE id=?""",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                n_files, n_short, wall, status, run_id,
            ),
        )
        self.conn.commit()

    # -- reads -----------------------------------------------------------

    def records(
        self,
        sites: list[str] | None = None,
        min_score: float | None = None,
        only_gate: bool = False,
        limit: int | None = None,
        order: str = "chronological",
    ) -> list[EventRecord]:
        where, params = [], []
        if sites:
            where.append(f"site IN ({','.join('?' for _ in sites)})")
            params += list(sites)
        if min_score is not None:
            where.append("score >= ?")
            params.append(min_score)
        if only_gate:
            where.append("passes_gate = 1")
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        order_by = {
            "chronological": "site, timestamp, source_name",
            "score": "score DESC",
            "score_norm": "score_norm DESC",
        }.get(order, "site, timestamp, source_name")
        sql = f"SELECT * FROM points {clause} ORDER BY {order_by}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [EventRecord.from_row(r) for r in self.conn.execute(sql, params)]

    def counts(self) -> dict:
        out: dict = {}
        for row in self.conn.execute(
            "SELECT site, COUNT(*) n, SUM(passes_gate) g, AVG(score) s FROM points GROUP BY site"
        ):
            out[row["site"]] = {
                "n": row["n"], "passes_gate": row["g"] or 0, "mean_score": row["s"] or 0.0
            }
        out["_total"] = self.conn.execute("SELECT COUNT(*) FROM points").fetchone()[0]
        return out

    def existing_names(self, site: str) -> set[str]:
        return {
            r[0] for r in self.conn.execute(
                "SELECT source_name FROM points WHERE site=?", (site,)
            )
        }

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "IndexStore":
        return self

    def __exit__(self, *exc) -> None:
        self.conn.commit()
        self.close()


def write_jsonl(records: Iterable[EventRecord], path: Path | str = INDEX_JSONL) -> int:
    """Write records as JSONL, one point per line, chronologically ordered."""
    path = Path(path)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(r.to_json() + "\n")
            n += 1
    return n


def read_jsonl(path: Path | str = INDEX_JSONL) -> Iterator[EventRecord]:
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield EventRecord(**json.loads(line))


# --------------------------------------------------------------------------
# Feature cache (so re-tuning weights never needs the WAV files again)
# --------------------------------------------------------------------------


def cache_path_for(source_path: Path | str, site: str) -> Path:
    stem = Path(source_path).stem
    safe_site = site.replace(" ", "-")
    return CACHE_DIR / safe_site / f"{stem}.npz"


#: Scalar FileFeatures fields persisted alongside the curves.  Kept as an
#: explicit list so the cache stays lossless as detectors are added: a newer
#: cache can always be read by an older field set, and an older cache simply
#: reports the new fields as their defaults (``CACHE_VERSION`` then marks them
#: stale so `scan` refills them).
CACHE_SCALARS = (
    "frame_rate", "duration_s", "noise_floor_db", "peak", "peak_db",
    "ulf_ratio", "low100", "low200", "ulf_rel_db", "tilt_db_oct", "sustain", "crest",
    "clip_run", "clip_runs", "clip_frac",
)

#: Bumped whenever the analysis that produces cached values changes.
CACHE_VERSION = 4


def save_feature_cache(
    source_path: Path | str, site: str, feats, cfg: AnalysisConfig
) -> Path:
    p = cache_path_for(source_path, site)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        novelty=feats.novelty,
        env_fast=feats.env_fast,
        env_slow=feats.env_slow,
        attack=feats.attack,
        flatness=feats.flatness,
        rms_db=feats.rms_db,
        clip_frame=feats.clip_frame if feats.clip_frame is not None
        else np.zeros(feats.rms_db.shape[0], dtype=np.float32),
        scalars=np.array([float(getattr(feats, k, 0.0)) for k in CACHE_SCALARS],
                         dtype=np.float64),
        cache_version=np.array([CACHE_VERSION], dtype=np.int32),
        config=np.array([json.dumps(cfg.as_dict())], dtype=object),
    )
    return p


def load_feature_cache(source_path: Path | str, site: str):
    """Load a cached feature set, or None if absent/stale/incompatible."""
    from .features import FileFeatures

    p = cache_path_for(source_path, site)
    if not p.exists():
        return None
    try:
        with np.load(p, allow_pickle=True) as z:
            if "scalars" not in z.files:
                return None                      # pre-scalars layout
            ver = int(z["cache_version"][0]) if "cache_version" in z.files else 1
            if ver < CACHE_VERSION:
                return None                      # produced by an older analysis
            sc = z["scalars"]
            values = {k: float(v) for k, v in zip(CACHE_SCALARS, sc)}
            return FileFeatures(
                novelty=z["novelty"],
                env_fast=z["env_fast"],
                env_slow=z["env_slow"],
                attack=z["attack"],
                flatness=z["flatness"],
                rms_db=z["rms_db"],
                clip_frame=(z["clip_frame"] if "clip_frame" in z.files
                            else np.zeros(z["rms_db"].shape[0], dtype=np.float32)),
                **values,
            )
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        # A truncated or corrupt cache must degrade to "no cache", never crash a
        # scan: caches are written by parallel workers and a killed run can
        # leave a partial file behind.
        return None


def n_cached(site: str | None = None) -> int:
    root = CACHE_DIR / site.replace(" ", "-") if site else CACHE_DIR
    return sum(1 for _ in root.rglob("*.npz")) if root.exists() else 0
