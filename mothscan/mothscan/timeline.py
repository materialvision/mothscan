"""The absolute timeline: one slot per clock hour, shared by every site.

Why this exists
---------------
Each site is recorded independently, but all AudioMoths were configured with the
same hourly cadence (120 s every 3600 s, 00:00-24:00 UTC), so *"hour H"* is a
meaningful, comparable unit across sites.  Collapsing each site into its own
variable-length stream would destroy that: you could no longer ask "what was
happening at every site at 03:00 on 14 July".

So every selected point is assigned an absolute **slot index** on one shared
grid:

    slot = (timestamp - grid_epoch) / slot_seconds

``slot_seconds`` is ``cut + gap`` - the real duration one hour occupies in the
rendered stream.  Two sites that both have a file for hour H therefore put that
hour's excerpt at exactly the same stream offset, and playing site A's stream
next to site B's stays aligned for the whole deployment.

Missing data
------------
When a site has no file for an hour the slot is still *numbered*, and the site's
stream either

* **silence** - emit ``slot_seconds`` of silence so the stream's offsets remain
  exactly equal to the grid's (``MissingPolicy.SILENCE``, the default); or
* **compact** - skip the slot so the stream is shorter and gapless, with the
  grid index carried in the metadata for scheduling (``MissingPolicy.SKIP``).

Sites whose deployments do not overlap still share the grid; a site that only
ran for the first 805 hours simply has slots after that, which the viewer draws
as an explicit "no deployment" band rather than confusing silence.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

from .config import SITE_DIRS, SITE_ROOT


class MissingPolicy(str, Enum):
    """What to do with an hour for which a site has no recording."""

    #: Emit a silent slot, keeping stream offsets identical to the grid.
    SILENCE = "silence"
    #: Omit the slot entirely; the stream becomes shorter but gapless.
    SKIP = "skip"


# --------------------------------------------------------------------------
# Slots
# --------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class HourSlot:
    """One clock hour on the absolute grid."""

    index: int                 # slot number from the grid epoch
    dt: datetime               # UTC instant of the hour

    @property
    def label(self) -> str:
        return self.dt.strftime("%Y-%m-%d %H:%M")

    @property
    def hour_label(self) -> str:
        return self.dt.strftime("%H:%M")

    @property
    def day_label(self) -> str:
        return self.dt.strftime("%Y-%m-%d")

    def stream_offset_s(self, slot_seconds: float) -> float:
        return self.index * slot_seconds


class Timeline:
    """A shared hourly grid plus the site -> slot mapping on top of it."""

    def __init__(self, slot_seconds: float = 0.55, epoch: datetime | None = None):
        self.slot_seconds = float(slot_seconds)
        self.epoch = epoch or datetime(1970, 1, 1, tzinfo=timezone.utc)
        self._by_site: dict[str, dict[int, object]] = defaultdict(dict)

    # -- construction ----------------------------------------------------

    @staticmethod
    def slot_of(dt: datetime, epoch: datetime) -> int:
        return int((dt - epoch).total_seconds() // 3600)

    @staticmethod
    def parse_timestamp(ts: str) -> datetime | None:
        """Parse the ISO-8601 UTC stamp the pipeline stores ('...Z', seconds)."""
        if not ts:
            return None
        try:
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            dt = datetime.fromisoformat(ts)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def add_records(self, records: Iterable) -> int:
        """Attach records to their slots. Returns how many were mapped."""
        n = 0
        for r in records:
            dt = self.parse_timestamp(getattr(r, "timestamp", ""))
            if dt is None:
                continue
            idx = self.slot_of(dt, self.epoch)
            # Collapse to the top of the hour so a file named 14:47:52 still
            # occupies the 14:00 slot it was recorded in.
            self._by_site[r.site][idx] = r
            n += 1
        return n

    # -- queries ---------------------------------------------------------

    @property
    def sites(self) -> list[str]:
        return sorted(self._by_site)

    def slot_indices(self, site: str) -> list[int]:
        return sorted(self._by_site.get(site, {}))

    def record_at(self, site: str, slot: int):
        return self._by_site.get(site, {}).get(slot)

    def first_slot(self, site: str | None = None) -> int | None:
        idx = self.slot_indices(site) if site else [
            i for s in self.sites for i in self.slot_indices(s)
        ]
        return min(idx) if idx else None

    def last_slot(self, site: str | None = None) -> int | None:
        idx = self.slot_indices(site) if site else [
            i for s in self.sites for i in self.slot_indices(s)
        ]
        return max(idx) if idx else None

    def span(self, site: str | None = None) -> tuple[int, int] | None:
        lo, hi = self.first_slot(site), self.last_slot(site)
        if lo is None or hi is None:
            return None
        return lo, hi

    def dt_of(self, index: int) -> datetime:
        return self.epoch + timedelta(hours=index)

    def offset_s(self, index: int) -> float:
        return index * self.slot_seconds

    # -- gap analysis ----------------------------------------------------

    def gaps(
        self, site: str, lo: int | None = None, hi: int | None = None,
        max_report: int = 500,
    ) -> list[dict]:
        """Missing hours inside a site's own deployment window.

        Returns one entry per *contiguous* gap, so a site that was down for a
        week reports one record with ``hours: 168`` rather than 168 records.
        """
        present = set(self.slot_indices(site))
        if not present:
            return []
        lo = self.first_slot(site) if lo is None else lo
        hi = self.last_slot(site) if hi is None else hi
        if lo is None or hi is None:
            return []

        out: list[dict] = []
        run_start: int | None = None
        for i in range(lo, hi + 1):
            if i in present:
                if run_start is not None:
                    out.append(self._gap_entry(run_start, i - 1))
                    run_start = None
            elif run_start is None:
                run_start = i
        if run_start is not None:
            out.append(self._gap_entry(run_start, hi))
        return out[:max_report]

    def _gap_entry(self, start: int, end: int) -> dict:
        return {
            "start_slot": start,
            "end_slot": end,
            "hours": end - start + 1,
            "start": self.dt_of(start).isoformat().replace("+00:00", "Z"),
            "end": self.dt_of(end).isoformat().replace("+00:00", "Z"),
            "start_offset_s": round(self.offset_s(start), 4),
        }

    def coverage(self, site: str) -> dict:
        present = self.slot_indices(site)
        if not present:
            return {"slots": 0, "hours": 0}
        lo, hi = present[0], present[-1]
        window = hi - lo + 1
        hours = [self.dt_of(i).hour for i in present]
        by_hour = {h: hours.count(h) for h in range(24)}
        return {
            "slots": len(present),
            "first_slot": lo,
            "last_slot": hi,
            "first": self.dt_of(lo).isoformat().replace("+00:00", "Z"),
            "last": self.dt_of(hi).isoformat().replace("+00:00", "Z"),
            "window_hours": window,
            "missing_hours": window - len(present),
            "coverage_pct": round(100.0 * len(present) / window, 3),
            "hour_histogram": by_hour,
        }

    def summary(self) -> dict:
        lo, hi = self.first_slot(), self.last_slot()
        return {
            "slot_seconds": self.slot_seconds,
            "sites": {
                s: self.coverage(s) for s in self.sites
            },
            "union_first_slot": lo,
            "union_last_slot": hi,
            "union_hours": (hi - lo + 1) if lo is not None and hi is not None else 0,
            "union_span": (
                [
                    self.dt_of(lo).isoformat().replace("+00:00", "Z"),
                    self.dt_of(hi).isoformat().replace("+00:00", "Z"),
                ]
                if lo is not None and hi is not None
                else None
            ),
        }


# --------------------------------------------------------------------------
# Stream planning
# --------------------------------------------------------------------------


@dataclass
class PlannedSlot:
    """One slot in a site's rendered stream."""

    ordinal: int              # position in this stream (0-based)
    grid_index: int           # absolute slot on the shared timeline
    has_audio: bool
    record: object | None = None
    stream_start_s: float = 0.0
    stream_start_sample: int = 0

    def as_dict(self) -> dict:
        r = self.record
        return {
            "o": self.ordinal,
            "grid": self.grid_index,
            "audio": self.has_audio,
            "stream_s": round(self.stream_start_s, 6),
            "stream_sample": self.stream_start_sample,
            # `name` is the file the audio was actually read from, which differs
            # from the hour's own file when a neighbour was borrowed.
            "name": getattr(r, "audio_name", None) or getattr(r, "source_name", None),
            "own_name": getattr(r, "source_name", None),
            # `cut_s` pairs with `name` (the file actually read), so it is the
            # neighbour's offset when borrowed; `own_cut_s` is this hour's own.
            "cut_s": round(getattr(r, "audio_cut_s", 0.0), 6) if r else None,
            "own_cut_s": round(getattr(r, "cut_s", 0.0), 6) if r else None,
            # `event_s` pairs with `cut_s` (same file); `own_event_s` is this
            # hour's own event, which the viewer reports.
            "event_s": round(getattr(r, "audio_event_s", 0.0), 6) if r else None,
            "own_event_s": round(getattr(r, "event_s", 0.0), 6) if r else None,
            "score": round(getattr(r, "score", 0.0), 6) if r else None,
            "score_norm": round(getattr(r, "score_norm", 0.0), 6) if r else None,
            "peak_db": round(getattr(r, "peak_db", -120.0), 3) if r else None,
            "gate": bool(getattr(r, "passes_gate", False)) if r else False,
            "ts": getattr(r, "timestamp", None),
            # Substitution provenance: set when this slot's audio was taken from
            # a nearby hour because its own hour was unusable.
            "sub": bool(getattr(r, "substituted", False)) if r else False,
            "origin": (getattr(r, "origin_name", None)
                       if getattr(r, "substituted", False) else None),
            "origin_grid": getattr(r, "origin_grid", None) if r else None,
            "noise": (round(float(getattr(r, "noise_score", 0.0)), 3)
                      if r else None),
            "noise_wind": bool(getattr(r, "noise_wind", False)) if r else False,
            "noise_sustained": bool(getattr(r, "noise_sustained", False)) if r else False,
        }


@dataclass
class SiteStream:
    """The render plan for one site."""

    site: str
    slots: list[PlannedSlot]
    sample_rate: int
    slot_seconds: float
    policy: MissingPolicy
    n_audio_slots: int = 0
    n_silent_slots: int = 0
    #: Grid index of this stream's slot 0.  With a shared origin this is
    #: identical for every site, which is what makes stream offsets directly
    #: comparable between sites.
    grid_origin: int = 0
    bound_lo: int = 0
    bound_hi: int = -1
    #: Multiplier applied to every stream position. 1.0 is real time; a `speed`
    #: render sets it to `speed`, so the whole stream is time-compressed and the
    #: grid spacing shrinks to match the (also compressed) excerpts.  Without
    #: this the excerpts shortened but the spacing did not, leaving silence -
    #: see the note on `time_scale` in `plan_site_stream`.
    time_scale: float = 1.0

    #: Length of one excerpt in samples.  When excerpts overlap they run past
    #: their own slot, so the buffer has to allow for the overhang - see
    #: :attr:`total_samples`.
    cut_samples: int = 0

    @property
    def total_samples(self) -> int:
        """Samples needed to hold every excerpt, including any overhang.

        Each excerpt spans ``[i*pitch, i*pitch + cut]``, so the last one ends at
        ``(n-1)*pitch + cut``.  With ``gap >= 0`` this is >= ``cut`` and reduces
        to the old ``n*pitch`` behaviour only when the clip exactly fills its
        slot; with a negative gap (overlap) the overhang is real and would
        otherwise be truncated.
        """
        if not self.slots:
            return 0
        pitch = int(round(self.slot_seconds * self.sample_rate))
        cut = self.cut_samples or pitch
        return self.slots[-1].stream_start_sample + cut

    @property
    def duration_s(self) -> float:
        return self.total_samples / self.sample_rate

    def slot_of_stream_sample(self, sample: int) -> PlannedSlot | None:
        """Which slot a stream position falls in (for click-to-source)."""
        if not self.slots:
            return None
        per = self.slot_seconds * self.sample_rate
        i = min(len(self.slots) - 1, max(0, int(sample // per)))
        return self.slots[i]

    def as_dict(self) -> dict:
        return {
            "site": self.site,
            "sample_rate": self.sample_rate,
            "slot_seconds": self.slot_seconds,
            "policy": self.policy.value,
            "grid_origin": self.grid_origin,
            "bound_lo": self.bound_lo,
            "bound_hi": self.bound_hi,
            "n_slots": len(self.slots),
            "n_audio_slots": self.n_audio_slots,
            "n_silent_slots": self.n_silent_slots,
            "duration_s": self.duration_s,
        }


def plan_site_stream(
    timeline: Timeline,
    site: str,
    sample_rate: int,
    policy: MissingPolicy = MissingPolicy.SILENCE,
    lo: int | None = None,
    hi: int | None = None,
    align: bool = True,
    time_scale: float = 1.0,
) -> SiteStream:
    """Work out which slots a site's stream contains, and where they sit.

    ``align=True`` uses the timeline's *union* bounds, so every site's stream
    starts at the same grid index and slot ``i`` covers the same wall-clock hour
    for all of them.  Site A's stream offset for hour H then equals site B's,
    which is what lets the streams be compared or played side by side.  A site
    whose deployment ended early (``moth-kvarvet`` stops ~1,571 h before the
    others) simply has silent slots for the remainder.

    ``align=False`` uses the site's own first/last slot, giving the shortest
    stream that still contains all of its data.

    ``lo``/``hi`` override both.  With ``policy=SILENCE`` gaps inside the range
    become silent slots; with ``SKIP`` they are dropped entirely.
    """
    present = timeline.slot_indices(site)
    if not present:
        return SiteStream(site, [], sample_rate, timeline.slot_seconds, policy)

    site_lo, site_hi = present[0], present[-1]
    if lo is None or hi is None:
        if align:
            union = timeline.span() or (site_lo, site_hi)
            lo = union[0] if lo is None else lo
            hi = union[1] if hi is None else hi
        else:
            lo = site_lo if lo is None else lo
            hi = site_hi if hi is None else hi
    if hi < lo:
        hi = lo

    present_set = set(present)
    if policy is MissingPolicy.SILENCE:
        indices = list(range(lo, hi + 1))
    else:
        indices = [i for i in range(lo, hi + 1) if i in present_set]

    # `time_scale` compresses the *stream* without touching the grid: the slot
    # pitch becomes `slot_seconds / time_scale`.  It exists so that a render-time
    # `speed` produces a genuinely faster stream. Compressing only the excerpt
    # audio (which is all an earlier version did) leaves the spacing untouched,
    # so every slot becomes a short burst followed by a long silence - audibly a
    # click train rather than a sped-up scan.
    pitch_s = timeline.slot_seconds / max(time_scale, 1e-9)

    slots: list[PlannedSlot] = []
    for ordinal, grid_index in enumerate(indices):
        start_s = ordinal * pitch_s
        slots.append(
            PlannedSlot(
                ordinal=ordinal,
                grid_index=grid_index,
                has_audio=grid_index in present_set,
                record=timeline.record_at(site, grid_index),
                stream_start_s=start_s,
                stream_start_sample=int(round(start_s * sample_rate)),
            )
        )
    return SiteStream(
        site=site,
        slots=slots,
        sample_rate=sample_rate,
        slot_seconds=pitch_s,
        policy=policy,
        n_audio_slots=sum(1 for s in slots if s.has_audio),
        n_silent_slots=sum(1 for s in slots if not s.has_audio),
        grid_origin=lo,
        bound_lo=lo,
        bound_hi=hi,
        time_scale=time_scale,
    )


def build_timeline(
    records: Sequence,
    slot_seconds: float = 0.55,
    sites: Iterable[str] | None = None,
) -> Timeline:
    """Convenience: build a Timeline from index records."""
    tl = Timeline(slot_seconds=slot_seconds)
    tl.add_records(sorted(records, key=lambda r: (r.site, r.timestamp)))
    if sites:
        for s in sites:
            tl._by_site.setdefault(s, {})
    return tl
