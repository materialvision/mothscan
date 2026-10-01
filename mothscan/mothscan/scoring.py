"""The event-scoring framework: choose the best 0.5 s window in a 2-minute file.

Design
------
A single statistic is easy to fool.  Pure spectral novelty loves a gust of wind
that shifts the whole spectrum at once; pure envelope peak-detection loves one
loud footstep and ignores a sustained warbler.  So the score combines four
normalised, *complementary* window statistics, each measured only against the
same recording's own statistics:

``novelty``   spectral-flux density - "how much is changing in here".  Catches
              birdsong phrases, insect stridulation, wingbeats, rain onsets.
``transient`` envelope attack - "is there a sharp onset".  Catches raindrops,
              clicks, wing claps, sudden calls.
``contrast``  log ratio of the loudest envelope in the window to its own local
              background - "does the event stand out from its surroundings".
              This is the term that rescues a quiet bird in a quiet hour while
              refusing a loud-but-constant rain drone.
``flatness``  broadband-ness of the window, with a small positive weight because
              non-tonal events (rain, rustle) are exactly what a spectral-flux
              detector under-serves.  ``w_tonality`` adds the opposite bias.

Event-centred placement
-----------------------
The window offset is a first-class, configurable parameter, and it is an
**absolute** pre-roll (``AnalysisConfig.event_lead_s``, default ~11 ms) rather
than a fraction of the analysed window.  That distinction is what makes the
extract length a free render-time choice: the event sits just inside the start of
the excerpt whether it is later cut to 0.1 s or to 2 s, so short excerpts still
catch the attack.  Scoring and placement happen in one search rather than being
bolted together afterwards, and the reported ``start_s`` is directly usable as a
cut point.

Normalisation
-------------
Terms are normalised on the *windowed* distribution of the file, not on raw
frames.  A 120 s recording is mostly empty frames, so a per-frame percentile sits
in the noise and every window ratio explodes; the windowed high percentile makes
a term of 1.0 mean "about as eventful as the busiest part of this hour", which is
the comparable scale the hybrid score needs.

Search
------
Two passes: a coarse sweep at 25 ms over the whole file (integral images make the
cost independent of window length), then a 5 ms refinement around the winner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import AnalysisConfig
from .features import FileFeatures, moving_average
from .noise import ContaminationConfig

EPS = 1e-9

#: Saturation point for a normalised term.  Clipping keeps one wildly outlying
#: hour from making its absolute score dwarf every other file, which matters when
#: the preview stream is ranked or thresholded.
TERM_CLIP = 3.0

#: Terms that get file-local normalisation before being weighted.
NORMALISED_TERMS = ("novelty", "transient", "contrast", "flatness", "tonality")


# --------------------------------------------------------------------------
# Sliding-window primitives
# --------------------------------------------------------------------------


def _sliding_sum_1d(x: np.ndarray, width: int) -> np.ndarray:
    """Sum over every ``width``-sample window on a stride-1 grid.

    ``r[i] = sum(x[i : i + width])``, length ``len(x) - width + 1``.
    """
    if width <= 0:
        return np.zeros(len(x), dtype=np.float64)
    if width > len(x):
        return np.array([x.sum()], dtype=np.float64)
    csum = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    return csum[width:] - csum[:-width]


def _sliding_max_1d(x: np.ndarray, width: int) -> np.ndarray:
    """Max over every ``width``-sample window on a stride-1 grid."""
    n = len(x)
    if width <= 0:
        return np.zeros(n, dtype=np.float32)
    if width > n:
        return np.array([x.max()], dtype=np.float32)
    return np.lib.stride_tricks.sliding_window_view(x, width).max(axis=1).astype(
        np.float32, copy=False
    )


def _robust_scale(x: np.ndarray, pct: float = 95.0) -> float:
    """A high percentile as the normaliser, guarded against all-zero input."""
    if x.size == 0:
        return 1.0
    scale = float(np.percentile(x, pct))
    if not np.isfinite(scale) or scale <= EPS:
        scale = float(x.max())
    return scale if scale > EPS else 1.0


def _take(comps: dict, idx: int) -> dict:
    """Extract candidate ``idx`` from a per-candidate component mapping.

    Array values are indexed; bookkeeping scalars (keys starting with ``_``) and
    scalar ``passes`` flags are passed through untouched.
    """
    out: dict = {}
    for k, v in comps.items():
        if k.startswith("_"):
            continue
        arr = np.asarray(v)
        out[k] = arr.ravel()[idx] if arr.ndim >= 1 else arr
    return out


# --------------------------------------------------------------------------
# Result record
# --------------------------------------------------------------------------


@dataclass
class WindowScore:
    """One candidate moment: where to cut, how good it is, and why."""

    start_frame: int
    start_s: float
    score: float
    score_norm: float         # score / best in file -> selected point == 1.0
    event_frame: int          # frame of the event the placement was centred on
    event_s: float
    novelty: float
    transient: float
    contrast: float
    flatness: float
    rms_db: float
    peak_db: float
    passes_gate: bool
    terms: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict:
        d = {
            "start_frame": self.start_frame,
            "start_s": round(self.start_s, 6),
            "event_s": round(self.event_s, 6),
            "score": round(self.score, 6),
            "score_norm": round(self.score_norm, 6),
            "novelty": round(self.novelty, 6),
            "transient": round(self.transient, 6),
            "contrast": round(self.contrast, 6),
            "flatness": round(self.flatness, 6),
            "rms_db": round(self.rms_db, 3),
            "peak_db": round(self.peak_db, 3),
            "passes_gate": bool(self.passes_gate),
        }
        d.update({f"term_{k}": round(v, 6) for k, v in self.terms.items()})
        return d


# --------------------------------------------------------------------------
# The scorer
# --------------------------------------------------------------------------


class EventScorer:
    """Pre-computes per-component curves for one file, then answers queries."""

    def __init__(
        self,
        feat: FileFeatures,
        cfg: AnalysisConfig,
        noise_cfg: ContaminationConfig | None = None,
    ):
        self.f = feat
        self.cfg = cfg
        self.noise_cfg = noise_cfg or ContaminationConfig()
        self.frame_rate = feat.frame_rate
        self.n = feat.n_frames()

        # ---- component 1: novelty density --------------------------------
        # A second smoothing at the event time-scale turns frame-level flux into
        # a density, so a window mean behaves like "is an event happening here".
        event_smooth = max(
            1, int(round(cfg.novelty_event_smooth_ms / 1000.0 * feat.frame_rate))
        )
        self.novelty = moving_average(
            feat.novelty.astype(np.float64), event_smooth
        ).astype(np.float32)

        # ---- component 2: transient attack (window peak) -----------------
        self.transient = feat.attack.astype(np.float32)

        # ---- component 3: contrast (window peak) -------------------------
        self.contrast = np.log10(
            np.maximum(feat.env_fast / np.maximum(feat.env_slow, 1e-7), 1.0)
        ).astype(np.float32)

        # ---- component 4/5: flatness and its complement ------------------
        self.flatness = feat.flatness.astype(np.float32)
        self.tonality = (1.0 - self.flatness).astype(np.float32)

        self._env_db = feat.rms_db

        # ---- sampling-defect indicator (clipping at the converter rail) ----
        # Localised by nature: a window containing rail frames is a defect,
        # while the rest of the same recording may be perfectly clean.
        cf = getattr(feat, "clip_frame", None)
        if cf is None:
            self.clip_frame = np.zeros(self.n, dtype=np.float32)
        else:
            self.clip_frame = np.asarray(cf, dtype=np.float32)[: self.n]
            if self.clip_frame.size < self.n:
                self.clip_frame = np.pad(
                    self.clip_frame, (0, self.n - self.clip_frame.size)
                )

        # Gates.
        self.noise_floor_db = feat.noise_floor_db
        self.floor_gate_db = self.noise_floor_db + cfg.gate_floor_db
        self.abs_gate_db = cfg.gate_abs_db

        # Geometry.
        self.max_window = max(1, int(round(cfg.scan_seconds * self.frame_rate)))
        self._edge_guard_frames = int(round(cfg.edge_guard_s * self.frame_rate))

        # Integral images for the mean-based terms, at the maximum length.
        self._sum = {
            "novelty": _sliding_sum_1d(self.novelty, self.max_window),
            "flatness": _sliding_sum_1d(self.flatness, self.max_window),
            "tonality": _sliding_sum_1d(self.tonality, self.max_window),
        }
        self._sum_db = _sliding_sum_1d(self._env_db, self.max_window)

        self._norm_scale: dict[str, float] = {k: 1.0 for k in NORMALISED_TERMS}
        self._calibrated = False

    # -- geometry ---------------------------------------------------------

    def lead_frames(self, width: int) -> int:
        """Pre-roll before the event, in frames, for a window of ``width``.

        ``"lead"`` mode is deliberately independent of ``width``: the event must
        sit the same short distance from the start of the excerpt whether the
        later cut is 0.1 s or 2 s.  A *fraction* of the analysed window is not
        that - it is a fixed number of milliseconds set at analysis time, which
        is why 0.35 x 0.5 s = 176 ms silently swallowed the event in anything
        shorter than ~0.18 s.
        """
        cfg = self.cfg
        if getattr(cfg, "event_lead_mode", "lead") == "fraction":
            return int(round(cfg.event_offset * width))
        lead = int(round(cfg.event_lead_s * self.frame_rate))
        # Keep the window from collapsing onto its own pre-roll.
        keep = int(round(cfg.min_scan_after_lead_s * self.frame_rate))
        return int(max(0, min(lead, width - keep if width > keep else 0)))

    def place(self, event_frame: int, width: int) -> tuple[int, int]:
        """Convert an event frame into a clamped window ``(start, event)``.

        The window is clamped to the file, so an event near the very start or end
        shifts the window rather than being lost; the returned event frame is then
        the *real* position, which is what the export records.
        """
        lead = self.lead_frames(width)
        start = int(event_frame) - lead
        start = max(0, min(start, max(0, self.n - width)))
        event = int(event_frame)
        return start, max(start, min(event, start + width - 1))

    @property
    def window_frames(self) -> int:
        return self.max_window

    # -- phase 1: raw windowed statistics ---------------------------------

    def raw_terms(self, starts: np.ndarray, width: int) -> dict[str, np.ndarray]:
        """Un-normalised window statistics for a batch of window starts."""
        starts = np.asarray(starts, dtype=np.int64)
        width = int(width)

        if width == self.max_window:
            novelty_mean = self._sum["novelty"][starts] / width
            flat_mean = self._sum["flatness"][starts] / width
            ton_mean = self._sum["tonality"][starts] / width
            rms_mean = self._sum_db[starts] / width
        else:
            novelty_mean = _sliding_sum_1d(self.novelty, width)[starts] / width
            flat_mean = _sliding_sum_1d(self.flatness, width)[starts] / width
            ton_mean = _sliding_sum_1d(self.tonality, width)[starts] / width
            rms_mean = _sliding_sum_1d(self._env_db, width)[starts] / width

        windows_t = np.lib.stride_tricks.sliding_window_view(self.transient, width)
        windows_c = np.lib.stride_tricks.sliding_window_view(self.contrast, width)
        transient_peak = windows_t[starts].max(axis=1).astype(np.float64)
        contrast_peak = windows_c[starts].max(axis=1).astype(np.float64)
        del windows_t, windows_c

        rms_peak = _sliding_max_1d(self._env_db, width)[starts].astype(np.float64)

        return {
            "novelty": novelty_mean,
            "transient": transient_peak,
            "contrast": contrast_peak,
            "flatness": flat_mean,
            "tonality": ton_mean,
            "rms_mean": rms_mean,
            "rms_peak": rms_peak,
        }

    # -- phase 2: calibration and combination -----------------------------

    def calibrate(self, raw: dict[str, np.ndarray], pct: float = 99.0) -> None:
        """Derive this file's term normalisers from its windowed distributions."""
        for k in NORMALISED_TERMS:
            self._norm_scale[k] = _robust_scale(raw[k], pct)
        self._calibrated = True

    @property
    def norm_scale(self) -> dict[str, float]:
        return dict(self._norm_scale)

    def combine(self, starts: np.ndarray, raw: dict, width: int) -> tuple[np.ndarray, dict]:
        """Apply normalisation, weights, gates and the edge penalty."""
        cfg = self.cfg
        norm = {
            k: np.minimum(np.asarray(raw[k], dtype=np.float64) / self._norm_scale[k],
                          TERM_CLIP)
            for k in NORMALISED_TERMS
        }

        score = (
            cfg.w_novelty * norm["novelty"]
            + cfg.w_transient * norm["transient"]
            + cfg.w_contrast * norm["contrast"]
            + cfg.w_flatness * norm["flatness"]
            + cfg.w_tonality * norm["tonality"]
        )

        rms_peak = np.asarray(raw["rms_peak"], dtype=np.float64)
        passes = (rms_peak >= self.floor_gate_db) & (rms_peak >= self.abs_gate_db)
        if not cfg.allow_silent_fallback:
            score = np.where(passes, score, -np.inf)

        # Clipping penalty.  A window whose samples sit on the converter rail
        # is a *defect* - the excerpt would be a buzz, not a recording of the
        # environment - so it is attenuated hard rather than softly.
        clip_windows = None
        if self.clip_frame.any():
            rail = (self.clip_frame >= self.noise_cfg.clip_frame_frac).astype(np.float32)
            clip_windows = _sliding_max_1d(rail, width)[np.asarray(starts)] > 0
            if clip_windows.any():
                score = np.where(
                    clip_windows,
                    score * self.noise_cfg.clip_window_penalty,
                    score,
                )

        # Edge penalty: handling noise, wind bursts and the recorder's own
        # power-up transient cluster at the very start and end of a deployment
        # file, so a window near an edge needs to be better to win.
        guard = self._edge_guard_frames
        if guard > 0 and self.n > 2 * guard:
            centre = np.asarray(starts) + width / 2.0
            dist = np.clip(np.minimum(centre, self.n - centre) / guard, 0.0, 1.0)
            score = score * (cfg.edge_penalty + (1.0 - cfg.edge_penalty) * dist)

        comps = {
            **{k: v for k, v in norm.items()},
            "rms_mean": np.asarray(raw["rms_mean"], dtype=np.float64),
            "rms_peak": rms_peak,
            "passes": np.asarray(passes),
        }
        if clip_windows is not None:
            comps["clipped"] = np.asarray(clip_windows)
        return np.asarray(score, dtype=np.float64), comps

    def evaluate(self, event_frames: np.ndarray, width: int | None = None):
        """Score events, returning (starts, components) with placement applied."""
        width = int(width or self.max_window)
        width = min(width, self.n)
        events = np.asarray(event_frames, dtype=np.int64)
        if events.size == 0:
            events = np.array([0], dtype=np.int64)
        events = np.clip(events, 0, max(0, self.n - 1))
        starts = np.array([self.place(int(e), width)[0] for e in events], dtype=np.int64)

        raw = self.raw_terms(starts, width)
        if not self._calibrated:
            self.calibrate(raw)
        score, comps = self.combine(starts, raw, width)
        out = {
            **comps,
            **{f"raw_{k}": v for k, v in raw.items()},
            "score": score,
            "event": events,
        }
        # Stash geometry under a private key so the value dict stays purely
        # per-candidate (everything else is indexed by candidate).
        out["_width"] = width  # type: ignore[assignment]
        return starts, out

    # -- public search ----------------------------------------------------

    def candidate_events(self, step_frames: int, width: int) -> np.ndarray:
        """Event positions to consider: every frame where a window can be placed.

        Because placement shifts the window, the set of *events* is simply the
        set of frames; the granularity of the sweep is ``step_frames``.
        """
        lead = self.lead_frames(width)
        lo = lead
        hi = self.n - width + lead
        if hi < lo:
            return np.array([self.n // 2], dtype=np.int64)
        events = np.arange(lo, hi + 1, max(1, step_frames), dtype=np.int64)
        if events[-1] != hi:
            events = np.append(events, hi)
        return events

    def find_best(
        self, scan_start_s: float | None = None, n_alternatives: int = 5
    ) -> tuple[WindowScore, list[WindowScore]]:
        """Return (best window, alternatives) for this recording.

        ``score_norm`` divides by this file's own best raw score, so the selected
        point is always 1.0 and any other file or alternative is comparable on a
        0..1 scale.
        """
        cfg = self.cfg
        width = self.max_window

        if scan_start_s is not None:
            # The operator pinned an exact cut point; score just that one.
            start = max(0, min(int(round(scan_start_s * self.frame_rate)),
                               max(0, self.n - width)))
            event = start + self.lead_frames(width)
            starts, comps = self.evaluate(np.array([event]), width)
            comps = _take(comps, 0)
            comps["event"] = event
            best_raw = float(comps["score"])
            best = self._make_window(int(starts[0]), int(event), comps,
                                     max(best_raw, EPS))
            return best, []

        # ---- coarse sweep over the whole file ---------------------------
        step = max(1, int(round(cfg.coarse_step_s * self.frame_rate)))
        events = self.candidate_events(step, width)
        starts, comps = self.evaluate(events, width)
        raw_score = comps["score"]
        order = np.argsort(-raw_score)
        best_idx = int(order[0])
        coarse_best_raw = float(raw_score[best_idx])

        # ---- refine at 5 ms around the coarse winner --------------------
        span = max(1, int(round(cfg.refine_span_s * self.frame_rate)))
        r_step = max(1, int(round(cfg.refine_step_s * self.frame_rate)))
        centre = int(events[best_idx])
        r_events = np.arange(max(0, centre - span), centre + span + 1, r_step,
                             dtype=np.int64)
        r_starts, r_comps = self.evaluate(r_events, width)
        r_best = int(np.argmax(r_comps["score"]))
        best_start = int(r_starts[r_best])
        best_event = int(r_comps["event"][r_best])
        best_comps = _take(r_comps, r_best)

        norm_div = max(coarse_best_raw, float(best_comps["score"]), EPS)
        best = self._make_window(best_start, best_event, best_comps, norm_div)

        # ---- alternatives: distinct, non-adjacent moments ---------------
        exclusion = max(width, int(round(cfg.alt_exclusion_s * self.frame_rate)))
        taken = [best_start]
        alternatives: list[WindowScore] = []
        for idx in order:
            if len(alternatives) >= n_alternatives:
                break
            s = int(starts[idx])
            if any(abs(s - t) < exclusion for t in taken):
                continue
            taken.append(s)
            alternatives.append(
                self._make_window(
                    s, int(comps["event"][idx]), _take(comps, idx), norm_div
                )
            )
        return best, alternatives

    def _make_window(
        self, start: int, event: int, comps: dict, norm_div: float
    ) -> WindowScore:
        score = float(comps["score"])
        return WindowScore(
            start_frame=int(start),
            start_s=float(start / self.frame_rate),
            score=score,
            score_norm=score / norm_div if norm_div > EPS else 0.0,
            event_frame=int(event),
            event_s=float(event / self.frame_rate),
            novelty=float(comps["novelty"]),
            transient=float(comps["transient"]),
            contrast=float(comps["contrast"]),
            flatness=float(comps["flatness"]),
            rms_db=float(comps["raw_rms_mean"]),
            peak_db=float(comps["raw_rms_peak"]),
            passes_gate=bool(comps["passes"]),
            terms={
                "novelty": float(comps["novelty"]),
                "transient": float(comps["transient"]),
                "contrast": float(comps["contrast"]),
                "flatness": float(comps["flatness"]),
                "tonality": float(comps["tonality"]),
            },
        )

    def score_series(self, scan_seconds: float | None = None):
        """Coarse score curve for diagnostics: (times_of_event, scores)."""
        width = min(
            int(round(scan_seconds * self.frame_rate)) if scan_seconds else self.max_window,
            self.n,
        )
        step = max(1, int(round(self.cfg.coarse_step_s * self.frame_rate)))
        events = self.candidate_events(step, width)
        _, comps = self.evaluate(events, width)
        return events / self.frame_rate, comps["score"]


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------


def explain(scorer: EventScorer, ws: WindowScore) -> str:
    """Human-readable justification for a pick - used in the CLI report."""
    top = sorted(ws.terms.items(), key=lambda kv: -kv[1])[:2]
    reasons = ", ".join(f"{k}={v:.2f}" for k, v in top)
    gate = "ok" if ws.passes_gate else "BELOW-GATE"
    return (
        f"cut={ws.start_s:7.3f}s event={ws.event_s:7.3f}s score={ws.score:6.3f} "
        f"peak={ws.peak_db:6.1f}dBFS [{gate}] driven by {reasons}"
    )
