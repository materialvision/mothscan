"""Wind-noise and microphone-clipping detection, and what to do about it.

Two problems to solve
---------------------
1. **Do not pick a contaminated moment.**  A gust of wind scores *extremely*
   well on the hybrid event score - it is spectrally novel, has a strong
   envelope, and stands far above the local background - so without an explicit
   veto the picker will happily choose wind over a bird.
2. **If an hour is nothing but wind, play a nearby hour instead.**  Silently
   emitting 0.5 s of roar would misrepresent the deployment; the honest options
   are to substitute the nearest acceptable hour or to mark the slot, and the
   operator gets to choose.

Why these indicators
--------------------
Wind and clipping are *physical* signatures, so they can be measured directly
rather than inferred:

``ulf_ratio``   fraction of energy below 40 Hz.  This deployment left the
                AudioMoth's optional 48 Hz DC-blocking filter **disabled**, so
                the band is intact and wind is unmistakable: median across the
                deployment is ~1%, while wind hours exceed 5%.
``tilt_db_oct`` long-term spectral slope.  Wind collapses with frequency
                (steeper than -4 dB/oct); rain and insect chorus are flatter.
``sustain``     fraction of frames within 6 dB of the loudest frame.  Wind holds
                one level for minutes; birdsong and rain are eventful.
``crest``       peak/RMS.  A clipped signal has almost none.
``clip_runs``   number of contiguous runs of >= 8 identical PCM samples.  Smooth
                audio never produces these; a rail does.  Quantisation noise
                makes runs of 2-3, hence the >= 8 threshold.

Classification is deliberately *conjunctive*: the indicators must agree.  Wind
is only claimed when the low band is both relatively strong **and** the spectrum
is steeply tilted, which prevents a genuine low-frequency animal (a bittern, a
frog chorus) from being dismissed as weather.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Protocol

from .timeline import MissingPolicy


class _HasIndicators(Protocol):
    ulf_ratio: float
    ulf_rel_db: float
    tilt_db_oct: float
    sustain: float
    crest: float
    clip_run: int
    clip_runs: int
    clip_frac: float
    peak_db: float
    noise_floor_db: float


@dataclass(frozen=True)
class ContaminationConfig:
    """Thresholds for calling something wind or clipped.

    Every default below was placed on a measured gap in this deployment's own
    indicator distributions (``scripts/calibrate_noise.py`` prints them; the
    galleries from ``scripts/validate_noise.py`` show the outcome), not chosen a
    priori:

    ``low100``   sub-100 Hz share is strongly bimodal - p50 0.04, p75 0.10, then
                 a jump to p90 0.41.  0.30 sits in that empty middle.
    ``peak_db``  p90 is already -0.66 dBFS and 463 files (8.3 %) sit exactly at
                 the 16-bit rail.  0 dBFS *is* clipping, so -0.5 is used to
                 allow for intersample headroom.
    ``clip_run`` every file at the rail has a run >= 12 identical samples; no
                 file *below* -1 dBFS has one.  So the test has no false
                 positives here and is safe to use on its own.
    """

    # ---- wind --------------------------------------------------------
    #: Share of energy below 100 Hz.
    wind_low100: float = 0.30
    #: Spectral slope, dB per octave over 50 Hz - 4 kHz.  Wind rumbles; a
    #: narrowband low-frequency animal does not tilt the whole spectrum.
    wind_tilt_db_oct: float = -4.0
    #: Peak level below which the file is simply quiet, not windy.
    wind_min_peak_db: float = -25.0
    #: How much of the file sits near its own maximum level (reported, and used
    #: to strengthen the verdict; not required on its own).
    wind_sustain: float = 0.30

    # ---- clipping ----------------------------------------------------
    #: Peak level at or above which the converter was at its rail.
    clip_peak_db: float = -0.5
    #: Longest run of identical PCM samples that still looks natural.
    clip_min_run: int = 12
    #: Many short runs at the rail are also systematic clipping.
    clip_min_runs: int = 5
    #: Crest factor below which the waveform is squashed rather than peaky.
    clip_crest: float = 3.0

    # ---- scoring -----------------------------------------------------
    #: Score at or above which choosing an excerpt here is refused outright.
    #: Between 0 and this, the score is attenuated (see `penalty`).
    exclude_score: float = 1.0
    #: Score at which the penalty starts.
    soft_score: float = 0.5
    #: Maximum score attenuation applied to a mildly contaminated window.
    max_penalty: float = 0.6

    # ---- per-window clipping -----------------------------------------
    #: Fraction of a frame's samples that must be at the rail to call that
    #: frame clipped.  Clipping is localised, so this penalty applies to the
    #: windows that contain the rail rather than to the whole file.
    clip_frame_frac: float = 0.02
    #: Score multiplier for a window containing clipped frames.  Harsh, because
    #: a clipped excerpt is a defect rather than a judgement call.
    clip_window_penalty: float = 0.05

    # ---- substitution -------------------------------------------------
    #: A recording counts as unusable as a *source* when wind dominates it or
    #: when clipping is sustained across the file (not one clipped sample).
    substitute_clip_frac: float = 0.02

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class Contamination:
    """Per-recording contamination verdict.

    Three distinct facts, because they call for three different responses:

    ``wind``      the whole hour is dominated by wind rumble.  Nothing useful
                  can be cut from it, so the hour should be substituted.
    ``clipped``   *some* sample touched the converter rail.  Informational on
                  its own - the defect may be a few milliseconds inside an
                  otherwise excellent hour, so only the affected windows are
                  penalised.
    ``sustained`` clipping is spread across the recording, which makes the
                  whole hour a poor source.  Wind counts as sustained by
                  definition, since it is not localised to any window.
    """

    score: float = 0.0
    wind: bool = False
    clipped: bool = False
    sustained: bool = False
    clip_frac: float = 0.0

    @property
    def unusable(self) -> bool:
        """True when this recording should not be used as a source at all."""
        return bool(self.wind or self.sustained)

    @property
    def clean(self) -> bool:
        return self.score < 1e-9

    def as_row(self) -> dict:
        return {
            "noise_score": round(self.score, 4),
            "noise_wind": int(self.wind),
            "noise_clipped": int(self.clipped),
            "noise_sustained": int(self.sustained),
        }


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 below ``lo``, 1 above ``hi``, linear in between."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


class _Attrs:
    """Attribute view over a mapping, so `classify` works for both a
    FileFeatures object and a calibration row read back from disk.

    Missing keys resolve to 0 rather than raising, which keeps the classifier
    usable on partial data (an older calibration file, for instance).
    """

    __slots__ = ("_d",)

    def __init__(self, d):
        self._d = d

    def __getattr__(self, item):
        try:
            v = self._d[item]
        except (KeyError, TypeError):
            return 0.0
        return 0.0 if v is None else v


def indicators_row(d) -> "_Attrs":
    """Adapt a mapping (or pass through an object) to the indicator interface."""
    if isinstance(d, dict) or hasattr(d, "keys"):
        return _Attrs(d)
    return d


def classify(indicators, cfg: ContaminationConfig | None = None) -> Contamination:
    """Turn the measured indicators into a wind/clipping verdict and a score.

    The score is a soft 0..~2 measure, not a probability: it is used to (a) veto
    windows above ``exclude_score`` and (b) attenuate the rest, so it needs to be
    monotonic and comparable across files rather than calibrated to a likelihood.
    """
    cfg = cfg or ContaminationConfig()

    tilt = float(getattr(indicators, "tilt_db_oct", 0.0) or 0.0)
    sustain = float(getattr(indicators, "sustain", 0.0) or 0.0)
    crest = float(getattr(indicators, "crest", 0.0) or 0.0)
    clip_run = int(getattr(indicators, "clip_run", 0) or 0)
    clip_runs = int(getattr(indicators, "clip_runs", 0) or 0)
    clip_frac = float(getattr(indicators, "clip_frac", 0.0) or 0.0)
    peak_db = float(getattr(indicators, "peak_db", -120.0) or -120.0)

    # Narrow and wide low-band shares.  `low100` is the primary discriminant;
    # `ulf_ratio` (sub-40 Hz) is accepted as an alternative for callers that
    # only have the older indicator.
    low100 = float(getattr(indicators, "low100", 0.0) or 0.0)
    if low100 <= 0.0:
        low100 = float(getattr(indicators, "ulf_ratio", 0.0) or 0.0)

    # ---- clipping ----------------------------------------------------
    # Being at the converter rail *is* clipping; a long run of identical samples
    # corroborates it.  Measured on this deployment, every file at 0 dBFS had a
    # run >= 12 and no file below -1 dBFS did, so this test has no false
    # positives here.
    at_rail = peak_db >= cfg.clip_peak_db
    rail = clip_run >= cfg.clip_min_run
    systematic = clip_runs >= cfg.clip_min_runs and crest <= cfg.clip_crest
    clipped = bool(at_rail or rail or systematic)

    # How *spread* is the clipping?  A single pinned wingbeat leaves a tiny
    # fraction of samples at the rail; a squashed recording leaves much more.
    # `clip_frac` (fraction of samples inside long runs) and the run count
    # together separate the two.
    spread = max(
        _ramp(clip_frac, cfg.substitute_clip_frac * 0.5, cfg.substitute_clip_frac * 4.0),
        _ramp(clip_runs, 200.0, 2000.0),
    )
    sustained = bool(clipped and spread >= 0.5)
    clip_score = max(
        _ramp(clip_frac, cfg.substitute_clip_frac * 0.25, cfg.substitute_clip_frac * 3.0),
        _ramp(clip_runs, 50.0, 800.0),
        0.65 if at_rail else 0.0,
        _ramp(clip_run, cfg.clip_min_run * 0.75, cfg.clip_min_run * 4.0),
        0.6 if systematic else 0.0,
    )

    # ---- wind --------------------------------------------------------
    # Conjunctive, so a genuine low-frequency animal is not dismissed as
    # weather: the sub-100 Hz band has to dominate *and* the whole spectrum has
    # to tilt steeply.  Sustained level and low crest strengthen the verdict
    # rather than gate it.
    low_evidence = _ramp(low100, cfg.wind_low100 * 0.67, cfg.wind_low100 * 2.0)
    tilt_evidence = _ramp(-tilt, -cfg.wind_tilt_db_oct * 0.6, -cfg.wind_tilt_db_oct * 1.8)
    loud_enough = peak_db >= cfg.wind_min_peak_db
    wind_base = min(low_evidence, tilt_evidence)
    wind = bool(wind_base >= 0.5 and loud_enough)
    continuity = 0.6 + 0.4 * _ramp(sustain, 0.0, cfg.wind_sustain)
    wind_score = wind_base * continuity if loud_enough else wind_base * 0.5

    return Contamination(
        score=float(max(wind_score, clip_score)),
        wind=wind,
        clipped=clipped,
        sustained=sustained,
        clip_frac=clip_frac,
    )


def penalty(score: float, cfg: ContaminationConfig | None = None) -> float:
    """Multiplier applied to a window's event score given its contamination.

    1.0 for clean material, falling to ``1 - max_penalty`` as the score
    approaches ``exclude_score``.  A veto (returning 0) is applied separately by
    the caller so that "excluded" is a distinct, reportable outcome rather than
    a very small number.
    """
    cfg = cfg or ContaminationConfig()
    if score <= cfg.soft_score:
        return 1.0
    t = _ramp(score, cfg.soft_score, cfg.exclude_score)
    return float(1.0 - cfg.max_penalty * t)


def is_excluded(score: float, cfg: ContaminationConfig | None = None) -> bool:
    cfg = cfg or ContaminationConfig()
    return score >= cfg.exclude_score


# --------------------------------------------------------------------------
# Nearby-hour substitution
# --------------------------------------------------------------------------


@dataclass
class SubstitutionConfig:
    """How to fill an hour whose own excerpt is unusable.

    ``SEARCH_BACK``  use the nearest earlier hour that is acceptable
    ``SEARCH_FORWARD`` use the nearest later hour
    ``SEARCH_BOTH``  nearest in either direction
    ``SILENCE``      emit silence and mark the slot
    """

    mode: str = "both"                 # both | back | forward | silence
    #: How far to look, in hours, before giving up.
    max_distance_h: int = 6
    #: A substitute must be at least this much cleaner than the original,
    #: otherwise it is not an improvement worth making.
    min_score_gain: float = 0.25
    #: Substitutes may not themselves be beyond this contamination score.
    max_substitute_score: float = 0.5
    #: When consecutive hours are all wind-dominated (a storm lasts hours, not
    #: minutes) they would otherwise all borrow the *same* neighbour and the
    #: preview would repeat one excerpt.  Prefer an unused neighbour, and reuse
    #: one only when nothing fresher is in range.
    avoid_reuse: bool = True


def find_substitute(
    site_slots: dict[int, object],
    grid_index: int,
    scores: dict[int, float],
    cfg: SubstitutionConfig | None = None,
    contam: ContaminationConfig | None = None,
    used: set[int] | None = None,
) -> tuple[int, object] | None:
    """Find the nearest acceptable hour to stand in for ``grid_index``.

    ``site_slots`` maps grid index -> record for one site, ``scores`` maps grid
    index -> contamination score.  ``used`` is the set of neighbours already
    borrowed by another hour; when ``cfg.avoid_reuse`` is set they are skipped.

    Candidates are ordered by distance, so the closest acceptable hour wins
    regardless of direction.  Returns ``(grid_index, record)`` or None.
    """
    cfg = cfg or SubstitutionConfig()
    contam = contam or ContaminationConfig()
    own = scores.get(grid_index, 0.0)
    used = used or set()

    candidates: list[int] = []
    for d in range(1, cfg.max_distance_h + 1):
        if cfg.mode in ("both", "back"):
            candidates.append(grid_index - d)
        if cfg.mode in ("both", "forward"):
            candidates.append(grid_index + d)
    if cfg.mode == "both":
        # Nearest hour overall, not all-backward-then-all-forward.
        candidates.sort(key=lambda g: (abs(g - grid_index), g))

    fallback: tuple[int, object] | None = None
    for g in candidates:
        rec = site_slots.get(g)
        if rec is None:
            continue
        s = scores.get(g, 0.0)
        if s > cfg.max_substitute_score:
            continue
        if own - s < cfg.min_score_gain:
            continue
        if cfg.avoid_reuse and g in used:
            if fallback is None:
                fallback = (g, rec)
            continue
        return g, rec
    return fallback
