"""Central configuration for mothscan.

Audio facts for this dataset (verified from CONFIG.TXT + ffprobe):
  AudioMoth "Basic" firmware, 48000 Hz, 16-bit PCM, MONO,
  120 s recording every hour (sleep 3480 s), 24 h active period, gain Medium-High.
Filenames are local-recorder timestamps: YYYYMMDD_HHMMSS.WAV
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .noise import ContaminationConfig, SubstitutionConfig

# --------------------------------------------------------------------------
# Dataset layout
# --------------------------------------------------------------------------

#: Root that holds one sub-directory per recording site.
#:
#: Override with ``MOTHSCAN_SITE_ROOT`` for your own deployment.  The default is
#: the drive layout this was developed against, and is only a default: nothing
#: else in the codebase assumes it.
SITE_ROOT = Path(os.environ.get("MOTHSCAN_SITE_ROOT", "/Volumes/Solardisk/Harpefoss"))

#: Site directory names, in the order they should be presented.
#:
#: Override with ``MOTHSCAN_SITES`` (comma-separated).  These are the three
#: AudioMoth locations in the Harpefoss deployment; a site may also be split
#: into ``YYYYMMDD/`` sub-directories, which is walked automatically.
SITE_DIRS = tuple(
    s.strip()
    for s in os.environ.get(
        "MOTHSCAN_SITES", "moth oculus,moth-juvet,moth-kvarvet"
    ).split(",")
    if s.strip()
)

#: Suffixes we must ignore.  `.reapeaks` are REAPER waveform-cache files that
#: appear next to any WAV that was dragged into REAPER; they carry no audio.
DISCARD_SUFFIXES = (".reapeaks", ".reapeak", ".peak", ".pk")

#: Audio files we do want.
AUDIO_SUFFIXES = (".wav",)

SAMPLE_RATE = 48_000
FILE_SECONDS = 120.0

# --------------------------------------------------------------------------
# Analysis parameters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisConfig:
    """Everything that controls *where* inside a 2-minute file we cut."""

    # --- STFT / mel representation ---
    n_fft: int = 1024
    hop: int = 256                     # 5.333 ms frame hop -> fine transient detail
    win_length: int = 1024
    n_mels: int = 48
    fmin: float = 40.0
    fmax: float = 20_000.0
    mel_power: float = 2.0
    log_eps: float = 1e-10

    # --- Novelty (spectral flux) ---
    #: Frames averaged when estimating the per-mel-band noise floor.
    novelty_smooth_frames: int = 9
    #: Second smoothing at the event time-scale (ms); turns flux into a density.
    novelty_event_smooth_ms: float = 60.0
    #: Fraction of the *positive* flux distribution treated as "noise".
    novelty_noise_pct: float = 75.0
    #: Percentile of the noise reference subtracted from flux before rectifying.
    novelty_noise_scale: float = 1.0

    # --- Envelope / transient ---
    #: Envelope derived by summing mel energy over these bands (Hz).
    env_band_hz: tuple[float, float] = (200.0, 16_000.0)
    #: Short envelope window for transient detection (ms).
    env_fast_ms: float = 10.0
    #: Long envelope window for the local background (ms).
    env_slow_ms: float = 400.0

    # --- Scoring weights (the "hybrid" model) ---
    w_novelty: float = 1.0             # spectrally eventful
    w_transient: float = 1.0           # sharp envelope attack
    w_contrast: float = 0.6            # event stands out from its own background
    w_flatness: float = 0.15           # broadband / non-tonal bonus (rain, clicks)
    w_tonality: float = 0.0            # set >0 to favour tonal (birdsong) content

    # --- Gates / penalties ---
    #: Window must exceed the file's own noise floor by this many dB.
    gate_floor_db: float = 15.0
    #: ...and exceed absolute quiet by this many dBFS.
    gate_abs_db: float = -62.0
    #: Frames at each end of the file we refuse to cut into (seconds).
    edge_guard_s: float = 1.0
    #: Penalty applied as score *= edge_factor ** normalised_edge_distance.
    edge_penalty: float = 0.5
    #: If the best window still fails the gates, keep it but flag low confidence.
    allow_silent_fallback: bool = True

    # --- Frame-integration windows ---
    #: The candidate length we score for (seconds).  Must be <= clip_seconds.
    scan_seconds: float = 0.5
    #: Hop between candidate events in the coarse pass (seconds).
    coarse_step_s: float = 0.025
    #: Hop in the refinement pass (seconds).
    refine_step_s: float = 0.005
    #: Half-width of the refinement neighbourhood (seconds).
    refine_span_s: float = 0.15
    #: Where the detected event sits inside the analysed window.
    #:
    #: Two ways to express it, and the distinction matters:
    #:
    #: ``"lead"`` (default) - an **absolute** pre-roll in seconds
    #:     (:attr:`event_lead_s`).  This is the robust choice, because the event
    #:     then sits the same distance from the excerpt start no matter how long
    #:     the excerpt is later cut.  A fraction of the *analysis* window, by
    #:     contrast, is a fixed number of milliseconds (0.35 x 0.5 s = 176 ms),
    #:     which leaves only 24 ms of headroom in a 0.2 s excerpt and loses the
    #:     event entirely below ~0.18 s.
    #: ``"fraction"`` - the old behaviour, ``event_offset`` of the window.
    event_lead_mode: str = "lead"
    #: Pre-roll before the detected event, in seconds, for ``event_lead_mode="lead"``.
    #: Small on purpose: long enough that an attack is not truncated by the
    #: excerpt's own fade-in and reads as an onset, short enough to survive very
    #: short excerpts. Snapped to the analysis hop, so it is quantised to ~5.3 ms.
    event_lead_s: float = 0.010667
    #: Fractional placement for ``event_lead_mode="fraction"``.
    event_offset: float = 0.35
    #: Never let the analysed window collapse onto the lead: the window must stay
    #: meaningfully longer than the pre-roll, or the score measures almost nothing.
    min_scan_after_lead_s: float = 0.05
    #: Minimum separation between the chosen point and any alternative (s).
    alt_exclusion_s: float = 0.75

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def frames_per_scan(self) -> int:
        return max(1, int(round(self.scan_seconds * SAMPLE_RATE / self.hop)))


@dataclass
class PreviewConfig:
    """Controls the concatenated preview stream."""

    #: Length of each excerpt in the stream (seconds).
    cut_seconds: float = 0.5
    #: Silence inserted *between* excerpts to form a pulse train (seconds).
    #:
    #: May be **negative**, which makes consecutive excerpts overlap: the next
    #: excerpt starts while the previous one is still sounding. The overlap is
    #: ``-gap_seconds``, so ``gap = -0.2`` with a 0.5 s cut means each excerpt
    #: starts 0.3 s after the last and 0.2 s of every pair plays together.
    #:
    #: The slot pitch is ``cut + gap`` and must stay positive - it is the fixed
    #: spacing that keeps the three sites sample-aligned, so it cannot be allowed
    #: to reach zero or invert. The tightest overlap is therefore
    #: ``gap > -cut``: at 0.5 s cut, up to a full-length overlap.
    gap_seconds: float = 0.05
    #: Fade applied to each excerpt edge to avoid clicks (seconds).
    #:
    #: When excerpts overlap this *is* the crossfade: the outgoing excerpt fades
    #: over the incoming one. Raise it for a softer, more continuous blend.
    #: Capped at ``cut_seconds / 2`` so the two fades can never meet and squash
    #: the middle of an excerpt to nothing.
    fade_seconds: float = 0.004
    #: Level handling for overlapping material.
    #:
    #: ``"auto"`` (default) - scale by ``1/sqrt(overlap)``, so overlapping the
    #:     clips keeps the overall loudness roughly constant instead of piling
    #:     up. No effect when the excerpts do not overlap.
    #: ``"unity"`` - sum at full level; overlaps simply add, so a dense passage
    #:     gets louder. The limiter still guarantees no clipping.
    #: a number - apply that fixed gain.
    overlap_scale: "str | float" = "auto"
    #: Playback-speed factor applied to excerpts (1.0 = real time).
    speed: float = 1.0
    #: How a render-time ``speed`` is realised in the *rendered file*.
    #:
    #: ``"varispeed"`` (default) - resampling. Speed and pitch move together
    #:     (tape behaviour): exactly transparent, no artefacts, and the pitch
    #:     shift is usually welcome when skimming (a 2x scan of birdsong sounds
    #:     like faster birds, which is what you expect to hear).
    #: ``"pitch"`` - WSOLA time stretch. The excerpt gets shorter while the
    #:     pitch stays put. Useful for study (slowing a fast bird down to hear
    #:     its structure), but overlap-add on field recordings is materially
    #:     harder than on speech and it *does* cost some transparency; prefer
    #:     the viewer's playback speed, which uses the browser's own
    #:     pitch-preserving stretch and is both better and live.
    #:
    #: The length of a slot never changes with speed: the grid pitch (cut + gap)
    #: is the sync mechanism between sites, so speeding up leaves more silence
    #: in each slot rather than moving anything in time.
    speed_mode: str = "varispeed"
    #: Peak-normalise each excerpt before concatenation.
    normalise: bool = False
    #: Target peak when normalising (dBFS).
    normalise_dbfs: float = -3.0
    #: Sample rate of the rendered preview.
    out_sample_rate: int = SAMPLE_RATE
    #: Pre-roll used when re-placing an excerpt so a shortened cut still contains
    #: its event.  Should match ``AnalysisConfig.event_lead_s``.
    event_lead_s: float = 0.010667

    # --- loudness levelling -------------------------------------------
    #: Push peaks down so a passing train does not drown out the birds.
    #:
    #: This is a look-ahead peak limiter, not a compressor with makeup gain: it
    #: only ever *reduces* gain and never lifts quiet material, so the relative
    #: level of quiet events is preserved and the scan stays honest.  Set
    #: ``limiter_ceiling_dbfs`` higher (e.g. -1.0) to switch it off in effect.
    limiter: bool = True
    #: Peak ceiling: nothing in the rendered preview exceeds this.
    limiter_ceiling_dbfs: float = -12.0
    #: How far above the ceiling the limiter starts acting, in dB.  A soft knee
    #: keeps quiet material completely untouched.
    limiter_knee_db: float = 6.0
    #: Look-ahead window, seconds.  Long enough to catch a transient before it
    #: passes, short enough not to pump.  5 ms is transparent for this material.
    limiter_lookahead_s: float = 0.005
    #: Release time constant, seconds.  Longer = smoother but more ducking.
    limiter_release_s: float = 0.12


@dataclass
class RunConfig:
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)
    sites: tuple[str, ...] = SITE_DIRS
    workers: int = max(1, (os.cpu_count() or 4) - 1)
    #: Wind / clipping thresholds.  Imported lazily to keep config.py free of
    #: cycles, since noise.py imports the timeline.
    contamination: "ContaminationConfig | None" = None
    substitution: "SubstitutionConfig | None" = None


# --------------------------------------------------------------------------
# On-disk locations
# --------------------------------------------------------------------------
#
# Everything the tool writes lives under one directory.  By default that is
# ``mothwork/`` - the parent of this package - so a fresh clone works with no
# configuration.  Override with ``MOTHSCAN_DATA_DIR`` to put the (potentially
# multi-gigabyte) caches and renders on a different volume.

PROJECT_DIR = Path(
    os.environ.get("MOTHSCAN_PROJECT_DIR", Path(__file__).resolve().parents[2])
)
DATA_DIR = Path(os.environ.get("MOTHSCAN_DATA_DIR", PROJECT_DIR / "data"))
CACHE_DIR = DATA_DIR / "cache"          # per-file .npz analysis caches
PREVIEW_DIR = DATA_DIR / "previews"      # rendered preview wavs + sidecars
EXPORT_DIR = DATA_DIR / "exports"        # max/msp + csv exports
INDEX_DB = DATA_DIR / "mothscan.sqlite3"
INDEX_JSONL = DATA_DIR / "points.jsonl"

for _d in (DATA_DIR, CACHE_DIR, PREVIEW_DIR, EXPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def is_audio_file(path: Path) -> bool:
    """True only for real audio; explicitly rejects REAPER peak caches."""
    name = path.name
    lower = name.lower()
    if any(lower.endswith(s) for s in DISCARD_SUFFIXES):
        return False
    if name.startswith("."):
        return False
    return lower.endswith(AUDIO_SUFFIXES)
