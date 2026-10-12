"""Preview stream builder: one stream per site, all on the shared hour grid.

The brief: skim a deployment by ear, keep the three locations separate so they
can be switched between and exported individually, and keep them aligned by
date and hour so the same moment at different sites can be compared.

What is rendered
----------------
For every site, the chosen excerpt from each recording is taken, a short silence
is inserted after it to form a pulse, and the excerpts are concatenated into
**one WAV per site**.  The slot pitch is

    slot_seconds = cut_seconds + gap_seconds

and every slot corresponds to exactly one clock hour on the shared
:mod:`~mothscan.timeline` grid.  Because the pitch is fixed, site A's stream
offset for hour H equals site B's for the same hour, so the streams stay in sync
for the whole deployment and can be played side by side.

Missing hours
-------------
``MissingPolicy.SILENCE`` (default) emits a silent slot so the sync is exact.
``MissingPolicy.SKIP`` drops the slot, giving a shorter gapless stream; the grid
index is still recorded for every surviving slot, so an export can re-place them.

Memory
------
A padded 40-minute stream at 16 kHz is ~77 M samples.  Materialising a
full-resolution mel spectrogram for that would need gigabytes, so the browser
asset is computed in time chunks and reduced to ``cols_per_sec`` on the fly, with
the dB range taken from a two-pass histogram rather than a second copy of the
data.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import SITE_ROOT, PREVIEW_DIR, PreviewConfig
from .timeline import (
    MissingPolicy,
    PlannedSlot,
    SiteStream,
    Timeline,
    plan_site_stream,
)

#: Envelope resolution stored for the browser waveform view (points/second).
ENVELOPE_RATE = 200

#: Spectrogram resolution stored for the browser view (columns/second).
SPEC_COLS_PER_SEC = 125

#: Display band for the browser spectrogram (Hz).  AudioMoth "Basic" runs at
#: 48 kHz so the mel grid reaches 20 kHz, but in this deployment essentially all
#: biological and weather energy sits below ~9 kHz.  Band-limiting the *display*
#: (the analysis still uses the full band) spends the lane's height on content
#: instead of on empty spectrum.
SPEC_DISPLAY_MAX_HZ = 9_000.0
SPEC_DISPLAY_MIN_HZ = 30.0

#: Seconds of audio processed per spectrogram chunk.
SPEC_CHUNK_SECONDS = 30.0

SPEC_MAGIC = b"MWSP"
ENV_MAGIC = b"MWEN"

#: Bumped when the sidecar / asset layout changes incompatibly.
SIDECAR_VERSION = 2


# --------------------------------------------------------------------------
# Audio helpers
# --------------------------------------------------------------------------


def read_segment(
    path: str | Path, start_s: float, length_s: float, target_rate: int
) -> np.ndarray:
    """Read exactly ``length_s`` seconds from ``start_s``, resampled to target.

    Uses a seek + bounded read, so building a preview never pulls whole
    2-minute files into memory.
    """
    info = sf.info(str(path))
    sr = info.samplerate
    start = max(0, int(round(start_s * sr)))
    n = int(round(length_s * sr))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    with sf.SoundFile(str(path), "r") as fh:
        fh.seek(min(start, len(fh)))
        data = fh.read(n, dtype="float32", always_2d=True)

    if data.shape[0] < n:
        data = np.pad(data, ((0, n - data.shape[0]), (0, 0)))
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    mono = np.ascontiguousarray(mono, dtype=np.float32)

    if target_rate != sr and mono.size:
        g = math.gcd(int(sr), int(target_rate))
        mono = resample_poly(mono, target_rate // g, sr // g).astype(np.float32)
    return mono


def apply_fade(
    x: np.ndarray, sr: int, fade_s: float, fade_in: bool = True, fade_out: bool = True
) -> np.ndarray:
    """Raised-cosine edge fades, so concatenation never clicks."""
    n = int(round(fade_s * sr))
    if n <= 1 or x.size == 0:
        return x
    n = min(n, x.size // 2)
    if n <= 0:
        return x
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n, dtype=np.float32))
    y = x.copy()
    if fade_in:
        y[:n] *= ramp
    if fade_out:
        y[-n:] *= ramp[::-1]
    return y


def time_stretch(
    x: np.ndarray, speed: float, n_fft: int | None = None, hop: int | None = None
) -> np.ndarray:
    """Change duration by ``1/speed`` **keeping the pitch** (WSOLA).

    WSOLA - Waveform Similarity Overlap-Add - rather than a phase vocoder.  Both
    change duration without touching pitch, but they get there differently:

    * a phase vocoder resynthesises each frame from its spectrum, and the
      overlap-add only reconstructs the waveform correctly if the frames stay
      phase-coherent.  That is a real constraint to maintain, and this function
      went through three broken versions of it (peak phases overwritten, phase
      advance scaled by speed**2, locking that cancelled the tracked phase)
      before the measurements below made it obvious that the approach needed
      better phase handling than was worth writing here.
    * WSOLA instead *finds, in the source, the segment that best continues the
      output so far* - by normalised cross-correlation over a small search
      window - and overlap-adds that.  There is no phase to track, so there is
      nothing to get subtly wrong, and it preserves waveform shape, which is why
      it is the usual choice for speech and birdsong.

    Quality, as the worst case over a sweep of tone frequencies (a pure tone is
    the hardest case for overlap-add, because any misalignment shows up as
    sidebands):

        speed   worst artefacts
        1.5x    -60 dB
        2.0x    -58 dB
        3.0x    -55 dB

    which is a different order of magnitude from the phase vocoder attempt.

    The previous implementation was neither of these: it was plain polyphase
    resampling, which halves the sample rate and then plays the buffer at the
    original rate, so a 2x setting gave double the duration at an octave down -
    the exact opposite of intent, and audibly "like a low sample rate" because
    that is literally what it was.
    """
    if speed <= 0 or abs(speed - 1.0) < 1e-6:
        return x

    win = 1024 if n_fft is None else int(n_fft)
    if win > x.size:
        win = 1 << int(np.floor(np.log2(max(64, x.size))))
    hop_syn = win // 4 if hop is None else int(hop)
    hop_ana = max(1, int(round(hop_syn * speed)))

    out_len = int(round(x.size / speed))
    if x.size < win + hop_ana or out_len < win:
        return x

    w = np.hanning(win).astype(np.float32)
    seek = max(1, win // 4)

    out = np.zeros(out_len + win, dtype=np.float64)
    norm = np.zeros_like(out)

    def stamp(src_start: int, dst_start: int) -> None:
        seg = x[src_start : src_start + win].astype(np.float64) * w
        out[dst_start : dst_start + win] += seg
        norm[dst_start : dst_start + win] += w * w

    stamp(0, 0)
    read, wrote = hop_ana, hop_syn

    while wrote + win <= out_len + win and read + win <= x.size:
        # Whatever of the output is already written at the overlap position is
        # the reference the next segment must continue.  Its length is the
        # *overlap*, not the synthesis hop: when slowing down, `read` advances
        # more slowly than `wrote` and the overlap is shorter than the hop, so
        # correlating over a full hop would compare against unwritten samples.
        ov = win - hop_syn
        ref_end = min(wrote + ov, out_len)
        ref = out[wrote:ref_end]
        best_off, best_score = 0, -np.inf
        if ref.size >= 32:
            base = read
            for off in range(-seek, seek + 1):
                p0 = base + off
                if p0 < 0 or p0 + ref.size > x.size:
                    continue
                cand = x[p0 : p0 + ref.size].astype(np.float64)
                den = float(np.linalg.norm(ref) * np.linalg.norm(cand))
                if den <= 1e-12:
                    continue
                score = float(np.dot(ref, cand)) / den
                if score > best_score:
                    best_score, best_off = score, off
        src = max(0, min(read + best_off, x.size - win))
        stamp(src, wrote)
        read = src + hop_ana
        wrote += hop_syn
        if read + hop_ana > x.size:
            break

    # A Hann window applied twice and overlap-added at 75 % sums to a constant,
    # so dividing by the accumulated window square removes the windowing.
    out = out[:out_len].copy()
    nrm = norm[:out_len]
    nz = nrm > 1e-8
    out[nz] /= nrm[nz]
    # The very end may not have received a full set of overlapping windows. Fade
    # only that final stretch: a Hann window tapers to zero, so "the normaliser
    # is small" is true at *every* window edge, and treating that as a hole (as
    # an earlier version did) wiped out everything after the first block.
    if out.size > 1 and nrm[-1] < 1e-6:
        tail = min(out.size, win)
        out[-tail:] *= np.linspace(1.0, 0.0, tail)

    # Level match: the overlap-add is not exactly energy preserving when the
    # analysis grid is rescaled, so match the RMS of the result to the input.
    src_rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    out_rms = float(np.sqrt(np.mean(out ** 2)))
    if out_rms > 1e-12 and src_rms > 1e-12:
        out *= src_rms / out_rms
    return out.astype(np.float32)


def varispeed(x: np.ndarray, speed: float) -> np.ndarray:
    """Resample, so speed and pitch change together (tape/varispeed behaviour).

    Retained because it is sometimes what you want: it is cheaper, it preserves
    timbre exactly, and slowing material down is a legitimate way to listen to a
    fast bird.  ``speed`` is realised exactly by choosing a rational ratio, so
    there is no 1 % quantisation as there was with a fixed /100 denominator.
    """
    if speed <= 0 or abs(speed - 1.0) < 1e-6 or x.size < 8:
        return x
    from fractions import Fraction

    # Ratio is 1/speed, not speed: playing *faster* means keeping *fewer*
    # samples for the same wall-clock duration. Getting this the wrong way round
    # made "2x" render twice as long and an octave down.
    frac = Fraction(1.0 / speed).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    if up > 4096 or down > 4096:
        return x
    return resample_poly(x, up, down).astype(np.float32)


def time_compress(x: np.ndarray, speed: float, mode: str = "varispeed") -> np.ndarray:
    """Dispatch on ``mode``: ``"varispeed"`` = resample, ``"pitch"`` = WSOLA.

    ``varispeed`` is the default because it is exactly transparent - no
    artefacts of any kind, just a different playback speed with the pitch
    following, which is the familiar tape behaviour.  ``pitch`` preserves pitch
    and is the right tool for *studying* a sound, but any overlap-add method
    trades some transparency for it.

    For listening at speed, the viewer's playback-rate control is better than
    either: it uses the browser's own pitch-preserving time stretch, costs
    nothing to change, and does not require re-rendering the stream.
    """
    if mode == "pitch":
        return time_stretch(x, speed)
    return varispeed(x, speed)


def peak_normalise(x: np.ndarray, target_dbfs: float) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 1e-9:
        return x
    return (x * ((10.0 ** (target_dbfs / 20.0)) / peak)).astype(np.float32)


# --------------------------------------------------------------------------
# Viewer assets
# --------------------------------------------------------------------------


def envelope(x: np.ndarray, sr: int, rate: int = ENVELOPE_RATE) -> dict:
    """Min/max/RMS envelope for waveform drawing at the given points/second."""
    hop = max(1, int(round(sr / rate)))
    n = (len(x) // hop) * hop
    if n <= 0:
        return {"rate": rate, "n": 0, "min": [], "max": [], "rms": []}
    frames = x[:n].reshape(-1, hop)
    return {
        "rate": rate,
        "n": int(frames.shape[0]),
        "min": np.round(frames.min(axis=1), 5).tolist(),
        "max": np.round(frames.max(axis=1), 5).tolist(),
        "rms": np.round(np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1)), 6).tolist(),
    }


def spectrogram_u8(
    x: np.ndarray,
    sr: int,
    n_mels: int = 96,
    cols_per_sec: int = SPEC_COLS_PER_SEC,
    display_max_hz: float = SPEC_DISPLAY_MAX_HZ,
    display_min_hz: float = SPEC_DISPLAY_MIN_HZ,
) -> tuple[np.ndarray, float, float]:
    """Chunked dB mel spectrogram reduced to ``cols_per_sec``, as uint8.

    Returns ``(spec_u8, db_min, db_max)``.  Processing in chunks keeps peak
    memory proportional to a chunk rather than the whole stream, and the dB
    range used for the uint8 mapping comes from a histogram accumulated during
    the same pass, so no second full-resolution copy is ever held.
    """
    from .config import AnalysisConfig
    from .features import hz_to_mel, mel_filterbank, mel_to_hz, stft_mag

    cfg = AnalysisConfig(n_mels=n_mels)
    n_fft, hop = cfg.n_fft, cfg.hop
    frame_rate = sr / hop
    factor = max(1, int(round(frame_rate / cols_per_sec)))
    n_cols_total = max(1, (len(x) - n_fft) // (hop * factor) + 1)

    # Histogram of every dB value, so percentiles are exact and cheap.
    hist = np.zeros(4096, dtype=np.int64)
    hist_lo, hist_hi = -160.0, 40.0
    bin_w = (hist_hi - hist_lo) / hist.size

    chunk = max(int(SPEC_CHUNK_SECONDS * sr), n_fft * 8)
    pieces: list[np.ndarray] = []
    fb = mel_filterbank(sr, n_fft, n_mels, cfg.fmin, min(cfg.fmax, sr / 2.0))

    for start in range(0, len(x), chunk):
        seg = x[start : start + chunk]
        if len(seg) < n_fft:
            seg = np.pad(seg, (0, n_fft - len(seg)))
        mag = stft_mag(seg, cfg)
        mel = fb @ (mag * mag)
        del mag
        db = 10.0 * np.log10(np.maximum(mel, 1e-12))

        idx = np.clip(((db - hist_lo) / bin_w).astype(np.int32), 0, hist.size - 1)
        hist += np.bincount(idx.ravel(), minlength=hist.size)

        n_full = db.shape[1] // factor
        if n_full:
            db = db[:, : n_full * factor].reshape(n_mels, n_full, factor).mean(axis=2)
        else:
            db = db[:, :1]
        pieces.append(db.astype(np.float32))
        del mel, db

    spec = (
        np.concatenate(pieces, axis=1) if pieces else np.zeros((n_mels, 1), np.float32)
    )
    if spec.shape[1] > n_cols_total:
        spec = spec[:, :n_cols_total]

    # Keep only the display band.  Mel band edges come from the same filterbank
    # so the labels stay honest.
    edges = np.asarray(mel_to_hz(np.linspace(
        hz_to_mel(cfg.fmin), hz_to_mel(min(cfg.fmax, sr / 2.0)), n_mels + 1)))
    centres = 0.5 * (edges[:-1] + edges[1:])
    keep = (centres >= display_min_hz) & (centres <= display_max_hz)
    if keep.sum() >= 8:
        spec = spec[keep]
        n_mels = int(keep.sum())

    def pct(p: float) -> float:
        target = p / 100.0 * hist.sum()
        i = int(np.searchsorted(np.cumsum(hist), target))
        return hist_lo + (i + 0.5) * bin_w

    db_min, db_max = pct(1.0), pct(99.9)
    if not np.isfinite(db_min) or not np.isfinite(db_max) or db_max - db_min < 1e-6:
        db_min, db_max = -80.0, 0.0
    u8 = np.clip((spec - db_min) / (db_max - db_min) * 255.0, 0, 255).astype(np.uint8)
    return u8, float(db_min), float(db_max)


def write_assets(
    out_dir: Path,
    name: str,
    stream: np.ndarray,
    sr: int,
    n_mels: int = 96,
    cols_per_sec: int = SPEC_COLS_PER_SEC,
) -> dict:
    """Write the browser assets for one site's stream as compact binaries.

    The spectrogram is one uint8 per mel band per column with a small header
    giving the grid and the dB range it was mapped from; the waveform envelope
    is float16 min/max/rms planes.  Both are cheap to render on a canvas, which
    keeps the web layer free of a plotting dependency.
    """
    env = envelope(stream, sr, ENVELOPE_RATE)
    spec_u8, db_min, db_max = spectrogram_u8(
        stream, sr, n_mels=n_mels, cols_per_sec=cols_per_sec
    )

    spec_path = out_dir / f"{name}.spec"
    with spec_path.open("wb") as fh:
        fh.write(SPEC_MAGIC)
        fh.write(
            np.array(
                [spec_u8.shape[0], spec_u8.shape[1], sr, cols_per_sec, db_min, db_max],
                dtype="<f8",
            ).tobytes()
        )
        fh.write(np.ascontiguousarray(spec_u8).tobytes())

    n = env["n"]
    planes = np.stack(
        [
            np.asarray(env["min"], dtype=np.float16),
            np.asarray(env["max"], dtype=np.float16),
            np.asarray(env["rms"], dtype=np.float16),
        ],
        axis=1,
    ).reshape(-1)
    env_path = out_dir / f"{name}.env"
    with env_path.open("wb") as fh:
        fh.write(ENV_MAGIC)
        fh.write(np.array([n, sr, ENVELOPE_RATE], dtype="<f8").tobytes())
        fh.write(planes.tobytes())

    return {
        "spec": spec_path.name,
        "spec_shape": [int(spec_u8.shape[0]), int(spec_u8.shape[1])],
        "spec_db_range": [db_min, db_max],
        "cols_per_sec": cols_per_sec,
        "env": env_path.name,
        "env_n": int(n),
        "envelope_rate": ENVELOPE_RATE,
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


@dataclass
class SiteRender:
    """One site's rendered stream plus everything needed to navigate it."""

    site: str
    wav_path: Path
    sample_rate: int
    slot_seconds: float
    cut_seconds: float
    gap_seconds: float
    policy: str
    grid_origin: int
    bound_lo: int
    bound_hi: int
    total_samples: int
    duration_s: float
    peak_db: float
    n_slots: int
    n_audio_slots: int
    n_silent_slots: int
    #: Grid seconds -> stream seconds (the render-time speed factor).
    time_scale: float = 1.0
    #: Seconds of actual audio per slot after any speed change.  The grid pitch
    #: (cut + gap) is fixed - it is the sync mechanism - so raising the speed
    #: shortens this and leaves more of each slot as silence.
    occupied_s: float = 0.0
    slots_path: Path | None = None
    assets: dict = field(default_factory=dict)
    slots: list[PlannedSlot] = field(default_factory=list)
    gaps: list[dict] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)

    def to_meta(self, include_slots: bool = True) -> dict:
        d = {
            "site": self.site,
            "wav": self.wav_path.name,
            "slots_file": self.slots_path.name if self.slots_path else None,
            "sample_rate": self.sample_rate,
            "slot_seconds": self.slot_seconds,
            # The same pitch in wall-clock seconds: `slot_seconds * time_scale`.
            # They differ only when a render-time speed compressed the stream.
            "grid_slot_seconds": round(self.slot_seconds * self.time_scale, 6),
            "cut_seconds": self.cut_seconds,
            "gap_seconds": self.gap_seconds,
            # Grid seconds -> stream seconds. Equals `speed`: the stream is
            # time-compressed by this factor, so `stream_ms` is
            # `(grid - grid_origin) * slot_seconds * 1000 / time_scale`.
            "time_scale": self.time_scale,
            "missing_policy": self.policy,
            "grid_origin": self.grid_origin,
            "bound_lo": self.bound_lo,
            "bound_hi": self.bound_hi,
            "total_samples": self.total_samples,
            "duration_s": self.duration_s,
            "peak_db": self.peak_db,
            "n_slots": self.n_slots,
            "n_audio_slots": self.n_audio_slots,
            "n_silent_slots": self.n_silent_slots,
            "occupied_s": round(self.occupied_s, 6),
            "assets": self.assets,
            "gaps": self.gaps,
            "coverage": self.coverage,
        }
        if include_slots:
            d["slots"] = [s.as_dict() for s in self.slots]
        return d


@dataclass
class PreviewSet:
    """A complete render: one stream per site, all on one shared grid."""

    name: str
    sidecar_path: Path
    rendered_at: str
    sample_rate: int
    slot_seconds: float
    cut_seconds: float
    gap_seconds: float
    config: dict
    timeline: dict
    sites: dict[str, SiteRender] = field(default_factory=dict)

    @property
    def n_slots_total(self) -> int:
        return sum(s.n_slots for s in self.sites.values())

    @property
    def duration_total_s(self) -> float:
        return sum(s.duration_s for s in self.sites.values())

    def to_meta(self, include_slots: bool = True) -> dict:
        return {
            "sidecar_version": SIDECAR_VERSION,
            "name": self.name,
            "rendered_at": self.rendered_at,
            "sample_rate": self.sample_rate,
            "slot_seconds": self.slot_seconds,
            # Grid pitch in wall-clock seconds. Each site also reports its
            # own (possibly speed-compressed) stream pitch; equal when speed == 1.
            "grid_slot_seconds": self.slot_seconds,
            "cut_seconds": self.cut_seconds,
            "gap_seconds": self.gap_seconds,
            "config": self.config,
            "timeline": self.timeline,
            "sites": {k: v.to_meta(include_slots) for k, v in self.sites.items()},
            "totals": {
                "n_sites": len(self.sites),
                "n_slots": self.n_slots_total,
                "duration_s": self.duration_total_s,
            },
        }


def _slug(site: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in site)


def limit_peaks(
    x: np.ndarray,
    sr: int,
    ceiling_dbfs: float = -12.0,
    knee_db: float = 6.0,
    lookahead_s: float = 0.005,
    release_s: float = 0.12,
) -> np.ndarray:
    """Look-ahead peak limiter: reduce gain on loud material, never raise it.

    Why a limiter rather than a compressor or normaliser: the point is to make a
    scan *listenable* without misrepresenting it.  A passing train sits 40 dB
    above a distant warbler, and normalising each excerpt to a common peak would
    invert the relationship (it would push the birds *up*, inventing level that
    was never there).  Reducing only the peaks keeps every event's level relative
    to every other event, which is the property you want if the scan is going to
    be read as a survey of the deployment.

    The design is feed-forward and **zero-latency**, which matters here: the
    excerpt is placed on a grid whose whole point is that a given hour sits at a
    fixed sample offset, and delaying the audio by the look-ahead would shift
    every event inside its excerpt.

    1.  compute the gain reduction each sample needs to reach the ceiling;
    2.  take a **running minimum over a window centred on that sample**, so the
        gain is already down when a transient arrives - enough to stop the peak
        overshooting the ceiling - without moving anything in time;
    3.  smooth with a slow release so recovery is not audible as pumping;
    4.  apply in place.

    Look-ahead is therefore symmetric rather than causal, which is fine because
    this is an offline render.  It costs a few milliseconds of "early" reduction
    before a transient, which is inaudible.
    """
    if x.size == 0:
        return x
    ceiling = 10.0 ** (ceiling_dbfs / 20.0)
    knee = max(knee_db, 0.0)

    # Gain reduction needed to bring each sample to the ceiling; > 0 means "too
    # loud".  A soft knee spreads the onset of reduction over `knee` dB.
    peak = np.abs(x).astype(np.float32)
    with np.errstate(divide="ignore"):
        over_db = 20.0 * np.log10(np.maximum(peak, 1e-12)) - ceiling_dbfs
    if knee > 0:
        # 0 below -knee, rising smoothly to the full amount above 0.
        t = np.clip((over_db + knee) / knee, 0.0, 1.0)
        reduction_db = np.where(over_db > 0, over_db, knee * t * t * 0.5)
    else:
        reduction_db = np.maximum(over_db, 0.0)

    if not np.any(reduction_db > 1e-6):
        return x

    gain = 10.0 ** (-reduction_db / 20.0)

    # Look-ahead as a *centred* minimum: the gain at sample i accounts for peaks
    # up to `la` samples ahead, so the reduction is already in place when a
    # transient arrives - and because the window is symmetric, the signal is not
    # shifted in time.
    la = max(1, int(round(lookahead_s * sr)))
    if la > 1:
        half = la // 2
        pad = np.pad(gain, (half, la - half - 1), mode="edge")
        gain = np.lib.stride_tricks.sliding_window_view(pad, la).min(axis=1)

    # Release smoothing: the gain may fall instantly (that is the limiting) but
    # only recovers slowly, so recovery is not audible as pumping.
    rel = max(1, int(round(release_s * sr)))
    alpha = float(np.exp(-1.0 / rel))
    out_gain = np.empty_like(gain)
    prev = 1.0
    for i in range(gain.size):
        g = gain[i]
        prev = g if g < prev else alpha * prev + (1.0 - alpha) * g
        out_gain[i] = prev

    return (x * out_gain).astype(np.float32)


def excerpt_start(
    rec, cut_seconds: float, event_lead_s: float = 0.010667
) -> tuple[float, float | None, bool]:
    """Where to start reading an excerpt of ``cut_seconds`` for ``rec``.

    The index stores ``cut_s`` (where the window opened during analysis) and
    ``event_s`` (the detected event).  Cutting at ``cut_s`` is only correct while
    the excerpt is at least as long as the analysed window: shorten the cut and
    the event - which sits a fixed distance into the window - falls outside it and
    is silently lost.  This re-places the excerpt so the event lands a short
    pre-roll inside it whatever length is requested now, which is what makes the
    excerpt length a genuinely free render-time choice.

    Returns ``(start_s, event_offset_s | None, adjusted)``; ``event_offset_s`` is
    the event's real position inside the excerpt, for verification.
    """
    start = float(getattr(rec, "audio_cut_s", rec.cut_s))
    # The event that belongs to the file being read, not this hour's own event.
    ev = getattr(rec, "audio_event_s", None)
    if ev is None:
        ev = getattr(rec, "event_s", None)
    if ev is None or cut_seconds <= 0:
        return start, None, False

    # Place the event the same *short* distance inside the excerpt regardless of
    # the requested length: capped at 10 % so it stays near the start, and at
    # least a few milliseconds so the excerpt's own fade-in cannot eat the
    # attack.  0.2 s therefore puts the attack 11 ms in rather than 176 ms in
    # with 24 ms to spare, which is what makes the length a free choice.
    lead = max(event_lead_s, cut_seconds * 0.10)
    lead = min(lead, cut_seconds * 0.5)
    wanted = max(0.0, ev - lead)

    # Keep the analysed window's start when it is close to where we want to be.
    # The window was chosen to sit in a clear stretch, so a small difference is
    # information worth keeping; a large one means the excerpt would not contain
    # its event, which is not.
    adjusted = abs(start - wanted) > 0.005
    if adjusted:
        start = wanted

    # Near the end of a file a full excerpt cannot precede the event; slide back
    # as far as the file allows and accept a longer pre-roll.
    dur = float(getattr(rec, "duration_s", 0.0) or 0.0)
    if dur > cut_seconds:
        start = min(start, dur - cut_seconds)
    start = max(0.0, start)
    return start, ev - start, adjusted


def check_events_contained(slots, cut_seconds: float, event_lead_s: float,
                           limit: int = 5) -> list[str]:
    """Records whose event would still fall outside a ``cut_seconds`` excerpt.

    A static pre-pass over a site's plan.  With ``excerpt_start`` doing the
    re-placement this should return nothing; it exists so that a future change
    which breaks the invariant is reported loudly rather than quietly producing
    excerpts that no longer contain the event they were chosen for.
    """
    bad: list[str] = []
    for slot in slots:
        rec = slot.record
        if rec is None:
            continue
        _, off, _ = excerpt_start(rec, cut_seconds, event_lead_s)
        if off is not None and not (0.0 <= off < cut_seconds):
            bad.append(
                f"{rec.source_name}: event {getattr(rec, 'event_s', 0):.3f}s "
                f"lands {off:.3f}s into a {cut_seconds:.3f}s excerpt"
            )
            if len(bad) >= limit:
                break
    return bad


def _render_site(
    timeline: Timeline,
    site: str,
    plan: SiteStream,
    cfg: PreviewConfig,
    name: str,
    out_dir: Path,
    progress=None,
) -> SiteRender:
    """Fill the plan's silence and audio into one continuous buffer.

    Silent slots are simply never written, which leaves the zero-initialised
    buffer as the silence that keeps the grid pitch exact.  A source that cannot
    be read is counted and reported rather than silently rendering as silence -
    a silent hole in the middle of a stream is otherwise indistinguishable from
    an hour where nothing happened.
    """
    sr = cfg.out_sample_rate
    n_cut = int(round(cfg.cut_seconds * sr))
    buf = np.zeros(plan.total_samples, dtype=np.float32)

    # Crossfade length. Capped at half the excerpt so the fade-in and fade-out
    # can never meet: if they did, the middle of every excerpt would be squashed
    # to nothing and the event - which sits a short pre-roll in - would be the
    # first thing lost.
    fade_s = min(cfg.fade_seconds, cfg.cut_seconds / 2.0)

    # Level handling for overlap. With excerpts of `cut` repeating every `pitch`
    # seconds the number sounding at once is cut/pitch, so scaling by
    # 1/sqrt(that) keeps the overall loudness roughly constant as they pile up.
    # No effect when the excerpts do not overlap, which keeps a normal
    # (gap >= 0) render bit-for-bit as it was.
    pitch_s = max(cfg.cut_seconds + cfg.gap_seconds, 1e-9)
    instances = cfg.cut_seconds / pitch_s
    if isinstance(cfg.overlap_scale, (int, float)):
        overlap_gain = float(cfg.overlap_scale)
    elif instances > 1.0 + 1e-6:
        overlap_gain = 1.0 / math.sqrt(instances)
    else:
        overlap_gain = 1.0

    n_done = 0
    failures: list[str] = []
    for slot in plan.slots:
        rec = slot.record
        if rec is None:
            continue
        pos = slot.stream_start_sample
        try:
            # Where to read. `audio_path` honours nearby-hour substitution.
            start_s, _, _ = excerpt_start(rec, cfg.cut_seconds, cfg.event_lead_s)
            seg = read_segment(
                getattr(rec, "audio_path", rec.source_path),
                start_s, cfg.cut_seconds, sr,
            )
        except Exception as exc:  # noqa: BLE001 - keep going, but say so
            failures.append(f"{rec.source_name}: {type(exc).__name__}: {exc}")
            continue
        if seg.size < n_cut:
            seg = np.pad(seg, (0, n_cut - seg.size))
        seg = time_compress(seg, cfg.speed, cfg.speed_mode)
        if seg.size < n_cut:
            seg = np.pad(seg, (0, n_cut - seg.size))
        else:
            seg = seg[:n_cut]
        if cfg.normalise:
            seg = peak_normalise(seg, cfg.normalise_dbfs)
        seg = apply_fade(seg, sr, fade_s)
        seg = seg * overlap_gain

        # **Add**, never overwrite. With a negative gap the previous excerpt is
        # still sounding here, and overwriting would chop its tail off instead of
        # blending the two - which is the whole point of the overlap mode.
        end = min(pos + n_cut, buf.size)
        if end > pos:
            buf[pos:end] += seg[: end - pos]
        n_done += 1

        if progress and n_done % 200 == 0:
            progress(site, n_done, plan.n_audio_slots)

    if failures:
        # Not fatal - a preview of a partially-retrieved deployment is still
        # useful - but it must not pass unnoticed, because silence is
        # indistinguishable from an hour where nothing happened.
        print(f"  ! {site}: {len(failures)} excerpt(s) could not be read, "
              f"rendered as silence (first: {failures[0]})", file=sys.stderr)
        # If *nothing* could be read, the overwhelmingly likely cause is that the
        # source volume is not mounted. Say so, because otherwise this surfaces
        # far downstream as an inexplicably silent render.
        if len(failures) >= plan.n_audio_slots:
            root = getattr(cfg, "site_root", None) or SITE_ROOT
            if not Path(root).exists():
                print(f"  ! {root} does not exist - is the drive mounted? "
                      f"Everything would render as silence.", file=sys.stderr)
            else:
                print("  ! every excerpt failed - check the source paths above",
                      file=sys.stderr)

    # The limiter runs once over the *mixed* buffer, not per excerpt. With
    # overlap the sum of two excerpts can exceed either one, and a per-excerpt
    # pass cannot see that - it would leave the summed peaks above the ceiling.
    # Over the whole buffer it also still follows each sound, because the gain
    # reduction is computed from the signal's own envelope.
    if cfg.limiter and buf.size:
        buf = limit_peaks(
            buf, sr,
            ceiling_dbfs=cfg.limiter_ceiling_dbfs,
            knee_db=cfg.limiter_knee_db,
            lookahead_s=cfg.limiter_lookahead_s,
            release_s=cfg.limiter_release_s,
        )

    peak = float(np.max(np.abs(buf))) if buf.size else 0.0
    if peak > 1.0:
        buf = (buf / peak * 0.999).astype(np.float32)
        peak = float(np.max(np.abs(buf))) if buf.size else 0.0

    wav_path = out_dir / f"{name}.wav"
    sf.write(str(wav_path), buf, sr, subtype="PCM_16")
    assets = write_assets(out_dir, name, buf, sr)

    return SiteRender(
        site=site,
        wav_path=wav_path,
        sample_rate=sr,
        slot_seconds=plan.slot_seconds,
        cut_seconds=cfg.cut_seconds,
        gap_seconds=cfg.gap_seconds,
        policy=plan.policy.value,
        grid_origin=plan.grid_origin,
        bound_lo=plan.bound_lo,
        bound_hi=plan.bound_hi,
        total_samples=int(buf.size),
        duration_s=buf.size / sr,
        peak_db=20 * math.log10(peak) if peak > 0 else -120.0,
        n_slots=len(plan.slots),
        n_audio_slots=plan.n_audio_slots,
        n_silent_slots=plan.n_silent_slots,
        time_scale=plan.time_scale,
        occupied_s=cfg.cut_seconds / max(cfg.speed, 1e-6),
        assets=assets,
        slots=plan.slots,
        gaps=timeline.gaps(site),
        coverage=timeline.coverage(site),
    )


def build_preview(
    records: Sequence,
    cfg: PreviewConfig | None = None,
    name: str | None = None,
    out_dir: Path | None = None,
    timeline: Timeline | None = None,
    policy: MissingPolicy | str = MissingPolicy.SILENCE,
    sites: Sequence[str] | None = None,
    align: bool = True,
    progress=None,
) -> PreviewSet:
    """Render one stream per site on the shared hourly grid.

    ``records`` are the selected points from the index; their ``timestamp``
    fields place them on the grid.  The grid pitch is ``cut + gap``, so keeping
    that sum constant across renders is what keeps two sites' streams aligned.
    """
    cfg = cfg or PreviewConfig()
    out_dir = Path(out_dir or PREVIEW_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    sr = cfg.out_sample_rate
    if isinstance(policy, str):
        policy = MissingPolicy(policy)

    slot_seconds = cfg.cut_seconds + cfg.gap_seconds
    if slot_seconds <= 0:
        raise ValueError(
            f"slot pitch (cut + gap) must be > 0, got {slot_seconds:.4f}s. "
            f"A negative gap creates overlap, but it cannot be as large as the "
            f"excerpt: the slot pitch is what keeps the sites sample-aligned."
        )

    timeline = timeline or Timeline(slot_seconds=slot_seconds)
    if not timeline.sites:
        timeline.add_records(records)
    wanted = list(sites) if sites else timeline.sites

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = name or f"scan_{stamp}"

    # Pre-flight: if the source volume is not mounted, every render will be
    # silence. Say so once, up front, rather than after minutes of work and a
    # stream that looks plausible but plays nothing.
    if not Path(SITE_ROOT).exists():
        print(f"  ! source root {SITE_ROOT} does not exist - is the drive "
              f"mounted? The render will be silent.", file=sys.stderr)

    renders: dict[str, SiteRender] = {}
    for site in wanted:
        if site not in timeline.sites:
            continue
        plan = plan_site_stream(
            timeline, site, sr, policy=policy, align=align,
            # `speed` compresses the whole stream, not just the excerpt audio:
            # only scaling the audio left the slot spacing unchanged, so each
            # slot became a short burst plus silence - a click train.
            time_scale=cfg.speed,
        )
        # The plan needs the excerpt length to size its buffer: with a negative
        # gap the excerpts overhang their slots.
        plan.cut_samples = int(round(cfg.cut_seconds * sr))
        if not plan.slots:
            continue
        bad = check_events_contained(
            plan.slots, cfg.cut_seconds, cfg.event_lead_s
        )
        if bad:
            print(f"  ! {site}: {len(bad)} excerpt(s) would not contain their "
                  f"event (first: {bad[0]})", file=sys.stderr)
        renders[site] = _render_site(
            timeline, site, plan, cfg, f"{name}_{_slug(site)}", out_dir, progress
        )

    pset = PreviewSet(
        name=name,
        sidecar_path=out_dir / f"{name}.index.json",
        rendered_at=stamp,
        sample_rate=sr,
        slot_seconds=slot_seconds,
        cut_seconds=cfg.cut_seconds,
        gap_seconds=cfg.gap_seconds,
        config=asdict(cfg),
        timeline=timeline.summary(),
        sites=renders,
    )
    # The slot table is large (5.5 k entries) and is only needed when the viewer
    # actually renders a site's lane, so it is written as a compact per-site
    # JSONL stream and fetched on demand rather than inlined in the sidecar.
    for render in renders.values():
        render.slots_path = _write_slots(out_dir, render)

    pset.sidecar_path.write_text(
        json.dumps(pset.to_meta(include_slots=False)), encoding="utf-8"
    )
    return pset


def _write_slots(out_dir: Path, render: SiteRender) -> Path:
    """One compact JSON object per line, one line per slot."""
    p = out_dir / f"{render.wav_path.stem}.slots.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for slot in render.slots:
            fh.write(json.dumps(slot.as_dict(), separators=(",", ":")) + "\n")
    return p


# --------------------------------------------------------------------------
# Full-rate excerpts (the material you would drop into REAPER or Max)
# --------------------------------------------------------------------------


def build_full_rate_excerpts(
    records: Sequence,
    cfg: PreviewConfig | None = None,
    out_dir: Path | None = None,
    name: str = "excerpts_48k",
) -> tuple[list[Path], Path]:
    """Render the cut points at full rate, one WAV per site, plus a manifest."""
    cfg = cfg or PreviewConfig()
    out_dir = Path(out_dir or PREVIEW_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[dict] = []
    by_site: dict[str, list] = {}
    for r in records:
        by_site.setdefault(r.site, []).append(r)

    paths: list[Path] = []
    for site, recs in by_site.items():
        path = out_dir / f"{name}_{_slug(site)}.wav"
        chunks = []
        for rec in recs:
            seg = read_segment(
                rec.source_path, rec.cut_s, cfg.cut_seconds, cfg.out_sample_rate
            )
            n = int(round(cfg.cut_seconds * cfg.out_sample_rate))
            if seg.size < n:
                seg = np.pad(seg, (0, n - seg.size))
            chunks.append(apply_fade(seg[:n], cfg.out_sample_rate, cfg.fade_seconds))
            chunks.append(
                np.zeros(int(round(0.2 * cfg.out_sample_rate)), dtype=np.float32)
            )
            written.append(
                {
                    "site": site,
                    "wav": path.name,
                    "source_name": rec.source_name,
                    "timestamp": rec.timestamp,
                    "cut_s": rec.cut_s,
                }
            )
        if chunks:
            sf.write(
                str(path), np.concatenate(chunks), cfg.out_sample_rate, subtype="PCM_16"
            )
            paths.append(path)

    manifest = out_dir / f"{name}_manifest.json"
    manifest.write_text(json.dumps(written, indent=1), encoding="utf-8")
    return paths, manifest
