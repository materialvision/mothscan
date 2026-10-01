"""Feature extraction for finding acoustically interesting points in AudioMoth files.

The representation is deliberately cheap and *self-normalising*: every statistic
is measured against the same recording's own noise floor, because absolute level
varies enormously between a quiet Nordic forest night and a wind/rain hour.

Pipeline
--------
1.  ``load_mono``      -> float32 mono samples at the native rate.
2.  ``mel_spectrogram``-> log-mel magnitudes, frames at 5.33 ms (hop 256).
3.  ``novelty_curve``  -> half-wave-rectified spectral flux with a per-band
    noise-floor subtraction, so a *change* in any band counts, not raw energy.
4.  ``envelope_curves``-> fast (10 ms) and slow (400 ms) envelopes + attack.
5.  ``frame_flatness`` -> per-frame spectral flatness (broadband vs tonal).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import SAMPLE_RATE, AnalysisConfig


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def load_mono(path: str | Path, cfg: AnalysisConfig | None = None) -> tuple[np.ndarray, int]:
    """Read a WAV as float32 mono.

    AudioMoth "Basic" writes mono, but a multi-channel file would be averaged
    rather than truncated so that no acoustic content is silently dropped.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    return np.ascontiguousarray(mono, dtype=np.float32), int(sr)


# --------------------------------------------------------------------------
# Mel filterbank (self-contained; avoids a librosa dependency)
# --------------------------------------------------------------------------


def hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float
) -> np.ndarray:
    """Slaney-style triangular mel filterbank, shape (n_mels, n_fft//2 + 1)."""
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sr / 2.0, n_bins)

    mel_min, mel_max = hz_to_mel(fmin), hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.asarray(mel_to_hz(mel_points))

    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for i in range(n_mels):
        left, centre, right = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        if right <= left:
            continue
        rising = (fft_freqs - left) / max(centre - left, 1e-9)
        falling = (right - fft_freqs) / max(right - centre, 1e-9)
        fb[i] = np.clip(np.minimum(rising, falling), 0.0, None)
    # Slaney area normalisation keeps quiet high bands comparable to low bands.
    enorm = 2.0 / np.maximum(hz_points[2 : n_mels + 2] - hz_points[:n_mels], 1e-9)
    fb *= enorm[:, None]
    return fb


# --------------------------------------------------------------------------
# Core representation
# --------------------------------------------------------------------------


def stft_mag(
    x: np.ndarray, cfg: AnalysisConfig, window: np.ndarray | None = None
) -> np.ndarray:
    """Magnitude STFT, shape (n_bins, n_frames), float32."""
    n_fft, hop = cfg.n_fft, cfg.hop
    if window is None:
        window = np.hanning(cfg.win_length).astype(np.float32)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))

    n_frames = 1 + (len(x) - n_fft) // hop
    # Strided view instead of a copy-heavy fancy-index: (n_frames, n_fft)
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, n_fft),
        strides=(x.strides[0] * hop, x.strides[0]),
        writeable=False,
    )
    spec = np.fft.rfft(frames * window, axis=1, n=n_fft)
    del frames
    mag = np.abs(spec)
    del spec
    return np.ascontiguousarray(mag.T, dtype=np.float32)


def mel_spectrogram(
    x: np.ndarray, sr: int, cfg: AnalysisConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Return (log_mel, linear_mel) with shape (n_mels, n_frames)."""
    mag = stft_mag(x, cfg)
    power = mag * mag
    del mag
    fb = mel_filterbank(sr, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax)
    lin = fb @ power                     # (n_mels, n_frames)
    del power
    logmel = np.log(lin + cfg.log_eps).astype(np.float32)
    return logmel, lin


def novelty_curve(logmel: np.ndarray, cfg: AnalysisConfig) -> np.ndarray:
    """Spectral flux with per-band adaptive noise floor, half-wave rectified.

    Subtracting a per-band percentile of the flux distribution means stationary
    broadband noise (rain, wind, insect drone) contributes ~0 novelty, while any
    band that *changes* relative to that floor produces a positive excursion.
    This is what makes the detector work in both a silent forest and a downpour.
    """
    # Difference along time, positive part only.
    diff = np.diff(logmel, axis=1, prepend=logmel[:, :1])
    pos = np.maximum(diff, 0.0).astype(np.float32)
    del diff

    # Per-band noise floor from a percentile of the positive flux.
    floor = np.percentile(pos, cfg.novelty_noise_pct, axis=1, keepdims=True)
    flux = np.maximum(pos - cfg.novelty_noise_scale * floor, 0.0)
    del pos

    nov = flux.sum(axis=0)
    del flux

    if cfg.novelty_smooth_frames > 1:
        nov = _moving_average(nov, cfg.novelty_smooth_frames)
    return nov.astype(np.float32)


def _moving_average(x: np.ndarray, width: int) -> np.ndarray:
    """Centred moving average, edge-padded, O(n) via cumulative sum."""
    width = int(width)
    if width <= 1:
        return x
    pad = width // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    csum = np.concatenate(([0.0], np.cumsum(padded, dtype=np.float64)))
    out = (csum[width:] - csum[:-width]) / width
    return out[: len(x)]


#: Public alias used by the scorer for its event-scale smoothing.
moving_average = _moving_average


def envelope_curves(
    mag: np.ndarray, sr: int, cfg: AnalysisConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Envelope in a mid band, at two time constants, plus the attack signal.

    Takes an already-computed magnitude STFT (n_bins, n_frames) and sums true
    FFT-bin energy inside ``cfg.env_band_hz``, so the band limits are exact
    rather than mel-quantised.  Returns (env_fast, env_slow, attack) in linear
    amplitude units at the STFT frame rate.
    """
    frame_rate = sr / cfg.hop
    fft_freqs = np.linspace(0.0, sr / 2.0, cfg.n_fft // 2 + 1)
    lo, hi = cfg.env_band_hz
    in_band = (fft_freqs >= lo) & (fft_freqs <= hi)
    n_frames = mag.shape[1]

    if in_band.any():
        sub = mag[in_band]
        band_energy = np.einsum("ij,ij->j", sub, sub)
    else:
        band_energy = np.einsum("ij,ij->j", mag, mag)
    env = np.sqrt(np.maximum(band_energy, 0.0))
    if env.shape[0] != n_frames:
        env = (
            env[:n_frames]
            if env.shape[0] > n_frames
            else np.pad(env, (0, n_frames - env.shape[0]), mode="edge")
        )

    fast_w = max(1, int(round(cfg.env_fast_ms / 1000.0 * frame_rate)))
    slow_w = max(1, int(round(cfg.env_slow_ms / 1000.0 * frame_rate)))
    env_fast = _moving_average(env, fast_w)
    env_slow = _moving_average(env, slow_w)

    # Attack = how far the fast envelope jumps above the recent background.
    attack = np.maximum(env_fast - env_slow, 0.0)
    return (
        env_fast.astype(np.float32),
        env_slow.astype(np.float32),
        attack.astype(np.float32),
    )


def frame_flatness(lin_mel: np.ndarray) -> np.ndarray:
    """Per-frame spectral flatness in [0, 1]: geometric / arithmetic mean.

    ~1.0 = broadband/noise-like (rain, rustle, clicks), ~0.0 = tonal
    (birdsong, insect stridulation).
    """
    x = np.maximum(lin_mel, 1e-12)
    log_mean = np.log(x).mean(axis=0)
    arith = x.mean(axis=0)
    flat = np.exp(log_mean) / np.maximum(arith, 1e-12)
    return np.clip(flat, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# Contamination detectors: wind noise and microphone clipping
# --------------------------------------------------------------------------
#
# Wind and clipping are *physical* signatures, not content judgements, so they
# can be measured directly rather than inferred from "this sounds wrong".
#
# Wind on an unprotected AudioMoth is a slow, very low frequency roar.  Three
# things follow from that, and all three are measured here:
#
#   * almost all of its energy is below ~40 Hz.  This deployment has the
#     optional 48 Hz DC-blocking filter DISABLED (see CONFIG.TXT), so the band
#     is present in the data and the ratio is very informative.
#   * its long-term spectrum falls steeply with frequency, unlike rain or an
#     insect chorus, which are comparatively flat.
#   * it is *sustained*: the level sits near its maximum for minutes, whereas
#     birdsong, rain and wingbeats are eventful (high peak-to-median spread).
#
# Clipping is a waveform pinned at a rail, which produces contiguous runs of
# identical PCM values plus a low crest factor.  Quantisation noise produces
# runs of 2-3 samples too, so a run of >= 8 is used as the discriminating
# threshold and the run length is kept, not just a boolean.


def low_band_shares(
    x: np.ndarray, sr: int, cutoffs: tuple[float, ...] = (40.0, 100.0, 200.0)
) -> dict[str, float]:
    """Fraction of total energy below each cutoff, from one periodogram pass.

    The sub-100 Hz share is the single most useful wind discriminant: across
    this deployment a clean hour sits near 2-3 %, an ordinary eventful hour up
    to ~20 %, while a wind-dominated hour exceeds ~30 % because the rumble
    occupies the whole band rather than one narrow resonance.
    """
    n = 1 << 15
    hop = n // 2
    if len(x) < n:
        return {f"low{int(c)}": 0.0 for c in cutoffs}
    n_frames = (len(x) - n) // hop
    stride = max(1, n_frames // 240)
    win = np.hanning(n).astype(np.float32)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    acc = np.zeros(freqs.size)
    used = 0
    for i in range(0, n_frames, stride):
        seg = x[i * hop : i * hop + n]
        acc += np.abs(np.fft.rfft(seg * win)) ** 2
        used += 1
    if used == 0:
        return {f"low{int(c)}": 0.0 for c in cutoffs}
    psd = acc / used
    total = float(psd.sum())
    if total <= 0:
        return {f"low{int(c)}": 0.0 for c in cutoffs}
    return {
        f"low{int(c)}": float(psd[freqs < c].sum() / total) for c in cutoffs
    }


def ulf_ratio(x: np.ndarray, sr: int, cutoff: float = 40.0) -> tuple[float, float]:
    """Fraction of energy below ``cutoff`` Hz, and that band's level vs. total.

    A Welch-style periodogram average over the file, so a single low thump does
    not dominate - wind is sustained.
    """
    n = 1 << 15
    hop = n // 2
    if len(x) < n:
        return 0.0, -120.0
    n_frames = (len(x) - n) // hop
    stride = max(1, n_frames // 240)
    win = np.hanning(n).astype(np.float32)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    lo = freqs < cutoff
    acc = np.zeros(freqs.size)
    used = 0
    for i in range(0, n_frames, stride):
        seg = x[i * hop : i * hop + n]
        acc += np.abs(np.fft.rfft(seg * win)) ** 2
        used += 1
    if used == 0:
        return 0.0, -120.0
    psd = acc / used
    total = float(psd.sum())
    low = float(psd[lo].sum())
    ratio = low / total if total > 0 else 0.0
    return float(ratio), float(
        10 * np.log10(max(low, 1e-20)) - 10 * np.log10(max(total, 1e-20))
    )


def spectral_tilt(
    x: np.ndarray, sr: int, fmin: float = 50.0, fmax: float = 4000.0
) -> float:
    """Slope of the long-term spectrum in dB per octave over ``fmin..fmax``.

    Steeply negative for wind, near zero or positive for broadband insect
    chorus.  Uses a least-squares fit over log-frequency, which is robust enough
    to ignore a few narrowband tones.
    """
    n = 1 << 14
    hop = n // 2
    if len(x) < n:
        return 0.0
    n_frames = (len(x) - n) // hop
    stride = max(1, n_frames // 200)
    win = np.hanning(n).astype(np.float32)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    acc = np.zeros(freqs.size)
    used = 0
    for i in range(0, n_frames, stride):
        seg = x[i * hop : i * hop + n]
        acc += np.abs(np.fft.rfft(seg * win)) ** 2
        used += 1
    if used == 0:
        return 0.0
    psd = np.maximum(acc / used, 1e-20)
    band = (freqs >= fmin) & (freqs <= fmax) & (freqs > 0)
    if band.sum() < 8:
        return 0.0
    return float(np.polyfit(np.log2(freqs[band]), 10 * np.log10(psd[band]), 1)[0])


def level_sustain(rms_db: np.ndarray, window_db: float = 6.0) -> float:
    """Fraction of frames within ``window_db`` of the file's loudest frame."""
    if rms_db.size == 0:
        return 0.0
    top = float(np.percentile(rms_db, 99.5))
    return float((rms_db >= top - window_db).mean())


def clip_frames(
    x: np.ndarray, n_frames: int, hop: int, rail_frac: float = 0.985
) -> np.ndarray:
    """Fraction of samples at the converter rail, per analysis frame.

    Clipping is *localised*: a single loud wingbeat can pin a few samples while
    the rest of the hour is pristine.  So the penalty belongs on the windows
    that actually contain the rail, not on the whole file - which is exactly
    what this per-frame curve allows.
    """
    out = np.zeros(n_frames, dtype=np.float32)
    if x.size == 0 or n_frames <= 0:
        return out
    rail = (np.abs(x) >= rail_frac).astype(np.float32)
    if not rail.any():
        return out
    usable = min(x.size, n_frames * hop)
    counts = rail[: n_frames * hop].reshape(n_frames, hop).sum(axis=1) if usable >= n_frames * hop else None
    if counts is None:
        # Short file: bin whatever there is.
        for i in range(n_frames):
            lo = i * hop
            hi = min(lo + hop, rail.size)
            if hi > lo:
                out[i] = float(rail[lo:hi].sum())
        return out / float(hop)
    return (counts / float(hop)).astype(np.float32)


def clipping_stats(x: np.ndarray, min_run: int = 8) -> tuple[int, int, float]:
    """Detect flat-topped runs of identical PCM values.

    Returns ``(longest_run, n_runs, fraction_of_samples_in_runs)``.  Values are
    quantised to 16-bit first because that is what the recorder wrote; a genuine
    clip shows as *identical integers*, which smooth audio never produces over
    more than a couple of samples.
    """
    if x.size < 4:
        return 0, 0, 0.0
    q = np.round(x.astype(np.float64) * 32767.0)
    same = np.diff(q) == 0
    if not same.any():
        return 1, 0, 0.0
    # Run-length encode the boolean mask.
    idx = np.flatnonzero(np.diff(np.concatenate(([0], same.view(np.int8), [0]))))
    starts, ends = idx[0::2], idx[1::2]
    lengths = ends - starts + 1
    long_mask = lengths >= min_run
    n_runs = int(long_mask.sum())
    in_runs = int(lengths[long_mask].sum())
    return int(lengths.max()), n_runs, float(in_runs / x.size)


# --------------------------------------------------------------------------
# Bundle
# --------------------------------------------------------------------------


@dataclass
class FileFeatures:
    """All frame-rate signals needed by the scorer for one recording."""

    novelty: np.ndarray        # spectral-change density
    env_fast: np.ndarray       # fast amplitude envelope
    env_slow: np.ndarray       # background amplitude envelope
    attack: np.ndarray         # envelope attack (fast - slow)
    flatness: np.ndarray       # broadband-ness
    rms_db: np.ndarray         # absolute level per frame (dBFS)
    frame_rate: float
    duration_s: float
    noise_floor_db: float      # 5th percentile of frame RMS
    peak: float = 0.0          # true sample peak, linear 0..1
    peak_db: float = -120.0    # 20*log10(peak), dBFS

    # --- contamination indicators (wind / clipping) ---
    #: Fraction of energy below 40 Hz.
    ulf_ratio: float = 0.0
    #: Fraction of energy below 100 Hz - the primary wind discriminant.
    low100: float = 0.0
    #: Fraction of energy below 200 Hz, for context.
    low200: float = 0.0
    #: Level of that band relative to total, dB (negative).
    ulf_rel_db: float = -120.0
    #: Long-term spectral slope, dB/octave over 50 Hz - 4 kHz.
    tilt_db_oct: float = 0.0
    #: Fraction of frames within 6 dB of the loudest frame.
    sustain: float = 0.0
    #: Peak / RMS of the whole file.
    crest: float = 0.0
    #: Longest run of identical PCM samples (>=8 suggests a clipped rail).
    clip_run: int = 0
    #: Number of such runs.
    clip_runs: int = 0
    #: Fraction of samples inside those runs.
    clip_frac: float = 0.0
    #: Per-frame fraction of samples at the rail (window-level clip penalty).
    clip_frame: np.ndarray | None = None

    def n_frames(self) -> int:
        return int(self.novelty.shape[0])


def extract_features(path: str | Path, cfg: AnalysisConfig) -> FileFeatures:
    """Full feature pass over one audio file.

    The STFT is computed exactly once; the mel projection, the band envelope and
    the flatness measure are all derived from that single magnitude array.
    """
    x, sr = load_mono(path, cfg)
    duration_s = len(x) / sr

    mag = stft_mag(x, cfg)
    n_frames = mag.shape[1]
    power = mag * mag

    # --- mel projection -> novelty -------------------------------------
    fb = mel_filterbank(sr, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax)
    lin = np.ascontiguousarray(fb @ power, dtype=np.float32)
    logmel = np.log(lin + cfg.log_eps).astype(np.float32)
    nov = novelty_curve(logmel, cfg)
    del logmel

    flatness = frame_flatness(lin)
    del lin

    # --- band envelope straight from the true FFT bins ------------------
    env_fast, env_slow, attack = envelope_curves(mag, sr, cfg)
    del mag, power

    # --- absolute level per frame --------------------------------------
    frame_len = cfg.hop
    usable = n_frames * frame_len
    if usable > len(x):
        usable = (len(x) // frame_len) * frame_len
    p = np.concatenate(([0.0], np.cumsum(x[:usable].astype(np.float64) ** 2)))
    mean_power = (p[frame_len::frame_len] - p[:-frame_len:frame_len]) / frame_len
    mean_power = mean_power[:n_frames]
    if mean_power.shape[0] < n_frames:
        mean_power = np.pad(mean_power, (0, n_frames - mean_power.shape[0]), mode="edge")
    rms_db = (10.0 * np.log10(np.maximum(mean_power, 1e-20))).astype(np.float32)
    del p, mean_power

    noise_floor_db = float(np.percentile(rms_db, 5.0))
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    peak_db = 20.0 * np.log10(peak) if peak > 0 else -120.0

    # ---- contamination indicators -------------------------------------
    shares = low_band_shares(x, sr)
    ulf, ulf_rel = ulf_ratio(x, sr)
    tilt = spectral_tilt(x, sr)
    sustain = level_sustain(rms_db)
    clip_run, clip_runs, clip_frac = clipping_stats(x)
    clip_frame = clip_frames(x, n_frames, cfg.hop)
    rms_lin = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0.0
    crest = peak / rms_lin if rms_lin > 1e-12 else 0.0

    return FileFeatures(
        novelty=nov,
        env_fast=env_fast,
        env_slow=env_slow,
        attack=attack,
        flatness=flatness,
        rms_db=rms_db,
        frame_rate=sr / cfg.hop,
        duration_s=duration_s,
        noise_floor_db=noise_floor_db,
        peak=peak,
        peak_db=float(peak_db),
        ulf_ratio=ulf,
        low100=shares.get("low100", 0.0),
        low200=shares.get("low200", 0.0),
        ulf_rel_db=ulf_rel,
        tilt_db_oct=tilt,
        sustain=sustain,
        crest=crest,
        clip_run=clip_run,
        clip_runs=clip_runs,
        clip_frac=clip_frac,
        clip_frame=clip_frame,
    )
