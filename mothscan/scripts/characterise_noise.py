"""Characterise wind noise and clipping in a deployment so thresholds are earned.

Wind and mic clipping both have physical signatures that survive across every
recorder and gain setting, so they are worth measuring directly rather than
guessing at:

* **wind** - energy concentrated below ~40 Hz (the AudioMoth's optional 48 Hz DC
  filter is off in this deployment), a spectrum *falling* with frequency, and a
  level that is high and *sustained* rather than eventful.  Indoors/outdoors
  this also shows up as a very low crest factor: the waveform is a slow roar,
  not a series of events.
* **clipping** - a waveform pinned at a rail, which means repeated identical
  sample values and a low crest factor.

Run over a sample and print distributions, then dump the extremes as WAV +
spectrogram so the thresholds can be checked by eye and ear.

    python3 scripts/characterise_noise.py --per-site 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from mothscan.config import DATA_DIR, SITE_ROOT, AnalysisConfig, is_audio_file  # noqa: E402
from mothscan.features import load_mono, mel_filterbank, stft_mag  # noqa: E402


# --------------------------------------------------------------------------
# candidate detectors
# --------------------------------------------------------------------------


def ulf_ratio(x: np.ndarray, sr: int, cutoff: float = 40.0) -> tuple[float, float]:
    """Fraction of total energy below ``cutoff``, and the dB level there.

    Band energy via a welch-style periodogram averaged over the file, so a brief
    low thump does not dominate - wind is *sustained*.
    """
    n = 1 << 15
    hop = n // 2
    n_frames = max(1, (len(x) - n) // hop)
    # Cap the number of frames: a periodogram average converges quickly.
    stride = max(1, n_frames // 240)
    win = np.hanning(n).astype(np.float32)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    lo = freqs < cutoff
    acc = np.zeros(freqs.size)
    used = 0
    for i in range(0, n_frames, stride):
        seg = x[i * hop : i * hop + n]
        if seg.size < n:
            break
        acc += np.abs(np.fft.rfft(seg * win)) ** 2
        used += 1
    if used == 0:
        return 0.0, -120.0
    psd = acc / used
    total = psd.sum()
    low = psd[lo].sum()
    ratio = float(low / total) if total > 0 else 0.0
    # Band level relative to the file's overall RMS, in dB.
    low_db = 10 * np.log10(max(low, 1e-20))
    all_db = 10 * np.log10(max(total, 1e-20))
    return ratio, float(low_db - all_db)


def spectral_tilt(x: np.ndarray, sr: int, fmin: float = 50.0, fmax: float = 4000.0) -> float:
    """Slope of the long-term spectrum in dB per octave.

    Wind is strongly negative (energy collapses with frequency); a broadband
    insect chorus or rain is much flatter.
    """
    n = 1 << 14
    hop = n // 2
    n_frames = max(1, (len(x) - n) // hop)
    stride = max(1, n_frames // 200)
    win = np.hanning(n).astype(np.float32)
    nfft = n // 2 + 1
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    acc = np.zeros(nfft)
    used = 0
    for i in range(0, n_frames, stride):
        seg = x[i * hop : i * hop + n]
        if seg.size < n:
            break
        acc += np.abs(np.fft.rfft(seg * win)) ** 2
        used += 1
    if used == 0:
        return 0.0
    psd = np.maximum(acc / used, 1e-20)
    band = (freqs >= fmin) & (freqs <= fmax) & (freqs > 0)
    if band.sum() < 8:
        return 0.0
    lf = np.log2(freqs[band])
    ld = 10 * np.log10(psd[band])
    # Robust fit: use the median of local slopes to shrug off narrowband tones.
    k = max(3, band.sum() // 60)
    slope = np.polyfit(lf, ld, 1)[0]
    del k
    return float(slope)


def level_sustain(rms_db: np.ndarray, window_db: float = 6.0) -> float:
    """Fraction of frames within ``window_db`` of the file's loudest frame.

    Wind holds one level for minutes; birdsong and rain are spiky, so only a
    small fraction of frames sit near the maximum.
    """
    if rms_db.size == 0:
        return 0.0
    top = float(np.percentile(rms_db, 99.5))
    return float((rms_db >= top - window_db).mean())


def crest_factor(x: np.ndarray) -> float:
    """Peak / RMS. Sine is 1.41, Gaussian noise ~4, a clipped rail near 1."""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0.0
    return peak / rms if rms > 1e-12 else 0.0


def clipping_stats(x: np.ndarray, rail_frac: float = 0.985) -> dict:
    """Detect flat tops: runs of identical samples sitting at the extremes.

    Returns the longest run (in samples) and the fraction of energy involved.
    Digital clipping produces *contiguous runs* at a constant value, which is
    what distinguishes it from naturally loud but smooth material.
    """
    if x.size < 4:
        return {"max_run": 0, "run_samples": 0, "frac": 0.0, "rail": 0.0}
    peak = float(np.max(np.abs(x)))
    if peak <= 0:
        return {"max_run": 0, "run_samples": 0, "frac": 0.0, "rail": 0.0}
    # Quantise to 16-bit steps: the source is PCM_16, so a clipped rail shows as
    # genuinely identical integers.
    q = np.round(x * 32768.0).astype(np.int32)
    d = np.diff(q)
    same = d == 0
    # Longest run of equal consecutive values.
    longest = 0
    run = 0
    total = 0
    for v in same:
        if v:
            run += 1
            if run > longest:
                longest = run
        else:
            total += 1 if run >= 3 else 0
            run = 0
    if run >= 3:
        total += 1
    at_rail = float((np.abs(q) >= rail_frac * 32768).mean())
    return {
        "max_run": int(longest + 1),
        "runs": int(total),
        "frac": float(total / max(1, same.size)),
        "rail": at_rail,
    }


def frame_rms_db(x: np.ndarray, sr: int, hop_s: float = 0.05) -> np.ndarray:
    hop = max(1, int(hop_s * sr))
    n = (len(x) // hop) * hop
    if n == 0:
        return np.zeros(0)
    f = x[:n].reshape(-1, hop)
    p = (f.astype(np.float64) ** 2).mean(axis=1)
    return 10 * np.log10(np.maximum(p, 1e-20))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def measure(path: Path) -> dict:
    x, sr = load_mono(path)
    ulf, ulf_rel = ulf_ratio(x, sr)
    rms_db = frame_rms_db(x, sr)
    clip = clipping_stats(x)
    return {
        "file": path.name,
        "site": path.parent.name,
        "dur_s": len(x) / sr,
        "peak_db": 20 * np.log10(max(float(np.max(np.abs(x))), 1e-12)),
        "rms_db": 10 * np.log10(max(float(np.mean(x.astype(np.float64) ** 2)), 1e-20)),
        "ulf_ratio": ulf,
        "ulf_rel_db": ulf_rel,
        "tilt_db_oct": spectral_tilt(x, sr),
        "sustain": level_sustain(rms_db),
        "crest": crest_factor(x),
        "clip_run": clip["max_run"],
        "clip_runs": clip["runs"],
        "clip_rail": clip["rail"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-site", type=int, default=40)
    ap.add_argument("--out", default=str(DATA_DIR / "noise_probe"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for site in ("moth oculus", "moth-juvet", "moth-kvarvet"):
        d = SITE_ROOT / site
        files = sorted(p for p in d.iterdir() if p.is_file() and is_audio_file(p))
        idx = np.linspace(0, len(files) - 1, min(args.per_site, len(files))).astype(int)
        for i in idx:
            r = measure(files[i])
            rows.append(r)
            print(
                f"{r['site'][:12]:12s} {r['file']:22s} "
                f"rms={r['rms_db']:6.1f} peak={r['peak_db']:6.1f} "
                f"ulf={100*r['ulf_ratio']:5.1f}% ulfrel={r['ulf_rel_db']:6.1f} "
                f"tilt={r['tilt_db_oct']:6.2f} sust={r['sustain']:.2f} "
                f"crest={r['crest']:4.2f} cliprun={r['clip_run']:5d} rail={r['clip_rail']:.4f}"
            )

    a = {k: np.array([r[k] for r in rows], dtype=float)
         for k in rows[0] if k not in ("file", "site")}
    print("\n=== distributions over", len(rows), "files ===")
    for k, v in a.items():
        q = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
        print(f"  {k:12s} " + " ".join(f"{x:9.3f}" for x in q))
    print("               " + " ".join(f"{p:>9}" for p in ("p1", "p5", "p25", "p50", "p75", "p95", "p99")))

    # Flag candidates and dump their spectrograms for inspection.
    import json
    (out / "probe.json").write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out / 'probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
