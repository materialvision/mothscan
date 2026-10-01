"""Diagnostic: verify that the detector really lands on interesting audio.

Produces, for a set of sample files:
  * a rendered spectrogram timeline of the whole 2-minute file
  * the score curve drawn underneath it
  * a marker at the chosen moment
  * a WAV of the chosen 0.5 s and, for comparison, of the loudest 0.5 s
  * a WAV stitching chosen / loudest / random picks back to back

Usage:
  python3 scripts/inspect.py --n 8 --out data/diag
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from mothscan.config import DATA_DIR, SITE_ROOT, AnalysisConfig, is_audio_file  # noqa: E402
from mothscan.features import extract_features, load_mono  # noqa: E402
from mothscan.scoring import EventScorer  # noqa: E402


def mel_db_image(x: np.ndarray, sr: int, cfg: AnalysisConfig, n_time: int = 900):
    """Downsampled dB mel spectrogram for plotting, (n_mels, n_time)."""
    n_fft, hop = 1024, max(1, len(x) // (n_time * 4))
    win = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (len(x) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, n_fft), strides=(x.strides[0] * hop, x.strides[0])
    )
    spec = np.abs(np.fft.rfft(frames * win, axis=1))
    from mothscan.features import mel_filterbank

    fb = mel_filterbank(sr, n_fft, 64, 40.0, 20_000.0)
    mel = fb @ (spec * spec).T
    db = 10.0 * np.log10(np.maximum(mel, 1e-12))
    # Downsample time axis by block-mean to ~n_time columns.
    if db.shape[1] > n_time:
        trim = (db.shape[1] // n_time) * n_time
        db = db[:, :trim].reshape(db.shape[0], n_time, -1).mean(axis=2)
    return db


def render_plot(
    path: Path, out_png: Path, cfg: AnalysisConfig, best, alts, score_t: np.ndarray,
    score_y: np.ndarray
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, sr = load_mono(path, cfg)
    db = mel_db_image(x, sr, cfg)
    t_db = np.linspace(0, len(x) / sr, db.shape[1])

    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.4, 1.4]})

    ax = axes[0]
    # A wide, per-file colour range: these recordings are quiet and the
    # biological events live 10-35 dB above the floor.
    vmin = float(np.percentile(db, 20))
    vmax = float(np.percentile(db, 99.9))
    im = ax.pcolormesh(t_db, np.arange(db.shape[0]), db, shading="auto", cmap="magma",
                       vmin=vmin, vmax=vmax)
    ax.axvline(best.event_s, color="#f43f5e", lw=1.6, ls=":",
               label=f"event {best.event_s:.3f}s")
    ax.axvspan(best.start_s, best.start_s + cfg.scan_seconds, color="#22d3ee",
               alpha=0.28, label=f"cut {best.start_s:.3f}s +{cfg.scan_seconds}s")
    for i, a in enumerate(alts):
        ax.axvspan(a.start_s, a.start_s + cfg.scan_seconds, color="#a3e635", alpha=0.13)
        ax.axvline(a.event_s, color="#a3e635", lw=0.9, ls="--", alpha=0.8,
                   label="alternatives" if i == 0 else None)
    ax.set_ylabel("mel band")
    ax.set_title(
        f"{path.name}   cut={best.start_s:.3f}s  event={best.event_s:.3f}s  "
        f"score={best.score:.2f}  peak={best.peak_db:.1f} dBFS  "
        f"{'gate OK' if best.passes_gate else 'BELOW GATE'}"
    )
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    fig.colorbar(im, ax=ax, pad=0.01, label=f"dB ({vmin:.0f}..{vmax:.0f})")

    ax2 = axes[1]
    ax2.plot(score_t, score_y, lw=0.9, color="#f97316")
    ax2.axvline(best.event_s, color="#f43f5e", lw=1.6, ls=":")
    ax2.axvspan(best.start_s, best.start_s + cfg.scan_seconds, color="#22d3ee",
                alpha=0.28)
    for a in alts:
        ax2.axvline(a.event_s, color="#a3e635", lw=0.8, ls="--", alpha=0.7)
    ax2.set_ylabel("score")
    ax2.set_title("hybrid event score (at candidate event position)")

    ax3 = axes[2]
    t = np.arange(len(x)) / sr
    step = max(1, len(x) // 20_000)
    ax3.plot(t[::step], x[::step], lw=0.5, color="#94a3b8")
    ax3.axvspan(best.start_s, best.start_s + cfg.scan_seconds, color="#22d3ee",
                alpha=0.28)
    ax3.axvline(best.event_s, color="#f43f5e", lw=1.6, ls=":")
    ax3.set_ylabel("amplitude")
    ax3.set_xlabel("time (s)")
    ax3.set_xlim(0, len(x) / sr)

    fig.tight_layout()
    fig.savefig(out_png, dpi=88)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default=str(DATA_DIR / "diag"))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--w-tonality", type=float, default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = AnalysisConfig()
    if args.w_tonality is not None:
        object.__setattr__(cfg, "w_tonality", args.w_tonality)

    files: list[Path] = []
    for site in ("moth oculus", "moth-juvet", "moth-kvarvet"):
        allf = sorted(
            p for p in (SITE_ROOT / site).iterdir() if p.is_file() and is_audio_file(p)
        )
        idx = np.linspace(0, len(allf) - 1, max(1, args.n // 3)).astype(int)
        files += [allf[i] for i in idx]

    rng = np.random.default_rng(args.seed)
    print(f"diagnostics -> {out}")
    for f in files:
        feats = extract_features(f, cfg)
        scorer = EventScorer(feats, cfg)
        best, alts = scorer.find_best()
        score_t, score_y = scorer.score_series()

        png = out / f"{f.stem}_{f.parent.name.replace(' ', '-')}_diag.png"
        render_plot(f, png, cfg, best, alts, score_t, score_y)

        # Audio comparison montage: picked | loudest | random.
        x, sr = load_mono(f, cfg)
        n_cut = int(round(cfg.scan_seconds * sr))
        starts = {
            "picked": int(round(best.start_s * sr)),
            "loudest": int(np.argmax(feats.rms_db)),
            "random": int(rng.integers(0, max(1, len(x) - n_cut))),
        }
        montage = []
        for name, s in starts.items():
            seg = x[s : s + n_cut]
            if len(seg) < n_cut:
                seg = np.pad(seg, (0, n_cut - len(seg)))
            montage += [seg, np.zeros(int(0.35 * sr), dtype=np.float32)]
        wav = out / f"{f.stem}_{f.parent.name.replace(' ', '-')}_compare.wav"
        sf.write(str(wav), np.concatenate(montage), sr)

        print(
            f"{f.name:26s} cut={best.start_s:7.3f}s event={best.event_s:7.3f}s score={best.score:5.2f} "
            f"peak={best.peak_db:6.1f} dB  gate={'OK ' if best.passes_gate else 'low'} "
            f"| loudest@{starts['loudest']/sr:6.2f}s  -> {png.name}"
        )
    print("\ncompare wav order: picked | loudest | random")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
